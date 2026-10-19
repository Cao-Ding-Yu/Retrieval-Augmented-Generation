"""提示词构造的测试。

提示词是 RAG 里"最容易被随手改坏"的部分：改一句措辞不会报错，
但可能让模型开始用先验知识作答、或者不再标注来源。
这些测试把几条硬约束钉住：编号连续、上下文有上限、检索为空时有明确出口。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from rag_app.llm.prompts import NO_CONTEXT_REPLY, build_messages, format_context
from rag_app.models import RetrievedChunk
from tests.conftest import make_chunk


class TestFormatContext:
    """上下文拼接。"""

    def test_empty_chunks_returns_empty_string(self):
        assert format_context([], max_chars=1000) == ""

    def test_chunks_are_numbered_from_one(self):
        """编号必须从 1 开始且连续 —— 模型被要求用 [1][2] 标注来源，
        编号错位会让引用指向错误的文档，而且从答案里看不出来。"""
        chunks = [make_chunk("a", text="内容甲"), make_chunk("b", text="内容乙")]

        context = format_context(chunks, max_chars=1000)

        assert "[1]" in context
        assert "[2]" in context
        assert "[0]" not in context

    def test_source_label_uses_basename_only(self):
        """来源标签只保留文件名，完整路径又长又占 token。"""
        chunk = make_chunk("a", text="正文", source="C:\\projects\\docs\\manual.pdf")

        context = format_context([chunk], max_chars=1000)

        assert "manual.pdf" in context
        # 完整路径不应出现在上下文里
        assert "C:\\projects" not in context

    def test_posix_and_windows_paths_both_normalized(self):
        """Windows 反斜杠与 POSIX 斜杠都应正确取到文件名。"""
        windows = make_chunk("a", text="x", source="C:\\dir\\file.pdf")
        posix = make_chunk("b", text="y", source="/home/user/file.pdf")

        w_ctx = format_context([windows], max_chars=1000)
        p_ctx = format_context([posix], max_chars=1000)

        assert "file.pdf" in w_ctx
        assert "file.pdf" in p_ctx
        assert "\\" not in w_ctx.split("来源:")[1].split("·")[0]

    def test_chunk_index_is_human_readable(self):
        """分块序号对用户展示时应从 1 开始（内部是 0 起）。"""
        chunk = make_chunk("a", text="正文", chunk_index=4)

        context = format_context([chunk], max_chars=1000)

        assert "第 5 块" in context

    def test_content_is_preserved_verbatim(self):
        """正文内容不能被改写或转义 —— 它是模型的唯一依据。"""
        chunk = make_chunk("a", text="BM25 使用 IDF 加权，包含**特殊字符**与 emoji 🎯")

        context = format_context([chunk], max_chars=1000)

        assert "BM25 使用 IDF 加权" in context
        assert "**特殊字符**" in context
        assert "🎯" in context

    def test_later_chunks_dropped_when_exceeding_limit(self):
        """超出长度上限时应从**末尾**丢弃，保留最相关的靠前结果。

        列表本身按相关性降序，因此末尾是最不重要的内容，
        优先丢弃它们比平均截断每条更合理。
        """
        # 每条正文约 300 字符，限制 700 字符 -> 只能放下 2 条
        chunks = [
            make_chunk("a", text="A" * 300),
            make_chunk("b", text="B" * 300),
            make_chunk("c", text="C" * 300),
        ]

        context = format_context(chunks, max_chars=700)

        assert "AAA" in context
        assert "BBB" in context
        assert "CCC" not in context, "超限时应丢弃最不相关的末尾分块"

    def test_single_oversized_chunk_is_truncated_not_dropped(self):
        """单条上下文本身就超限时，应截断保留而非完全丢弃。

        完全丢弃会让模型没有任何依据，反而更容易产生幻觉；
        保留截断后的开头部分至少覆盖了该分块的主要信息。
        """
        chunks = [make_chunk("a", text="X" * 5000)]

        context = format_context(chunks, max_chars=200)

        assert context, "不应返回空上下文"
        assert "X" in context
        # 允许略微超出（截断逻辑按字符切，加上标签文字）
        assert len(context) < 5000


class TestBuildMessages:
    """消息列表构造。"""

    def test_returns_system_and_human_messages(self):
        chunks = [make_chunk("a", text="参考资料内容")]

        messages = build_messages("问题是什么", chunks, max_context_chars=2000)

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_system_prompt_forbids_prior_knowledge(self):
        """系统提示必须显式禁止使用先验知识。

        没有这条约束，模型会用预训练知识补全，答案看起来更完整，
        但无法溯源、也无法通过更新知识库来纠正。
        """
        messages = build_messages("问题", [make_chunk("a")], max_context_chars=2000)

        system_text = messages[0].content
        assert "只使用" in system_text and "参考资料" in system_text
        assert "不要使用你自己的先验知识" in system_text or "先验知识" in system_text

    def test_system_prompt_requires_citation(self):
        """系统提示必须要求标注来源编号，这是可验证性的基础。"""
        messages = build_messages("问题", [make_chunk("a")], max_context_chars=2000)

        assert "[1]、[2]" in messages[0].content

    def test_system_prompt_allows_saying_unknown(self):
        """必须给出"无法回答"的出口，否则模型会硬编答案。"""
        messages = build_messages("问题", [make_chunk("a")], max_context_chars=2000)

        assert "无法回答" in messages[0].content
        assert "禁止编造" in messages[0].content

    def test_context_and_question_in_human_message(self):
        """参考资料与问题都应出现在 user 消息里。

        放在 user 侧而不是 system 侧，是因为本地小模型对 system 的
        遵循度偏弱，把资料与问题放在一起关联更紧密。
        """
        chunks = [make_chunk("a", text="唯一的关键事实")]

        messages = build_messages("我的问题", chunks, max_context_chars=2000)
        human_text = messages[1].content

        assert "唯一的关键事实" in human_text
        assert "我的问题" in human_text
        assert "【参考资料】" in human_text
        assert "【问题】" in human_text

    def test_no_context_tells_model_not_to_answer_from_knowledge(self):
        """检索为空时必须明确告知模型"没有资料"，
        并要求它直接说无法回答，而不是自由发挥。"""
        messages = build_messages("问题", [], max_context_chars=2000)
        human_text = messages[1].content

        assert "未返回任何相关内容" in human_text
        assert "不要使用你自己的知识作答" in human_text

    def test_context_limit_is_respected(self):
        """构造消息时应把长度上限传到上下文拼接逻辑。"""
        chunks = [make_chunk("a", text="A" * 400), make_chunk("b", text="B" * 400)]

        messages = build_messages("问题", chunks, max_context_chars=500)
        human_text = messages[1].content

        assert "AAA" in human_text
        assert "BBB" not in human_text


class TestFallbackReply:
    """兜底文案。"""

    def test_fallback_mentions_no_context(self):
        """兜底文案应说明"没检索到"，而不是假装回答了问题。"""
        assert "没有检索到" in NO_CONTEXT_REPLY
