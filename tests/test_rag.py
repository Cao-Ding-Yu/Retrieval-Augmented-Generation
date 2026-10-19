"""RAG 纯逻辑的单元测试：提示词拼接 + 思考块剥离。

这两块都不碰外部服务，可以完全离线验证。
思考块部分还兼作**字符完整性守卫**：标签里的尖括号一旦被改写成 Unicode 近似字符，
``find()`` 就永远匹配不上、剥离会静默失效 —— 这种 bug 从代码上完全看不出来。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from rag_app.rag import (
    NO_CONTEXT_REPLY,
    ThinkingStripper,
    build_messages,
    format_context,
    strip_thinking,
)
from tests.conftest import make_hit

# 测试里也用字符码构造，与被测代码面对同样的编码风险
LT, GT = chr(60), chr(62)
OPEN = LT + "think" + GT
CLOSE = LT + "/think" + GT


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
class TestFormatContext:
    def test_empty_hits_returns_empty(self):
        assert format_context([], max_chars=1000) == ""

    def test_hits_are_numbered_from_one(self):
        """编号必须从 1 开始且连续，模型要靠它标注引用来源。"""
        text = format_context(
            [make_hit("a", text="内容甲"), make_hit("b", text="内容乙")],
            max_chars=1000,
        )

        assert "[1]" in text
        assert "[2]" in text
        assert "[0]" not in text

    def test_source_label_uses_basename_only(self):
        """只保留文件名 —— 完整路径又长又占 token。"""
        text = format_context(
            [make_hit("a", text="正文", source="C:\\projects\\docs\\manual.pdf")],
            max_chars=1000,
        )

        assert "manual.pdf" in text
        assert "C:\\projects" not in text

    def test_chunk_index_is_human_readable(self):
        """对用户展示时块序号从 1 开始（内部是 0 起）。"""
        text = format_context([make_hit("a", chunk_index=4)], max_chars=1000)
        assert "第 5 块" in text

    def test_later_hits_dropped_when_over_limit(self):
        """超长时从**末尾**丢弃：列表按相关性降序，末尾最不重要。"""
        text = format_context(
            [make_hit("a", text="A" * 300), make_hit("b", text="B" * 300),
             make_hit("c", text="C" * 300)],
            max_chars=700,
        )

        assert "AAA" in text
        assert "BBB" in text
        assert "CCC" not in text

    def test_single_oversized_hit_is_truncated_not_dropped(self):
        """第一条就超长时截断保留 —— 完全不给上下文更容易产生幻觉。"""
        text = format_context([make_hit("a", text="X" * 5000)], max_chars=200)

        assert text
        assert len(text) < 5000


class TestBuildMessages:
    def test_returns_system_and_human(self):
        messages = build_messages("问题", [make_hit("a")], max_chars=2000)

        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_system_prompt_forbids_prior_knowledge(self):
        """没有这条约束，模型会用预训练知识作答 —— 那就无法溯源了。"""
        messages = build_messages("问题", [make_hit("a")], max_chars=2000)

        assert "只使用" in messages[0].content
        assert "禁止编造" in messages[0].content

    def test_context_and_question_in_human_message(self):
        """资料与问题一起放 Human：本地小模型对 system 的遵循度偏弱。"""
        messages = build_messages("我的问题", [make_hit("a", text="关键事实")], max_chars=2000)
        body = messages[1].content

        assert "关键事实" in body
        assert "我的问题" in body
        assert "【参考资料】" in body

    def test_no_context_tells_model_not_to_guess(self):
        """检索为空时必须明确告知，并要求直接说无法回答。"""
        messages = build_messages("问题", [], max_chars=2000)
        body = messages[1].content

        assert "未返回任何相关内容" in body
        assert "不要使用你自己的知识" in body


# ---------------------------------------------------------------------------
# 思考块剥离
# ---------------------------------------------------------------------------
class TestTagCharacters:
    """守卫：标签必须是能匹配真实模型输出的纯 ASCII。"""

    def test_open_tag_is_plain_ascii(self):
        assert ThinkingStripper.OPEN_TAG == OPEN
        assert len(ThinkingStripper.OPEN_TAG) == 7
        assert ThinkingStripper.OPEN_TAG.isascii()

    def test_close_tag_is_plain_ascii(self):
        assert ThinkingStripper.CLOSE_TAG == CLOSE
        assert len(ThinkingStripper.CLOSE_TAG) == 8
        assert ThinkingStripper.CLOSE_TAG.isascii()


class TestStripping:
    def test_strips_complete_block(self):
        assert strip_thinking(f"{OPEN}推理过程{CLOSE}答案是混合检索。") == "答案是混合检索。"

    def test_plain_text_unchanged(self):
        text = "普通回答，包含 BM25 与 RRF。"
        assert strip_thinking(text) == text

    def test_text_before_and_after_preserved(self):
        assert strip_thinking(f"前缀{OPEN}推理{CLOSE}后缀") == "前缀后缀"

    def test_multiple_blocks_removed(self):
        assert strip_thinking(f"{OPEN}一{CLOSE}甲{OPEN}二{CLOSE}乙") == "甲乙"

    def test_disabled_passes_through(self):
        text = f"{OPEN}推理{CLOSE}答案"
        assert strip_thinking(text, enabled=False) == text

    def test_only_thinking_yields_empty(self):
        assert strip_thinking(f"{OPEN}纯推理{CLOSE}") == ""

    def test_citation_inside_thinking_is_dropped(self):
        """思考里的 [1] 会被前端误当成引用标记，必须一并丢弃。"""
        text = f"{OPEN}根据资料 [1] 与 [2] 推断…{CLOSE}结论是混合检索 [1]。"

        result = strip_thinking(text)

        assert result == "结论是混合检索 [1]。"
        assert "[2]" not in result


class TestStreamingAcrossChunks:
    """跨 chunk 切分 —— 这正是不能用正则的原因。"""

    def _run(self, chunks: list[str], enabled: bool = True) -> str:
        stripper = ThinkingStripper(enabled=enabled)
        return "".join(stripper.feed(c) for c in chunks) + stripper.flush()

    def test_open_tag_split_across_chunks(self):
        result = self._run(["答案", OPEN[:4], OPEN[4:], "隐藏", CLOSE, "可见"])

        assert result == "答案可见"
        assert "think" not in result

    def test_close_tag_split_across_chunks(self):
        result = self._run([OPEN + "推理", CLOSE[:5], CLOSE[5:], "正式答案"])

        assert result == "正式答案"
        assert "think" not in result

    def test_character_by_character_streaming(self):
        """每次只来一个字符 —— 最容易暴露状态机缺陷的场景。"""
        full = f"开头{OPEN}这段推理不应出现{CLOSE}结尾"

        result = self._run(list(full))

        assert result == "开头结尾"
        assert "推理" not in result

    def test_tail_is_flushed(self):
        """短输出会被缓冲几个字符，flush 时必须补发，否则末尾字符丢失。"""
        assert self._run(["短"]) == "短"
        assert self._run(["12345678"]) == "12345678"

    def test_unterminated_thinking_discarded(self):
        """模型被截断（没有结束标签）时，剩余内容属推理，应丢弃。"""
        assert self._run([OPEN + "还没想完"]) == ""

    def test_disabled_does_not_buffer(self):
        stripper = ThinkingStripper(enabled=False)
        assert stripper.feed("甲") == "甲"
        assert stripper.flush() == ""


@pytest.mark.parametrize(
    "text",
    ["", "无标签", OPEN + CLOSE, OPEN + "只有开始" + CLOSE + "结束",
     "嵌套" + OPEN + OPEN + "双开始" + CLOSE + "尾"],
)
def test_never_crashes(text: str):
    """任意输入都不该抛异常 —— 它处在流式主链路上，崩了会中断整个回答。"""
    assert isinstance(strip_thinking(text), str)


def test_no_context_reply_is_explicit():
    """兜底文案要说明"没检索到"，而不是假装回答了问题。"""
    assert "没有检索到" in NO_CONTEXT_REPLY
