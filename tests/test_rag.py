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
    RAG,
    ThinkingStripper,
    build_messages,
    format_context,
    resolve_device,
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

    def test_pdf_hit_labelled_with_page_number(self):
        """PDF 要标页码，而不是分块序号 —— 用户是按页找原文的。"""
        hit = make_hit("a", text="正文")
        hit.page = 7
        hit.chunk_index = 42

        text = format_context([hit], max_chars=1000)

        assert "第 7 页" in text
        assert "第 43 块" not in text

    def test_non_pdf_hit_falls_back_to_chunk_index(self):
        hit = make_hit("a", text="正文", chunk_index=4)

        text = format_context([hit], max_chars=1000)

        assert "第 5 块" in text
        assert "页" not in text


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

    def test_only_thinking_yields_empty(self):
        assert strip_thinking(f"{OPEN}纯推理{CLOSE}") == ""

    def test_citation_inside_thinking_is_dropped(self):
        """思考里的 [1] 会被前端误当成引用标记，必须与正文分开。"""
        text = f"{OPEN}根据资料 [1] 与 [2] 推断…{CLOSE}结论是混合检索 [1]。"

        result = strip_thinking(text)

        assert result == "结论是混合检索 [1]。"
        assert "[2]" not in result


class TestThinkingExtraction:
    """思考内容不再被丢弃，而是作为独立部分返回，供前端展示。"""

    def test_feed_returns_answer_and_thinking(self):
        stripper = ThinkingStripper()

        answer, thinking = stripper.feed(f"开头{OPEN}推理内容{CLOSE}结尾")
        tail, _ = stripper.flush()

        # 末尾 "结尾"（2 字符）不足一个标签长度，会被缓冲到 flush
        assert answer + tail == "开头结尾"
        assert thinking == "推理内容"

    def test_thinking_does_not_leak_tags(self):
        stripper = ThinkingStripper()

        _answer, thinking = stripper.feed(f"{OPEN}推理{CLOSE}答案")

        assert "think" not in thinking
        assert thinking == "推理"

    def test_no_thinking_yields_empty_thinking(self):
        stripper = ThinkingStripper()

        answer, thinking = stripper.feed("普通回答")
        tail, _ = stripper.flush()

        assert thinking == ""
        assert answer + tail == "普通回答"

    def test_flush_returns_unclosed_thinking(self):
        """模型被截断时，未闭合的思考仍算思考 —— 可以展示，但绝不能进答案。"""
        stripper = ThinkingStripper()
        stripper.feed(f"{OPEN}还没想完")

        answer, thinking = stripper.flush()

        assert answer == ""
        assert thinking == "还没想完"

    def test_flush_returns_buffered_answer(self):
        stripper = ThinkingStripper()
        stripper.feed("短")

        answer, thinking = stripper.flush()

        assert answer == "短"
        assert thinking == ""


class TestStreamingAcrossChunks:
    """跨 chunk 切分 —— 这正是不能用正则的原因。"""

    def _run(self, chunks: list[str]) -> tuple[str, str]:
        """返回 (正文, 思考) 的完整拼接结果。"""
        stripper = ThinkingStripper()
        answers, thinkings = [], []
        for c in chunks:
            a, t = stripper.feed(c)
            answers.append(a)
            thinkings.append(t)
        a, t = stripper.flush()
        answers.append(a)
        thinkings.append(t)
        return "".join(answers), "".join(thinkings)

    def test_open_tag_split_across_chunks(self):
        answer, _ = self._run(["答案", OPEN[:4], OPEN[4:], "隐藏", CLOSE, "可见"])

        assert answer == "答案可见"
        assert "think" not in answer

    def test_close_tag_split_across_chunks(self):
        answer, _ = self._run([OPEN + "推理", CLOSE[:5], CLOSE[5:], "正式答案"])

        assert answer == "正式答案"
        assert "think" not in answer

    def test_character_by_character_streaming(self):
        """每次只来一个字符 —— 最容易暴露状态机缺陷的场景。"""
        full = f"开头{OPEN}这段推理不应出现在正文{CLOSE}结尾"

        answer, thinking = self._run(list(full))

        assert answer == "开头结尾"
        assert thinking == "这段推理不应出现在正文"

    def test_thinking_and_answer_are_disjoint(self):
        """同一段文本要么进正文、要么进思考，不能两边都有。"""
        full = f"甲{OPEN}乙{CLOSE}丙{OPEN}丁{CLOSE}戊"

        answer, thinking = self._run(list(full))

        assert answer == "甲丙戊"
        assert thinking == "乙丁"

    def test_tail_is_flushed(self):
        """短输出会被缓冲几个字符，flush 时必须补发，否则末尾字符丢失。"""
        assert self._run(["短"])[0] == "短"
        assert self._run(["12345678"])[0] == "12345678"

    def test_unterminated_thinking_kept_out_of_answer(self):
        """模型被截断（没有结束标签）时，剩余内容不能出现在答案里。"""
        answer, thinking = self._run([OPEN + "还没想完"])

        assert answer == ""
        assert thinking == "还没想完"


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


# ---------------------------------------------------------------------------
# RAG.astream 的事件拆分
# ---------------------------------------------------------------------------
class _FakeChunk:
    """模拟 ChatOllama 的流式 chunk。

    新版 Ollama 把思考放在独立字段，langchain_ollama 转存为
    ``additional_kwargs["reasoning_content"]``；``content`` 只含答案。
    """

    def __init__(self, content: str = "", reasoning: str | None = None) -> None:
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}


class _FakeLLM:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks

    async def astream(self, messages):
        for chunk in self._chunks:
            yield chunk


async def _collect(settings, chunks, show_thinking: bool = True):
    settings = settings.model_copy(update={"show_thinking": show_thinking})
    rag = RAG(settings)
    rag.llm = _FakeLLM(chunks)
    return [(kind, text) async for kind, text in rag.astream("问题", [])]


class TestAstreamSplitsThinking:
    """思考与答案必须分事件产出，且两个来源都要覆盖。"""

    async def test_reasoning_content_field_becomes_thinking_event(self, settings):
        """字段型思考（新版 Ollama）—— 之前只读 content，这类思考会被整个丢掉。"""
        events = await _collect(settings, [
            _FakeChunk(reasoning="先分析一下…"),
            _FakeChunk(content="最终答案"),
        ])

        assert ("thinking", "先分析一下…") in events
        assert ("answer", "最终答案") in events
        answer = "".join(t for k, t in events if k == "answer")
        assert "分析" not in answer

    async def test_inline_tags_also_split(self, settings):
        """标签型思考（老版本 Ollama）仍要能识别。"""
        full = f"{OPEN}标签里的推理{CLOSE}正文答案"

        events = await _collect(settings, [_FakeChunk(content=full)])

        thinking = "".join(t for k, t in events if k == "thinking")
        answer = "".join(t for k, t in events if k == "answer")
        assert thinking == "标签里的推理"
        assert answer == "正文答案"

    async def test_both_sources_are_merged(self, settings):
        """同一轮里既有字段、又有标签时，两者都要进入思考流。"""
        events = await _collect(settings, [
            _FakeChunk(reasoning="字段推理"),
            _FakeChunk(content=f"{OPEN}标签推理{CLOSE}答案"),
        ])

        thinking = "".join(t for k, t in events if k == "thinking")
        assert "字段推理" in thinking
        assert "标签推理" in thinking

    async def test_show_thinking_false_drops_thinking(self, settings):
        """关闭展示时，思考完全不产出，但答案不受影响。"""
        events = await _collect(
            settings,
            [_FakeChunk(reasoning="不该出现"), _FakeChunk(content="答案")],
            show_thinking=False,
        )

        assert [k for k, _ in events] == ["answer"]
        assert events[0][1] == "答案"

    async def test_no_thinking_yields_only_answer(self, settings):
        events = await _collect(settings, [_FakeChunk(content="纯答案")])

        assert [k for k, _ in events] == ["answer"]


# ---------------------------------------------------------------------------
# 设备解析
# ---------------------------------------------------------------------------
class TestResolveDevice:
    def test_cpu_is_always_available(self):
        assert resolve_device("cpu") == "cpu"

    def test_auto_returns_a_usable_device(self):
        assert resolve_device("auto") in {"cpu", "cuda", "mps"}

    def test_unknown_value_passes_through(self):
        """未知值原样透传，交给 sentence-transformers 报错，避免这里猜错。"""
        assert resolve_device("cuda:1") == "cuda:1"

    def test_blank_means_auto(self):
        assert resolve_device("") in {"cpu", "cuda", "mps"}

    def test_explicit_cuda_raises_when_unavailable(self):
        """显式要 cuda 但没有时，必须报错而不是悄悄退回 CPU。

        悄悄退回的话，用户会以为自己在用 GPU，性能问题根本查不出来。
        """
        import torch

        if torch.cuda.is_available():
            pytest.skip("本机 CUDA 可用，无法验证报错分支")

        with pytest.raises(RuntimeError, match="无法使用 CUDA"):
            resolve_device("cuda")
