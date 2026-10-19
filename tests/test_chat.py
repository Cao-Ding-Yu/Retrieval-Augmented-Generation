"""思考块剥离（ThinkingStripper）的测试。

这个模块的测试有额外价值：它同时充当**字符完整性守卫**。

背景：思考标签包含尖括号，而这类字符在文本往返（工具写入、编码转换、
复制粘贴）中容易被改写成 Unicode 近似字符。一旦被改写，``find()`` 就永远
匹配不上模型真实输出，剥离会**静默失效**——代码看起来完全正常，
只是思考内容照样漏给用户。所以第一组测试专门断言"标签必须是纯 ASCII"。
"""

from __future__ import annotations

import pytest

from rag_app.llm.chat import ThinkingStripper, strip_thinking

# 在测试里也用字符码构造，确保与被测代码面对同样的编码风险
LT = chr(60)
GT = chr(62)
OPEN = LT + "think" + GT
CLOSE = LT + "/think" + GT


class TestTagCharacters:
    """守卫测试：标签必须是可以匹配真实模型输出的纯 ASCII。"""

    def test_open_tag_is_plain_ascii(self):
        """开始标签必须正好是 7 个 ASCII 字符。"""
        assert ThinkingStripper.OPEN_TAG == OPEN
        assert len(ThinkingStripper.OPEN_TAG) == 7
        assert ThinkingStripper.OPEN_TAG.isascii(), "标签混入了非 ASCII 字符，剥离会静默失效"

    def test_close_tag_is_plain_ascii(self):
        """结束标签必须正好是 8 个 ASCII 字符。"""
        assert ThinkingStripper.CLOSE_TAG == CLOSE
        assert len(ThinkingStripper.CLOSE_TAG) == 8
        assert ThinkingStripper.CLOSE_TAG.isascii(), "标签混入了非 ASCII 字符，剥离会静默失效"

    def test_tags_are_actually_different(self):
        """两个标签不能相同，否则状态机会立刻错乱。"""
        assert ThinkingStripper.OPEN_TAG != ThinkingStripper.CLOSE_TAG

    def test_close_tag_starts_with_open_tag_body(self):
        """结束标签应包含相同的词根（仅多一个斜杠），这是模型的约定。"""
        assert "think" in ThinkingStripper.OPEN_TAG
        assert "think" in ThinkingStripper.CLOSE_TAG


class TestBasicStripping:
    """单块输入（不涉及跨 chunk 切分）。"""

    def test_strips_complete_thinking_block(self):
        text = f"{OPEN}让我想想…这里应该引用资料。{CLOSE}答案是混合检索。"

        result = strip_thinking(text)

        assert result == "答案是混合检索。"
        assert "让我想想" not in result

    def test_plain_text_passes_through_unchanged(self):
        """没有思考块时应原样返回，不丢字符。"""
        text = "这是一个普通回答，包含 BM25 与 RRF 等术语。"

        assert strip_thinking(text) == text

    def test_text_before_and_after_is_preserved(self):
        text = f"前缀{OPEN}推理过程{CLOSE}后缀"

        assert strip_thinking(text) == "前缀后缀"

    def test_multiple_thinking_blocks_are_all_removed(self):
        text = f"{OPEN}第一段推理{CLOSE}中间{OPEN}第二段推理{CLOSE}结尾"

        assert strip_thinking(text) == "中间结尾"

    def test_disabled_passes_everything_through(self):
        """关闭剥离时应原样返回，包括思考内容。"""
        text = f"{OPEN}推理{CLOSE}答案"

        assert strip_thinking(text, enabled=False) == text

    def test_only_thinking_yields_empty_string(self):
        """整段都是思考时应返回空串，而不是残留标签。"""
        assert strip_thinking(f"{OPEN}纯推理{CLOSE}") == ""


class TestStreamingAcrossChunks:
    """跨 chunk 切分 —— 这是必须用状态机而不是正则的原因。"""

    def _run(self, chunks: list[str], enabled: bool = True) -> str:
        """按给定切分喂入，返回拼接后的输出。"""
        stripper = ThinkingStripper(enabled=enabled)
        out = "".join(stripper.feed(c) for c in chunks)
        return out + stripper.flush()

    def test_open_tag_split_across_chunks(self):
        """开始标签被切成两半时，不能把标签碎片吐出去。"""
        chunks = ["答案", OPEN[:4], OPEN[4:], "隐藏", CLOSE, "可见"]

        result = self._run(chunks)

        assert result == "答案可见"
        assert "think" not in result, "标签碎片泄漏到了输出里"

    def test_close_tag_split_across_chunks(self):
        """结束标签被切成两半时同样不能泄漏。"""
        chunks = [OPEN + "推理", CLOSE[:5], CLOSE[5:], "正式答案"]

        result = self._run(chunks)

        assert result == "正式答案"
        assert "think" not in result

    def test_character_by_character_streaming(self):
        """极端情况：每次只来一个字符。

        这是最容易暴露状态机缺陷的场景 —— 逐字符喂入时，
        任何"提前输出"的假设都会导致标签碎片外泄。
        """
        full = f"开头{OPEN}这段推理不应出现{CLOSE}结尾"

        result = self._run(list(full))

        assert result == "开头结尾"
        assert "推理" not in result

    def test_thinking_split_across_three_chunks(self):
        chunks = [OPEN[:3], OPEN[3:] + "推理内容", CLOSE[:3], CLOSE[3:] + "答案"]

        assert self._run(chunks) == "答案"

    def test_tail_is_flushed(self):
        """流结束时缓冲区里的正文必须补发，否则末尾几个字符会丢。

        因为状态机会为"可能是半个标签"预留最多 7 个字符，
        短输出很容易全部卡在缓冲区里。
        """
        assert self._run(["短"]) == "短"
        assert self._run(["1234567"]) == "1234567"
        assert self._run(["12345678"]) == "12345678"

    def test_unterminated_thinking_is_discarded_on_flush(self):
        """模型被截断（有开始标签但没结束标签）时，剩余内容属推理，应丢弃。"""
        assert self._run([OPEN + "还没想完就被截断了"]) == ""

    def test_unterminated_partial_close_tag_in_body_is_kept(self):
        """正文里出现"半个结束标签"样子的文本，应作为正文保留。

        这是误判成本最低的选择：宁可多留几个字符，也不要吞掉正文。
        """
        result = self._run(["正文" + CLOSE[:3]])

        assert result == "正文" + CLOSE[:3]

    def test_multiple_blocks_across_chunks(self):
        chunks = [OPEN, "推理一", CLOSE, "甲", OPEN, "推理二", CLOSE, "乙"]

        assert self._run(chunks) == "甲乙"

    def test_disabled_mode_does_not_buffer(self):
        """关闭剥离时不应缓冲，避免无谓地推迟输出。"""
        stripper = ThinkingStripper(enabled=False)

        # 单字符应立即返回，而不是被缓冲
        assert stripper.feed("甲") == "甲"
        assert stripper.flush() == ""


class TestCitationSafety:
    """与项目引用机制的交互：思考里的 [1] 不能被当成引用漏出去。"""

    def test_citation_inside_thinking_is_dropped(self):
        """思考中的引用编号会误导前端渲染，必须一并丢弃。"""
        text = f"{OPEN}根据资料 [1] 与 [2] 可以推断…{CLOSE}结论是混合检索 [1]。"

        result = strip_thinking(text)

        assert result == "结论是混合检索 [1]。"
        # 思考块里的两个引用标记不应出现
        assert result.count("[1]") == 1
        assert "[2]" not in result


@pytest.mark.parametrize(
    "text",
    [
        "",
        "无标签",
        OPEN + CLOSE,
        OPEN + "只有开始" + CLOSE + "结束",
        "嵌套" + OPEN + OPEN + "双开始" + CLOSE + "尾",
    ],
)
def test_stripper_never_crashes(text: str):
    """任意输入都不应抛异常 —— 它处在流式主链路上，崩了会中断整个回答。"""
    assert isinstance(strip_thinking(text), str)
