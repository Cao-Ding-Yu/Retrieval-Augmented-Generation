"""重排器的单元测试。

关键点：**不加载真实模型**。测试里把一个桩模型直接注入到重排器内部，
于是可以精确控制每个候选的分数，从而断言"排序逻辑"是否正确。

真实模型的加载与推理放在 ``TestRealModelIntegration`` 里，
打上 integration 标记，默认不跑（模型约 2GB，下载与推理都不适合放在单元测试）。
"""

from __future__ import annotations

import pytest

from rag_app.retrieval.rerank import (
    CrossEncoderReranker,
    NoOpReranker,
    Reranker,
    build_reranker,
)


class StubCrossEncoder:
    """桩模型：predict 返回预置分数，不进行任何推理。

    接口刻意与 sentence_transformers.CrossEncoder.predict 保持一致，
    这样注入后被测代码无需任何改动。
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.call_count = 0
        self.last_pairs: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        self.call_count += 1
        self.last_pairs = list(pairs)
        # 只返回与输入等长的部分，模拟真实 API 的行为
        return self._scores[: len(pairs)]


def make_reranker_with_stub(scores: list[float]) -> tuple[CrossEncoderReranker, StubCrossEncoder]:
    """构造一个已注入桩模型的重排器。

    通过直接给内部 ``_model`` 赋值来跳过 ``load()``：
    load() 在 ``_model`` 非 None 时会立即返回，因此真实模型永远不会被加载。
    这样测试既不需要网络，也不占用 2GB 内存。
    """
    reranker = CrossEncoderReranker(model_name="stub-model")
    stub = StubCrossEncoder(scores)
    reranker._model = stub  # noqa: SLF001 - 测试中刻意注入桩实现
    return reranker, stub


class TestCrossEncoderRerankerOrdering:
    """排序逻辑。"""

    def test_sorts_by_score_descending(self):
        """分数高的候选必须排在前面。"""
        reranker, _ = make_reranker_with_stub([0.1, 0.9, 0.5])

        result = reranker.rerank("查询", ["doc0", "doc1", "doc2"])

        # doc1(0.9) > doc2(0.5) > doc0(0.1)
        assert [idx for idx, _ in result] == [1, 2, 0]

    def test_returns_original_indices_not_sorted_positions(self):
        """返回的必须是**原始下标**。

        这是最容易出错的地方：若误返回排序后的位置（0,1,2…），
        调用方按此取候选就会取错文本，且不会报错 —— 属于静默错误。
        """
        reranker, _ = make_reranker_with_stub([0.1, 0.9, 0.5])

        result = reranker.rerank("查询", ["a", "b", "c"])

        # 首位应是原下标 1（对应文本 "b"），而不是 0
        assert result[0][0] == 1
        assert result[0][1] == pytest.approx(0.9)

    def test_scores_are_paired_with_correct_indices(self):
        """分数与下标必须严格配对，不能错位。"""
        reranker, _ = make_reranker_with_stub([3.0, 1.0, 2.0])

        result = reranker.rerank("查询", ["a", "b", "c"])

        assert result == [(0, 3.0), (2, 2.0), (1, 1.0)]

    def test_top_k_truncates_after_sorting(self):
        """top_k 应在排序之后截断，取的是最相关的 k 条。"""
        reranker, _ = make_reranker_with_stub([0.1, 0.9, 0.5])

        result = reranker.rerank("查询", ["a", "b", "c"], top_k=2)

        assert [idx for idx, _ in result] == [1, 2]

    def test_top_k_none_returns_all(self):
        """top_k 为 None 时返回全部候选，供调用方自行截断。"""
        reranker, _ = make_reranker_with_stub([0.1, 0.9, 0.5])

        result = reranker.rerank("查询", ["a", "b", "c"], top_k=None)

        assert len(result) == 3

    def test_ties_keep_original_order(self):
        """分数相同时保持原始相对顺序（稳定排序）。

        目的是让结果可复现：若同分时顺序随机，
        相同输入会产出不同的上下文，进而产出不同的答案，
        使问题排查与效果评估失去可比性。
        """
        reranker, _ = make_reranker_with_stub([0.5, 0.5, 0.5])

        result = reranker.rerank("查询", ["a", "b", "c"])

        assert [idx for idx, _ in result] == [0, 1, 2]


class TestCrossEncoderRerankerBehavior:
    """输入输出契约。"""

    def test_empty_documents_returns_empty(self):
        """空输入返回空列表，不抛错、不调用模型。

        让调用方少写一层判空，同时避免无谓的模型前向。
        """
        reranker, stub = make_reranker_with_stub([])

        result = reranker.rerank("查询", [])

        assert result == []
        assert stub.call_count == 0, "空输入不应触发模型调用"

    def test_query_and_document_are_paired(self):
        """交叉编码器的输入必须是 (查询, 文档) 成对文本。

        这是 cross-encoder 与 bi-encoder 的本质区别：查询与文档要一起
        送进模型才能产生交互，拼接错误会得到无意义的相关性分数。
        """
        reranker, stub = make_reranker_with_stub([0.5, 0.6])

        reranker.rerank("我的问题", ["文档甲", "文档乙"])

        assert stub.last_pairs == [("我的问题", "文档甲"), ("我的问题", "文档乙")]

    def test_single_document(self):
        reranker, _ = make_reranker_with_stub([0.7])

        result = reranker.rerank("查询", ["唯一文档"])

        assert result == [(0, 0.7)]


class TestNoOpReranker:
    """关闭重排时的空实现。"""

    def test_preserves_original_order(self):
        """不重排时应保持融合后的原始顺序。"""
        reranker = NoOpReranker()

        result = reranker.rerank("查询", ["a", "b", "c"])

        assert [idx for idx, _ in result] == [0, 1, 2]

    def test_returns_zero_scores(self):
        """分数恒为 0，作为"未真正重排"的标记。

        上层据此判断不应把该分数写入 score 字段，
        否则最终相关性分数会全部变成 0。
        """
        reranker = NoOpReranker()

        result = reranker.rerank("查询", ["a", "b"])

        assert all(score == 0.0 for _, score in result)

    def test_top_k_still_applies(self):
        """即使不重排，top_k 截断语义仍应生效。"""
        reranker = NoOpReranker()

        result = reranker.rerank("查询", ["a", "b", "c"], top_k=2)

        assert len(result) == 2

    def test_empty_input(self):
        assert NoOpReranker().rerank("查询", []) == []


class TestBuildRerankerFactory:
    """工厂函数。"""

    def test_disabled_returns_noop(self):
        """配置关闭时必须返回 NoOpReranker，避免白白加载 2GB 模型。"""
        reranker = build_reranker(enabled=False, model_name="whatever")

        assert isinstance(reranker, NoOpReranker)

    def test_enabled_returns_cross_encoder(self):
        """开启时返回真实的交叉编码器重排器（此时不加载模型）。"""
        reranker = build_reranker(enabled=True, model_name="BAAI/bge-reranker-v2-m3")

        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3"

    def test_construction_does_not_load_model(self):
        """构造重排器不应立即加载模型。

        模型加载是秒级的重操作，必须推迟到应用 lifespan 里显式调用 warmup()，
        否则每次 import 或构造都会产生难以察觉的启动延迟。
        """
        reranker = build_reranker(enabled=True, model_name="BAAI/bge-reranker-v2-m3")

        # 内部模型仍为 None，说明尚未加载
        assert reranker._model is None  # noqa: SLF001 - 断言延迟加载行为

    def test_implements_reranker_protocol(self):
        """两种实现都应满足 Reranker 协议，保证可互换。"""
        assert isinstance(NoOpReranker(), Reranker)
        assert isinstance(CrossEncoderReranker(model_name="x"), Reranker)


@pytest.mark.integration
class TestRealModelIntegration:
    """真实模型集成测试（默认跳过）。

    运行方式::

        pytest -m integration tests/test_rerank.py

    首次运行会下载模型（bge-reranker-base 约 1GB），因此不纳入默认测试集。
    这个测试的价值在于验证"模型确实能区分相关与不相关文本"，
    属于对整条重排链路的端到端确认。
    """

    def test_reranker_prefers_relevant_document(self):
        """明显相关的文档应获得更高分数。"""
        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
        reranker.load()

        docs = [
            "今天天气不错，适合出门散步。",          # 无关
            "BM25 是一种基于词频与逆文档频率的检索算法。",  # 相关
        ]
        result = reranker.rerank("BM25 检索算法是什么", docs)

        assert result[0][0] == 1, "与查询相关的文档应排在首位"
        assert result[0][1] > result[1][1]
