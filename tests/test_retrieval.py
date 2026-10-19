"""混合检索编排（HybridRetriever）的单元测试。

用假 store / 假嵌入器 / 假重排器把外部依赖全部隔离，专注验证**编排逻辑**：
  * 两条融合路径是否都被正确调用，参数是否传对；
  * 分支名次是否被正确回填到 RetrievedChunk；
  * 重排是否真的改变了顺序、分数是否被正确写回；
  * 各种截断阈值（rerank_top_n / final_top_k）是否生效；
  * 降级逻辑（服务端融合不可用时）是否符合预期。

这些断言覆盖的是"会静默出错"的地方 —— 参数传错不会报错，
只会让检索质量变差，所以必须有测试钉住。
"""

from __future__ import annotations

import pytest

from rag_app.retrieval.hybrid import FusionMode, HybridRetriever
from rag_app.retrieval.rerank import NoOpReranker
from tests.conftest import (
    FakeEmbedder,
    FakeReranker,
    FakeStore,
    make_raw_hit,
)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def build_retriever(
    settings,
    store: FakeStore,
    embedder: FakeEmbedder,
    reranker=None,
    fusion_mode: FusionMode = FusionMode.CLIENT,
) -> HybridRetriever:
    """构造被测检索器，默认使用客户端融合。"""
    return HybridRetriever(
        store=store,          # type: ignore[arg-type] - 结构化鸭子类型，测试替身
        embedder=embedder,
        reranker=reranker or NoOpReranker(),
        settings=settings,
        fusion_mode=fusion_mode,
    )


class TestClientFusionPath:
    """客户端融合：两次检索 + 本地 RRF，应带完整分支名次。"""

    async def test_branch_ranks_are_populated(self, settings, embedder):
        """RDF 融合后每条结果都应带上它在两个分支的原始名次。"""
        store = FakeStore(
            dense_hits=[make_raw_hit("a"), make_raw_hit("b")],
            sparse_hits=[make_raw_hit("b"), make_raw_hit("a")],
        )
        retriever = build_retriever(settings, store, embedder)

        chunks = await retriever.retrieve("测试查询")

        assert len(chunks) == 2

        for chunk in chunks:
            assert chunk.dense_rank is not None, "客户端融合必须回填稠密名次"
            assert chunk.sparse_rank is not None, "客户端融合必须回填稀疏名次"
            assert chunk.rrf_score is not None

    async def test_hybrid_hit_is_flagged(self, settings, embedder):
        """双分支同时命中的结果，is_hybrid_hit 应为 True。

        这个标记会一路传到前端，用来提示"该结果既语义相关又命中关键词"，
        是排序可信度的直观信号。
        """
        store = FakeStore(
            dense_hits=[make_raw_hit("only_dense"), make_raw_hit("both")],
            sparse_hits=[make_raw_hit("only_sparse"), make_raw_hit("both")],
        )
        retriever = build_retriever(settings, store, embedder)

        chunks = await retriever.retrieve("查询")

        hybrid_hits = [c for c in chunks if c.is_hybrid_hit]
        assert len(hybrid_hits) == 1, "应恰好有一条双分支命中的结果"
        assert hybrid_hits[0].dense_rank == 1
        assert hybrid_hits[0].sparse_rank == 1

    async def test_hybrid_hit_has_both_ranks_and_wins(self, settings, embedder):
        """精确断言：双分支命中（各排第 2）胜过单分支头名。"""
        store = FakeStore(
            dense_hits=[make_raw_hit("solo_dense"), make_raw_hit("both")],
            sparse_hits=[make_raw_hit("solo_sparse"), make_raw_hit("both")],
        )
        retriever = build_retriever(settings, store, embedder)

        chunks = await retriever.retrieve("查询")

        # final_top_k=2，取前两条
        assert len(chunks) == 2
        top = chunks[0]
        assert top.is_hybrid_hit, "首位应是双分支命中的结果"
        assert top.dense_rank == 1 and top.sparse_rank == 1
        assert chunks[1].is_hybrid_hit is False

    async def test_recall_limits_come_from_settings(self, settings, embedder):
        """两个分支的召回条数应严格取自配置，而不是硬编码。"""
        store = FakeStore(dense_hits=[], sparse_hits=[])
        retriever = build_retriever(settings, store, embedder)

        await retriever.retrieve("查询")

        assert store.last_dense_limit == settings.dense_top_k
        assert store.last_sparse_limit == settings.sparse_top_k

    async def test_query_is_embedded_once(self, settings, embedder):
        """查询只应被向量化一次，避免重复调用嵌入模型浪费算力。"""
        store = FakeStore(dense_hits=[make_raw_hit("a")], sparse_hits=[])
        retriever = build_retriever(settings, store, embedder)

        await retriever.retrieve("只嵌入一次")

        assert embedder.calls == ["只嵌入一次"]

    async def test_metadata_is_propagated(self, settings, embedder):
        """溯源元数据（来源、文档 ID、分块序号）必须完整传递到结果。"""
        store = FakeStore(
            dense_hits=[
                make_raw_hit(
                    "k1",
                    text="正文内容",
                    doc_id="doc-abc",
                    source="docs/manual.pdf",
                    chunk_index=7,
                )
            ],
            sparse_hits=[],
        )
        retriever = build_retriever(settings, store, embedder)

        chunks = await retriever.retrieve("查询")

        assert len(chunks) == 1
        assert chunks[0].text == "正文内容"
        assert chunks[0].doc_id == "doc-abc"
        assert chunks[0].source == "docs/manual.pdf"
        assert chunks[0].chunk_index == 7


class TestServerFusionPath:
    """服务端融合：一次 hybrid_search，无分支名次。"""

    async def test_uses_hybrid_search_with_configured_rrf_k(self, settings, embedder):
        """应调用服务端 hybrid_search 并透传 rrf_k。"""
        store = FakeStore(server_hits=[make_raw_hit("a"), make_raw_hit("b")])
        retriever = build_retriever(
            settings, store, embedder, fusion_mode=FusionMode.SERVER
        )

        chunks = await retriever.retrieve("查询")

        assert store.last_rrf_k == settings.rrf_k
        assert len(chunks) == 2

    async def test_server_path_has_no_branch_ranks(self, settings, embedder):
        """服务端只回传融合结果，因此分支名次应为 None。

        这是选择客户端融合的直接理由：需要解释排序依据时，
        必须有分支名次可用。
        """
        store = FakeStore(server_hits=[make_raw_hit("a")])
        retriever = build_retriever(
            settings, store, embedder, fusion_mode=FusionMode.SERVER
        )

        chunks = await retriever.retrieve("查询")

        assert chunks[0].dense_rank is None
        assert chunks[0].sparse_rank is None
        assert chunks[0].is_hybrid_hit is False

    async def test_server_recall_limit_equals_rerank_budget(self, settings, embedder):
        """开启重排时，服务端召回量应等于重排预算（多召回供精排挑选）。"""
        store = FakeStore(server_hits=[])
        retriever = build_retriever(
            settings, store, embedder, fusion_mode=FusionMode.SERVER
        )

        await retriever.retrieve("查询")

        assert store.last_server_limit == settings.rerank_top_n

    async def test_fusion_mode_follows_config(self, settings, embedder):
        """融合策略直接取自配置，两种模式都能用。

        早期版本里融合策略是被 BM25 能力"自动修正"的，因为 Milvus 2.4/2.5
        的稀疏向量产出方式不同、只有一种能被独立检索。现在项目只支持 2.5+
        的服务端 BM25，两条路径都成立，因此这个自动纠错逻辑已被移除 ——
        配置说什么就是什么，行为可预期。
        """
        for requested in (FusionMode.CLIENT, FusionMode.SERVER):
            store = FakeStore(dense_hits=[], sparse_hits=[], server_hits=[])
            retriever = build_retriever(
                settings, store, embedder, fusion_mode=requested
            )
            assert retriever.fusion_mode is requested


class TestRerankStage:
    """重排阶段：顺序改变、分数写回、截断生效。"""

    async def test_rerank_order_is_applied(self, settings, embedder):
        """重排返回的顺序必须成为最终顺序（这是精排的意义所在）。"""
        store = FakeStore(
            dense_hits=[make_raw_hit("a"), make_raw_hit("b"), make_raw_hit("c")],
            sparse_hits=[],
        )
        # 强制把最后一名提到首位
        reranker = FakeReranker(order=[2, 0, 1], score=0.77)
        retriever = build_retriever(settings, store, embedder, reranker=reranker)

        chunks = await retriever.retrieve("查询")

        assert len(chunks) == settings.final_top_k
        # 原本排在末位的 c 应被提到第一
        assert chunks[0].dense_rank == 2

    async def test_rerank_score_is_written_back(self, settings, embedder):
        """重排分数应写入 score 与 rerank_score 两个字段。

        score 用于展示"最终排序依据"，rerank_score 保留原始重排分，
        两者一致便于核对。
        """
        store = FakeStore(dense_hits=[make_raw_hit("a")], sparse_hits=[])
        reranker = FakeReranker(order=[0], score=0.42)
        retriever = build_retriever(settings, store, embedder, reranker=reranker)

        chunks = await retriever.retrieve("查询")

        assert chunks[0].score == pytest.approx(0.42)
        assert chunks[0].rerank_score == pytest.approx(0.42)

    async def test_rrf_score_is_preserved_after_rerank(self, settings, embedder):
        """重排不应抹掉 RRF 分数 —— 它是对比"融合排序 vs 精排排序"的依据。"""
        store = FakeStore(
            dense_hits=[make_raw_hit("a")],
            sparse_hits=[make_raw_hit("a")],
        )
        reranker = FakeReranker(order=[0], score=0.9)
        retriever = build_retriever(settings, store, embedder, reranker=reranker)

        chunks = await retriever.retrieve("查询")

        assert chunks[0].rrf_score is not None
        assert chunks[0].rrf_score != chunks[0].rerank_score

    async def test_reranker_receives_at_most_rerank_top_n_documents(
        self, settings, embedder
    ):
        """送去重排的候选数必须被 rerank_top_n 限制。

        重排是逐对前向计算，候选不设上限会让延迟随召回量线性增长 ——
        这是最容易在"召回调大一点"时踩到的性能坑。
        """
        # 造 10 条稠密结果，而 rerank_top_n=5
        store = FakeStore(
            dense_hits=[make_raw_hit(f"k{i}") for i in range(10)],
            sparse_hits=[],
        )
        reranker = FakeReranker()
        retriever = build_retriever(settings, store, embedder, reranker=reranker)

        await retriever.retrieve("查询")

        assert len(reranker.received_docs) == 1
        assert len(reranker.received_docs[0]) == settings.rerank_top_n

    async def test_reranker_receives_original_query(self, settings, embedder):
        """重排必须用**原始查询**，而不是改写后的文本或向量。"""
        store = FakeStore(dense_hits=[make_raw_hit("a")], sparse_hits=[])
        reranker = FakeReranker()
        retriever = build_retriever(settings, store, embedder, reranker=reranker)

        await retriever.retrieve("原始查询文本")

        assert reranker.received_queries == ["原始查询文本"]

    async def test_final_top_k_is_respected(self, settings, embedder):
        """最终上下文条数必须等于 final_top_k。"""
        store = FakeStore(
            dense_hits=[make_raw_hit(f"k{i}") for i in range(8)],
            sparse_hits=[],
        )
        retriever = build_retriever(settings, store, embedder, reranker=FakeReranker())

        chunks = await retriever.retrieve("查询")

        assert len(chunks) == settings.final_top_k

    async def test_noop_reranker_does_not_zero_out_scores(self, settings, embedder):
        """关闭重排时，score 必须保留 RRF 分数，不能被 NoOp 的 0 覆盖。

        这是一个实际发生过的 bug 类型：NoOpReranker 返回 0 分，
        若无条件写回 score，界面上的相关性分数会全变成 0，
        上下文排序依据也随之丢失（顺序虽在，但用户无法理解为什么）。
        """
        store = FakeStore(
            dense_hits=[make_raw_hit("a")],
            sparse_hits=[make_raw_hit("a")],
        )
        retriever = build_retriever(
            settings, store, embedder, reranker=NoOpReranker()
        )

        chunks = await retriever.retrieve("查询")

        assert chunks[0].score > 0, "NoOp 重排不应把分数清零"
        assert chunks[0].rerank_score is None, "未真正重排时不应记录重排分数"


class TestEdgeCases:
    """边界输入。"""

    async def test_empty_query_returns_empty_without_calling_store(
        self, settings, embedder
    ):
        """空查询应直接返回空列表，不浪费一次嵌入与检索调用。"""
        store = FakeStore(dense_hits=[make_raw_hit("a")])
        retriever = build_retriever(settings, store, embedder)

        assert await retriever.retrieve("") == []
        assert await retriever.retrieve("   ") == []
        assert embedder.calls == []

    async def test_no_hits_returns_empty(self, settings, embedder):
        """两个分支都无命中时返回空列表，而不是抛错。"""
        store = FakeStore(dense_hits=[], sparse_hits=[])
        retriever = build_retriever(settings, store, embedder)

        assert await retriever.retrieve("库里没有相关内容") == []

    async def test_results_shorter_than_final_top_k(self, settings, embedder):
        """命中数少于 final_top_k 时应返回实际数量，不补空、不报错。"""
        store = FakeStore(dense_hits=[make_raw_hit("only")], sparse_hits=[])
        retriever = build_retriever(settings, store, embedder)

        chunks = await retriever.retrieve("查询")

        assert len(chunks) == 1

    async def test_scores_are_descending_after_fusion(self, settings, embedder):
        """未重排时，输出顺序必须按 RRF 分数降序。"""
        store = FakeStore(
            dense_hits=[make_raw_hit("a"), make_raw_hit("b")],
            sparse_hits=[make_raw_hit("b")],
        )
        retriever = build_retriever(settings, store, embedder)

        chunks = await retriever.retrieve("查询")

        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)
