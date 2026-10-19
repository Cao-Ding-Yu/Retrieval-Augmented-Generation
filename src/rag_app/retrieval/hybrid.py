"""混合检索编排：向量化 -> 双分支召回 -> RRF 融合 -> 交叉编码器重排

整个 RAG 的检索主干，把 store / fusion / rerank 三层串起来：

    查询文本
      │
      ├─► OllamaEmbeddings ─► 稠密向量 ─┐
      │                                 ├─► RRF 融合 ─► 截断 ─► CrossEncoder ─► top-k
      └─► 原始文本 ─────────► BM25 稀疏 ─┘

## 阻塞与异步的边界

Ollama 的嵌入接口、Milvus 的 SDK 都是同步阻塞的。若直接在 async 函数里调用，
会占住事件循环，使 SSE 流式输出期间无法处理其他请求 —— 这正是"看起来是异步、
实际串行"的典型性能陷阱。因此这里把所有阻塞调用统一包进
``asyncio.to_thread``，让它们在线程池里执行。
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any

from rag_app.config import Settings
from rag_app.models import RetrievedChunk
from rag_app.retrieval.milvus_store import MilvusStore, RawHit, fuse_two_branch
from rag_app.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)


class FusionMode(str, Enum):
    """融合策略（两种都支持，按需选择）。

    CLIENT: 两次独立检索 + 本地 RRF（``fuse_two_branch``）。
        能拿到各分支的原始名次，便于向用户解释排序依据。
        代价是多一次网络往返。

    SERVER: 一次 ``hybrid_search`` 交给 Milvus 用内置 ``RRFRanker`` 融合。
        往返更少（生产环境 QPS 压力大时更合适），但拿不到分支名次。

    两者算出的排序是一致的 —— Milvus 的 ``RRFRanker`` 用的就是同一个 RRF 公式。
    区别只在"谁来做融合"和"是否保留中间信息"。
    """

    CLIENT = "client"
    SERVER = "server"


class HybridRetriever:
    """两阶段混合检索器。

    Args:
        store: Milvus 存储封装。
        embedder: 提供 ``embed_query`` / ``embed_documents`` 的嵌入器。
        reranker: 重排器实现（可为 NoOpReranker）。
        settings: 全局配置。
        fusion_mode: 融合策略，由 ``FUSION_MODE`` 配置决定，两种都可用
            （见 :class:`FusionMode` 的说明）。
    """

    def __init__(
        self,
        store: MilvusStore,
        embedder: Any,
        reranker: Reranker,
        settings: Settings,
        fusion_mode: FusionMode = FusionMode.CLIENT,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings
        self._fusion_mode = fusion_mode

    @property
    def fusion_mode(self) -> FusionMode:
        """当前生效的融合策略。"""
        return self._fusion_mode

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """执行完整检索链路，返回最终送进提示词的上下文分块。

        Args:
            query: 用户查询文本。

        Returns:
            按相关性降序的 RetrievedChunk 列表，长度不超过 ``final_top_k``。
            查询为空或库为空时返回空列表（不抛错）。
        """
        query = (query or "").strip()
        if not query:
            logger.warning("收到空查询，跳过检索")
            return []

        # 整条链路都是阻塞的（嵌入 + Milvus 调用 + 重排），一并放进线程池
        return await asyncio.to_thread(self._retrieve_sync, query)

    # ------------------------------------------------------------------
    # 同步实现（在线程池中执行）
    # ------------------------------------------------------------------
    def _retrieve_sync(self, query: str) -> list[RetrievedChunk]:
        """检索链路的同步实现。"""
        # ---- 第 1 步：查询向量化 ----
        vector = self._embedder.embed_query(query)

        # ---- 第 2 步：召回 ----
        if self._fusion_mode is FusionMode.SERVER:
            candidates = self._recall_server_fusion(vector, query)
        else:
            candidates = self._recall_client_fusion(vector, query)

        if not candidates:
            logger.info("检索无命中：%s", query[:50])
            return []

        # ---- 第 3 步：截断到重排预算 ----
        # 重排是逐对前向，必须限制候选规模；不截断则延迟随召回量线性增长
        candidates = candidates[: self._settings.rerank_top_n]

        # ---- 第 4 步：交叉编码器精排 ----
        ranked = self._apply_rerank(query, candidates)

        logger.info(
            "检索完成：融合模式=%s，候选=%d，最终=%d",
            self._fusion_mode.value,
            len(candidates),
            len(ranked),
        )
        return ranked

    # ------------------------------------------------------------------
    # 召回：路径 A —— 客户端融合（含分支名次）
    # ------------------------------------------------------------------
    def _recall_client_fusion(self, vector: list[float], query: str) -> list[RetrievedChunk]:
        """两次检索 + 本地 RRF 融合。

        相比服务端融合，这条路径多一次网络往返，但换来每个候选在两条分支上的
        原始名次 —— 这是"可解释检索"的基础。
        """
        dense_hits, sparse_hits = self._store.search_and_fuse_branches(
            vector=vector,
            query_text=query,
            dense_limit=self._settings.dense_top_k,
            sparse_limit=self._settings.sparse_top_k,
        )

        # 建立 主键 -> RawHit 的索引。同一条分块可能在两个分支都出现，
        # 内容字段相同，因此后者（或前者）覆盖都无所谓，这里用 setdefault
        # 保留先出现的那个，避免无意义的重复赋值。
        hit_index: dict[str, RawHit] = {}
        for hit in dense_hits:
            hit_index.setdefault(hit.key, hit)
        for hit in sparse_hits:
            hit_index.setdefault(hit.key, hit)

        # 用纯函数做 RRF 融合，保证算法可单测
        fused = fuse_two_branch(
            dense_keys=[h.key for h in dense_hits],
            sparse_keys=[h.key for h in sparse_hits],
            k=self._settings.rrf_k,
        )

        chunks: list[RetrievedChunk] = []
        for item in fused:
            hit = hit_index.get(item.key)
            if hit is None:  # pragma: no cover - 理论上不会发生
                continue

            chunks.append(
                RetrievedChunk(
                    text=hit.text,
                    doc_id=hit.doc_id,
                    source=hit.source,
                    chunk_index=hit.chunk_index,
                    # 此刻的综合分数就是 RRF 分数；若有重排会在下一步被覆盖
                    score=item.score,
                    dense_rank=item.rank_in("dense"),
                    sparse_rank=item.rank_in("sparse"),
                    rrf_score=item.score,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # 召回：路径 B —— 服务端融合（一次往返）
    # ------------------------------------------------------------------
    def _recall_server_fusion(self, vector: list[float], query: str) -> list[RetrievedChunk]:
        """交由 Milvus 服务端完成 RRF 融合。

        注意：这条路径拿不到分支名次（Milvus 只返回融合后的结果），
        因此 dense_rank / sparse_rank 为 None，前端也就无法标出"双分支命中"。
        需要这层可解释性时应使用客户端融合。
        """
        # 服务端融合只有一次调用，召回量直接由重排预算决定：
        # 关了重排就只要最终条数，开着重排则多取一些供精排挑选。
        recall_limit = (
            self._settings.rerank_top_n
            if self._settings.rerank_enabled
            else self._settings.final_top_k
        )

        hits = self._store.hybrid_search(
            vector=vector,
            query_text=query,
            limit=recall_limit,
            rrf_k=self._settings.rrf_k,
        )

        return [
            RetrievedChunk(
                text=hit.text,
                doc_id=hit.doc_id,
                source=hit.source,
                chunk_index=hit.chunk_index,
                score=hit.score,
                # 服务端只回传融合分数，没有分支名次
                dense_rank=None,
                sparse_rank=None,
                rrf_score=hit.score,
            )
            for hit in hits
        ]

    # ------------------------------------------------------------------
    # 精排
    # ------------------------------------------------------------------
    def _apply_rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """对候选做交叉编码器重排，返回最终的 top-k。

        重排后会把 ``score`` 更新为重排分数，让"展示给用户的分数"与
        "实际决定排序的依据"保持一致；原始的 RRF 分数保留在 ``rrf_score`` 里，
        便于对比"融合排序"与"重排排序"的差异（这也是消融实验要看的数据）。
        """
        if not candidates:
            return []

        ordered = self._reranker.rerank(
            query=query,
            documents=[c.text for c in candidates],
            # 注意：这里传 None，先拿到完整排序，再自己截断。
            # 直接传 top_k 也可以，但那样拿不到被淘汰候选的分数，
            # 调试"为什么这条没进上下文"时会缺信息。
            top_k=None,
        )

        final: list[RetrievedChunk] = []
        for original_index, score in ordered[: self._settings.final_top_k]:
            chunk = candidates[original_index]
            chunk.score = score
            # NoOpReranker 返回的分数恒为 0，此时不应覆盖融合分数，
            # 否则上下文排序依据会变成"全是 0"的无意义值。
            if score != 0.0:
                chunk.rerank_score = score
            final.append(chunk)

        return final
