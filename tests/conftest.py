"""pytest 公共夹具（fixtures）。

设计原则：**单元测试不依赖外部服务**。
Milvus、Ollama、重排模型都不在单元测试里真正启动 —— 它们被替换成
可控的假实现。这样测试能在几百毫秒内跑完，也不需要开发机装好整套环境。

需要真实依赖的测试统一打 ``@pytest.mark.integration`` 标记，
并由 pyproject.toml 中的 ``addopts = "-m 'not integration'"`` 默认排除。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_app.config import Settings
from rag_app.deps import AppContext
from rag_app.models import RetrievedChunk
from rag_app.retrieval.milvus_store import RawHit


@pytest.fixture
def settings() -> Settings:
    """测试用配置。

    ``_env_file=None`` 很关键：它让 Settings 不去读工作目录下的 .env。
    否则开发机上的个人配置（换过的模型、调过的 top_k）会渗进测试，
    造成"我这里能过、CI 上不过"的随机失败。
    """
    return Settings(
        _env_file=None,
        # 固定与测试断言相关的参数，避免默认值变化导致测试失效
        embedding_dim=1024,
        dense_top_k=3,
        sparse_top_k=3,
        rrf_k=60,
        rerank_top_n=5,
        final_top_k=2,
    )


# ---------------------------------------------------------------------------
# 假实现（test doubles）
# ---------------------------------------------------------------------------
@dataclass
class FakeEmbedder:
    """假嵌入器：返回固定向量，不访问 Ollama。"""

    dim: int = 1024
    calls: list[str] = field(default_factory=list)

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [[0.1] * self.dim for _ in texts]


@dataclass
class FakeReranker:
    """假重排器：按预设顺序返回下标，用于验证"重排确实改变了顺序"。

    Attributes:
        order: 期望返回的下标顺序。None 时保持原顺序。
        score: 为每个下标分配的分数（用于断言分数被正确写回）。
    """

    order: list[int] | None = None
    score: float = 0.9
    received_queries: list[str] = field(default_factory=list)
    received_docs: list[list[str]] = field(default_factory=list)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        self.received_queries.append(query)
        self.received_docs.append(list(documents))

        indices = self.order if self.order is not None else list(range(len(documents)))
        result = [(i, self.score) for i in indices]
        return result[:top_k] if top_k is not None else result


@dataclass
class FakeStore:
    """假 Milvus 存储：返回预置的召回结果，不连接数据库。

    同时提供两条融合路径，用于分别测试服务端融合与客户端融合。
    """

    dense_hits: list[RawHit] = field(default_factory=list)
    sparse_hits: list[RawHit] = field(default_factory=list)
    server_hits: list[RawHit] = field(default_factory=list)

    # 记录调用参数，供断言使用
    last_dense_limit: int = 0
    last_sparse_limit: int = 0
    last_server_limit: int = 0
    last_rrf_k: int = 0

    def search_and_fuse_branches(
        self,
        vector: list[float],
        query_text: str,
        dense_limit: int,
        sparse_limit: int,
    ) -> tuple[list[RawHit], list[RawHit]]:
        self.last_dense_limit = dense_limit
        self.last_sparse_limit = sparse_limit
        return self.dense_hits, self.sparse_hits

    def hybrid_search(
        self,
        vector: list[float],
        query_text: str,
        limit: int,
        rrf_k: int,
    ) -> list[RawHit]:
        self.last_server_limit = limit
        self.last_rrf_k = rrf_k
        return self.server_hits


@dataclass
class FakeChat:
    """假对话模型：按需产出预设的文本增量。"""

    deltas: list[str] = field(default_factory=lambda: ["你好", "，", "世界"])

    async def astream_answer(self, question: str, chunks: list[RetrievedChunk]):
        for delta in self.deltas:
            yield delta

    async def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return "".join(self.deltas)


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------
def make_raw_hit(
    key: str,
    text: str = "示例文本",
    doc_id: str = "doc1",
    source: str = "sample.txt",
    chunk_index: int = 0,
    score: float = 0.5,
) -> RawHit:
    """快捷构造 RawHit，减少测试里的样板代码。"""
    return RawHit(
        key=key,
        text=text,
        doc_id=doc_id,
        source=source,
        chunk_index=chunk_index,
        score=score,
    )


def make_chunk(
    key: str,
    text: str = "示例文本",
    doc_id: str = "doc1",
    source: str = "sample.txt",
    chunk_index: int = 0,
    score: float = 0.5,
) -> RetrievedChunk:
    """快捷构造 RetrievedChunk。"""
    return RetrievedChunk(
        text=text,
        doc_id=doc_id,
        source=source,
        chunk_index=chunk_index,
        score=score,
    )


@pytest.fixture
def make_context():
    """返回一个工厂函数，用于构造可控的 AppContext。

    用法::

        def test_x(make_context):
            ctx = make_context(ready=True, retriever=FakeStore())
    """

    def _factory(
        settings: Settings,
        *,
        ready: bool = True,
        retriever: Any = None,
        chat: Any = None,
        store: Any = None,
        reranker: Any = None,
        startup_error: str | None = None,
    ) -> AppContext:
        ctx = AppContext(settings=settings)
        ctx.ready = ready
        ctx.startup_error = startup_error
        ctx.retriever = retriever
        ctx.chat = chat
        ctx.store = store
        if reranker is not None:
            ctx.reranker = reranker
        return ctx

    return _factory
