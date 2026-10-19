"""测试公用夹具。

原则：**单元测试不依赖外部服务**。
Milvus、Ollama、重排模型全部用测试替身，因此测试能在几百毫秒内跑完，
也不需要开发机装好整套环境。
"""

from __future__ import annotations

import pytest

from rag_app.config import Settings
from rag_app.store import Hit


@pytest.fixture
def settings() -> Settings:
    """测试配置。

    ``_env_file=None`` 很关键：不让开发机上的 .env 渗进测试，
    否则会出现"我这里能过、别人那里不过"的随机失败。
    """
    return Settings(
        _env_file=None,
        embedding_dim=1024,
        dense_top_k=3,
        sparse_top_k=3,
        rrf_k=60,
        rerank_top_n=5,
        final_top_k=2,
    )


def make_hit(
    doc_id: str,
    text: str = "示例文本",
    source: str = "sample.txt",
    chunk_index: int = 0,
    score: float = 0.5,
) -> Hit:
    """快捷构造一条命中。"""
    return Hit(
        id=doc_id,
        text=text,
        doc_id=doc_id,
        source=source,
        chunk_index=chunk_index,
        score=score,
    )


class FakeRAG:
    """假的 RAG 实例，供 API 测试注入。

    只实现 API 层真正用到的属性与方法，不需要 Milvus / Ollama。
    """

    class _Store:
        server_version = "v2.5.4"

        def count(self) -> int:
            return 42

        def close(self) -> None:
            pass

    def __init__(
        self,
        settings: Settings,
        hits=None,
        deltas=None,
        rerank_enabled: bool = True,
        warning: str | None = None,
    ) -> None:
        self.settings = settings
        self.store = self._Store()
        self.rerank_enabled = rerank_enabled
        self.rerank_warning = warning
        self._hits = hits if hits is not None else []
        self._deltas = deltas if deltas is not None else ["你好", "，", "世界"]

    async def aretrieve(self, question: str):
        return list(self._hits)

    async def astream(self, question: str, hits):
        for delta in self._deltas:
            yield delta

    async def answer(self, question: str, hits) -> str:
        return "".join(self._deltas)
