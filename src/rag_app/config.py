"""配置与日志。

整个应用只有这一个配置来源：环境变量 / .env 文件。
代码里不硬编码任何地址、模型名或阈值。

用法::

    from rag_app.config import get_settings
    settings = get_settings()          # 带缓存，全进程只解析一次
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。字段名（小写）自动对应环境变量（大写）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "deepseek-r1:14b"
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dim: int = Field(default=4096, ge=1)

    # ---- Milvus ----
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "rag_chunks_qwen3_4096"

    # ---- 检索 ----
    dense_top_k: int = Field(default=20, ge=1)
    sparse_top_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    bm25_analyzer_type: str = ""

    # ---- 重排 ----
    rerank_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = Field(default=30, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    hf_home: str = ""
    reranker_offline_first: bool = True
    reranker_allow_download: bool = True

    # ---- 入库 ----
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    data_dir: str = "./data"

    # ---- 生成 ----
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_context_chars: int = Field(default=6000, ge=500)
    strip_thinking: bool = True

    # ---- 服务 ----
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info: ValidationInfo) -> int:
        """重叠必须小于块大小，否则切分无法推进。"""
        chunk_size = info.data.get("chunk_size", 800)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap({v}) 必须小于 chunk_size({chunk_size})")
        return v

    @property
    def milvus_token_or_none(self) -> str | None:
        """pymilvus 无鉴权时应传 None 而非空串。"""
        return self.milvus_token or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全局唯一配置实例（带缓存，.env 只解析一次）。"""
    return Settings()


def configure_logging(level: str = "info", *, verbose: bool = False) -> None:
    """统一的日志格式与降噪。Web 服务与 CLI 共用，避免两处漂移。"""
    logging.basicConfig(
        level="DEBUG" if verbose else level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # 这些库在 INFO 级别会刷出大量连接/心跳信息，盖住业务日志
    for name in (
        "pymilvus",
        "milvus_lite",
        "urllib3",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "transformers",
        "filelock",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
