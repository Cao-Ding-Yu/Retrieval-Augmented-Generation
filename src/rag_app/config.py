"""应用配置。

所有可调参数集中在此处，通过环境变量 / .env 文件注入，符合"十二要素应用"的做法：
代码不硬编码任何本机地址、模型名或阈值。

用法：
    from rag_app.config import get_settings
    settings = get_settings()          # 带缓存，全进程只解析一次
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    全局配置对象。

    字段名（小写）与 .env 中的变量名（大写）自动对应，例如
    字段 ``ollama_base_url`` 对应环境变量 ``OLLAMA_BASE_URL``。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # 忽略 .env 中多余的变量，避免因为环境里存在无关变量而启动失败
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Ollama：本地模型服务
    # ------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama 服务地址，默认本机 11434 端口",
    )
    chat_model: str = Field(
        default="deepseek-r1:14b",
        description="对话生成模型，需已通过 ollama pull 拉取",
    )
    embedding_model: str = Field(
        default="qwen3-embedding:8b",
        description="稠密向量嵌入模型",
    )
    embedding_dim: int = Field(
        default=4096,
        description="嵌入维度，必须与 embedding_model 的真实输出一致：bge-m3=1024，nomic-embed-text=768",
    )

    # ------------------------------------------------------------------
    # Milvus：向量数据库
    # ------------------------------------------------------------------
    milvus_uri: str = Field(
        default="http://localhost:19530",
        description="Milvus gRPC 端点",
    )
    milvus_token: str = Field(
        default="",
        description="Milvus 鉴权 token（user:password），无鉴权时留空",
    )
    milvus_collection: str = Field(
        default="rag_chunks_bge_m3_1024",
        description="集合名。建议把嵌入模型与维度编入名字，换模型时自动落到新集合，避免维度冲突",
    )

    # ------------------------------------------------------------------
    # 检索参数
    # ------------------------------------------------------------------
    dense_top_k: int = Field(default=20, ge=1, description="稠密分支召回条数")
    sparse_top_k: int = Field(default=20, ge=1, description="BM25 稀疏分支召回条数")
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF 融合常数 k。越小越强调头部排名，60 为业界常用值",
    )
    fusion_mode: str = Field(
        default="client",
        description=(
            "融合策略：client=本地 RRF（可拿到分支名次，便于解释排序，多一次往返）；"
            "server=交给 Milvus 内置 RRFRanker（一次往返，但无分支名次）"
        ),
    )

    # ------------------------------------------------------------------
    # 重排参数
    # ------------------------------------------------------------------
    rerank_enabled: bool = Field(
        default=True,
        description="是否启用交叉编码器重排。置 false 可用于对比'有无重排'的效果差异",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="交叉编码器模型名；内存紧张可改 BAAI/bge-reranker-base",
    )
    rerank_top_n: int = Field(
        default=30,
        ge=1,
        description="送去重排的候选数量（融合之后截断）。重排是逐对前向计算，必须限制规模",
    )
    final_top_k: int = Field(
        default=5,
        ge=1,
        description="重排后最终作为上下文交给 LLM 的条数",
    )
    bm25_analyzer_type: str = Field(
        default="",
        description=(
            "文本字段分析器类型，仅在建集合时生效。留空=Milvus 默认(standard)分析器；"
            "中文语料建议设为 'chinese'（jieba 分词），否则 BM25 词法召回基本失效。"
            "改动此项必须重建集合并重新入库"
        ),
    )

    # ------------------------------------------------------------------
    # HuggingFace 模型加载（主要影响交叉编码器重排模型）
    # ------------------------------------------------------------------
    hf_home: str = Field(
        default="",
        description=(
            "HuggingFace 缓存根目录。留空则用默认位置（Windows: "
            "C:/Users/<用户>/.cache/huggingface），模型实际放在其下的 hub/ 子目录。"
            "仅在需要把缓存挪到别处时才填"
        ),
    )
    reranker_offline_first: bool = Field(
        default=True,
        description=(
            "优先以离线方式加载重排模型（local_files_only=True）。"
            "开启后完全不访问网络，避免每次启动都去 HuggingFace 校验 etag "
            "——在国内网络下这个校验常常超时数秒甚至直接失败"
        ),
    )
    reranker_allow_download: bool = Field(
        default=True,
        description=(
            "离线加载失败（本地缓存缺失）时是否允许联网下载。"
            "配合 RERANKER_OFFLINE_FIRST 使用：缓存命中时零网络，"
            "缓存缺失时首次自动下载，之后又回到离线"
        ),
    )

    # ------------------------------------------------------------------
    # 文本切分
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=800, ge=100, description="分块目标字符数")
    chunk_overlap: int = Field(default=120,ge=0,description="相邻分块重叠字符数")

    # ------------------------------------------------------------------
    # 生成参数
    # ------------------------------------------------------------------
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="采样温度")
    strip_thinking: bool = Field(
        default=True,
        description=(
            "剥离推理模型（如 deepseek-r1）输出中的  thinking... 思考过程。"
            "这类模型会把推理链混在正文里，直接流式输出会让用户看到大段思考内容"
        ),
    )
    max_context_chars: int = Field(
        default=6000,
        ge=500,
        description="拼进提示词的单条上下文最大字符数，防止超长上下文拖慢推理",
    )

    # ------------------------------------------------------------------
    # 服务
    # ------------------------------------------------------------------
    host: str = Field(default="127.0.0.1", description="监听地址")
    port: int = Field(default=8000, ge=1, le=65535, description="监听端口")
    log_level: str = Field(default="info", description="uvicorn 日志级别")
    data_dir: str = Field(default="./data", description="语料目录")

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_be_smaller_than_size(cls, v: int, info: ValidationInfo) -> int:
        """重叠必须小于块大小，否则切分会陷入死循环或产出重复块。"""
        chunk_size = info.data.get("chunk_size", 800)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap({v}) 必须小于 chunk_size({chunk_size})，否则切分无法推进"
            )
        return v

    @field_validator("milvus_token", "milvus_uri", mode="before")
    @classmethod
    def _strip(cls, v):
        """去掉 .env 里手写配置常见的首尾空白。"""
        return v.strip() if isinstance(v, str) else v

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def milvus_token_or_none(self) -> str | None:
        """把空字符串归一化为 None —— pymilvus 在无鉴权时应传 None 而非 ""。"""
        return self.milvus_token or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的 Settings 实例。

    使用 lru_cache 保证：
      1. .env 只被解析一次，避免每个请求都读盘；
      2. 全进程拿到同一个对象，配置不会中途变化。
    测试中若需覆盖配置，先调用 get_settings.cache_clear()。
    """
    return Settings()
