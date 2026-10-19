"""依赖组装（composition root）。

本模块是唯一"知道所有部件如何拼装"的地方，其余模块只依赖构造时注入的对象。
这样做的收益：
  * 各层之间没有全局变量与隐式依赖，单测可以直接构造局部对象；
  * 资源（Milvus 连接、2GB 的重排模型）在**应用启动时**创建一次，
    而不是每个请求创建一次。

## 关于启动失败的策略

``build_context`` **刻意不抛异常**。原因是：若 Milvus 没起来就让整个服务崩掉，
运维只能看到一句连接错误、无法通过 HTTP 接口判断"到底哪里没就绪"。
这里改为记录 ``ready=False`` 与具体原因，服务照常启动，
由 ``/health`` 明确报告状态，业务接口返回 503 并附原因。
这是更利于排障的行为。

同理，重排模型加载失败时会降级为 NoOpReranker 并把原因写入 ``warnings``，
服务仍可用（只是检索质量下降），且降级事实会出现在健康检查里，不会被静默忽略。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Request

from rag_app.config import Settings, get_settings
from rag_app.ingest.service import IngestService
from rag_app.llm.chat import RAGChat
from rag_app.retrieval.hybrid import FusionMode, HybridRetriever
from rag_app.retrieval.milvus_store import MilvusStore
from rag_app.retrieval.rerank import (
    NoOpReranker,
    Reranker,
    build_reranker,
    hf_cache_root,
)

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """应用级依赖容器，随应用生命周期存在。

    可选的部件（``store`` / ``retriever`` / ``chat`` / ``ingest``）在启动失败时为 None，
    因此调用方必须先检查 :attr:`ready`。
    """

    settings: Settings

    # ---- 可选部件：启动失败时为 None ----
    store: MilvusStore | None = None
    retriever: HybridRetriever | None = None
    chat: RAGChat | None = None
    ingest: IngestService | None = None
    embedder: Any = None

    # ---- 状态 ----
    reranker: Reranker = field(default_factory=NoOpReranker)
    """重排器。加载失败时会保持在 NoOpReranker（降级但可用）。"""

    ready: bool = False
    """检索链路是否可用。为 False 时业务接口应返回 503。"""

    startup_error: str | None = None
    """启动失败的原因，用于 /health 与 503 响应体。"""

    warnings: list[str] = field(default_factory=list)
    """非致命问题记录（例如重排降级），会在 /health 中暴露。"""

    @property
    def fusion_mode(self) -> str:
        """当前生效的融合策略。"""
        if self.retriever is None:
            return "unknown"
        return self.retriever.fusion_mode.value


def _parse_fusion_mode(raw: str) -> FusionMode:
    """把配置字符串解析成 FusionMode，非法值回退到 client 并记录。"""
    try:
        return FusionMode(raw.strip().lower())
    except ValueError:
        logger.warning("未知的 FUSION_MODE=%r，回退到 client", raw)
        return FusionMode.CLIENT


def _apply_hf_env(settings: Settings) -> None:
    """在导入任何 HuggingFace 相关库之前，把缓存位置写进环境变量。

    ## 为什么必须"尽早"

    ``huggingface_hub`` 在 **import 时** 就会读取 ``HF_HOME`` 并据此计算出
    ``HF_HUB_CACHE`` 等常量。一旦该模块被导入过，之后再改环境变量就不生效了。
    而 ``sentence_transformers`` / ``transformers`` 都会在导入期连锁导入它。

    因此这里刻意放在 ``build_context`` 的最前面、且在
    ``from langchain_ollama import ...`` 之前执行 —— 那些库可能间接拉起 HF 依赖。

    未配置 ``HF_HOME`` 时不做任何事，沿用 HuggingFace 的默认目录
    （Windows 通常是 ``C:/Users/<用户>/.cache/huggingface``）。

    Args:
        settings: 全局配置。
    """
    if not settings.hf_home:
        return

    target = Path(settings.hf_home).expanduser()
    # 必须用 setdefault 的语义"强制覆盖"：用户显式配置优先于外部环境变量，
    # 否则 .env 里的设置会被系统环境变量悄悄盖掉，表现为"配了没用"。
    os.environ["HF_HOME"] = str(target)
    logger.info("已将 HF_HOME 指向 %s", target)


def build_context(settings: Settings | None = None) -> AppContext:
    """构造应用上下文。此函数不会抛异常，失败信息记录在返回值中。

    Args:
        settings: 配置对象；None 时使用全局单例。

    Returns:
        已尽力初始化的 :class:`AppContext`。
    """
    settings = settings or get_settings()
    ctx = AppContext(settings=settings)

    # 必须先于任何 HF 相关 import 执行，详见函数文档
    _apply_hf_env(settings)

    # ------------------------------------------------------------------
    # 1) 对话与嵌入客户端（不依赖 Milvus，先建好）
    # ------------------------------------------------------------------
    # 延迟导入：langchain_ollama 会连带导入不少依赖，放在模块顶层会拖慢
    # 那些只需要用到纯函数（如 RRF）的测试与脚本。
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    ctx.embedder = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )

    chat_llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.chat_model,
        temperature=settings.temperature,
    )
    ctx.chat = RAGChat(llm=chat_llm, settings=settings)

    # ------------------------------------------------------------------
    # 2) 重排器（可选降级）
    # ------------------------------------------------------------------
    try:
        ctx.reranker = build_reranker(
            enabled=settings.rerank_enabled,
            model_name=settings.reranker_model,
            # 离线优先：命中本地缓存时完全不访问网络。
            # 这正是为了解决"模型已在本地缓存，每次启动却仍去 HuggingFace
            # 校验 etag"的问题 —— 该请求在国内网络下常超时数秒甚至失败。
            offline_first=settings.reranker_offline_first,
            # 缓存缺失时才允许联网，保证换机器后仍能一键跑通
            allow_download=settings.reranker_allow_download,
            cache_folder=settings.hf_home or None,
        )
        # 显式加载并预热：把这部分耗时从"第一个用户请求"提前到"服务启动"
        if hasattr(ctx.reranker, "warmup"):
            ctx.reranker.warmup()  # type: ignore[attr-defined]
    except Exception as exc:
        # 重排失败不阻断服务：降级为不重排，但把事实记录到 warnings。
        # 这条消息会显示在 Web 界面顶部的告警条里，所以要写得能直接指导排查，
        # 而不是只丢一句 "加载失败"。
        reason = (
            f"重排模型加载失败，已降级为不重排（检索仍可用，但精排失效）："
            f"{type(exc).__name__}: {exc} "
            f"| HF 缓存根目录：{hf_cache_root()} "
            f"| 完整堆栈见启动日志；修好后重启服务即可恢复"
        )
        logger.exception("重排模型加载失败：%s", exc)
        ctx.warnings.append(reason)
        ctx.reranker = NoOpReranker()

    # ------------------------------------------------------------------
    # 3) Milvus（检索链路的硬依赖，失败即整体不可用）
    # ------------------------------------------------------------------
    try:
        store = MilvusStore(settings=settings)
        # connect() 会同时校验服务端版本 >= 2.5（服务端 BM25 要求）
        store.connect()
        store.ensure_collection()
        ctx.store = store

        # ------------------------------------------------------------------
        # 4) 检索器与入库服务
        # ------------------------------------------------------------------
        ctx.retriever = HybridRetriever(
            store=store,
            embedder=ctx.embedder,
            reranker=ctx.reranker,
            settings=settings,
            fusion_mode=_parse_fusion_mode(settings.fusion_mode),
        )
        ctx.ingest = IngestService(
            store=store,
            embedder=ctx.embedder,
            settings=settings,
        )
        ctx.ready = True

        logger.info(
            "上下文就绪：Milvus=%s，融合=%s，重排=%s",
            store.server_version,
            ctx.fusion_mode,
            type(ctx.reranker).__name__,
        )

    except Exception as exc:
        # 记录失败原因，服务继续启动，由 /health 暴露状态
        ctx.startup_error = f"Milvus 初始化失败：{exc}"
        ctx.ready = False
        logger.exception("Milvus 初始化失败，检索功能不可用")

    return ctx


def teardown_context(ctx: AppContext) -> None:
    """释放上下文中持有的资源。应用关闭时调用。

    主要释放 Milvus 连接；重排模型由 Python 垃圾回收处理，
    进程退出时不需要也不应该做特殊处理。
    """
    if ctx.store is not None:
        try:
            ctx.store.close()
        except Exception:  # pragma: no cover - 关闭期异常不应影响退出
            logger.warning("关闭 Milvus 连接失败", exc_info=True)


def get_context(request: Request) -> AppContext:
    """FastAPI 依赖：从应用状态中取出上下文。

    挂在 ``app.state`` 上而不是模块级全局，好处是测试里可以用不同的
    Settings 构造多个应用实例而互不干扰。

    Args:
        request: FastAPI 注入的请求对象。

    Returns:
        当前应用的 AppContext。
    """
    return request.app.state.ctx  # type: ignore[no-any-return]
