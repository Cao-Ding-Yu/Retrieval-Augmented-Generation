"""FastAPI 应用装配。

## 生命周期管理

所有重量级资源（Milvus 连接、2GB 的重排模型）都在 ``lifespan`` 中创建与释放，
而不是放在模块顶层或每个请求里：

  * **模块顶层创建**：import 阶段就产生副作用，导致测试无法导入模块、
    也让 `--reload` 变得更慢更脆；
  * **每请求创建**：重排模型每次加载要几秒，服务等于不可用；
  * **lifespan 创建**：进程内只创建一次，与进程同生命周期，是正确的位置。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rag_app import __version__
# 这里刻意用"完整子模块路径"导入路由，而不是 `from rag_app.api import routes_chat`：
# 后者会在 rag_app.api.__init__ 尚未执行完时去读包的属性，
# 与 __init__ 里对 create_app 的再导出形成循环依赖。
from rag_app.api.routes_chat import router as chat_router
from rag_app.api.routes_ingest import router as ingest_router
from rag_app.config import Settings, get_settings
from rag_app.deps import build_context, teardown_context

logger = logging.getLogger(__name__)

# 仓库根目录下的前端静态资源目录。
# 路径推导：本文件是 <root>/src/rag_app/api/app.py，
# parents[0]=api, [1]=rag_app, [2]=src, [3]=<root>
WEB_DIR = Path(__file__).resolve().parents[3] / "web"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期钩子：启动时组装依赖，关闭时释放资源。"""
    logger.info("应用启动中，正在初始化依赖…")

    # 挂到 app.state 而非全局变量，使多个应用实例（测试场景）互不干扰
    ctx = build_context()
    app.state.ctx = ctx

    if ctx.ready:
        logger.info("启动完成，服务已就绪")
    else:
        # 启动失败也照常提供服务，只是检索接口返回 503。
        # 这样运维能通过 /health 看到具体原因，而不是面对一个无法连接的端口。
        logger.error("启动完成，但检索功能不可用：%s", ctx.startup_error)

    try:
        yield
    finally:
        logger.info("应用关闭中，正在释放资源…")
        teardown_context(ctx)


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造 FastAPI 应用。

    Args:
        settings: 可选的配置覆盖，供测试注入。为 None 时使用全局单例。

    Returns:
        配置完成的 FastAPI 实例。
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="本地 RAG 知识库问答服务",
        description=(
            "基于 Milvus 混合检索（稠密语义 + BM25 词法）、RRF 倒数排名融合、"
            "交叉编码器重排，以及 Ollama 本地大模型的检索增强生成服务。"
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS：仅在使用独立前端开发服务器（如 Vite 的 5173 端口）时需要。
    # 生产环境应把 allow_origins 收紧到具体域名，而不是保留 *。
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------
    @app.get("/health", tags=["ops"], summary="健康检查")
    async def health(request: Request) -> JSONResponse:
        """返回服务与依赖的可用状态。

        始终返回 HTTP 200（除非进程本身挂了），把"是否就绪"放在响应体里。
        这样负载均衡/监控能区分"服务没起来"与"服务起来了但依赖不可用"，
        后者返回 503 会让一些编排系统直接重启容器，反而掩盖真实原因。
        """
        ctx = request.app.state.ctx
        body = {
            "status": "ok" if ctx.ready else "degraded",
            "version": __version__,
            "ready": ctx.ready,
            "error": ctx.startup_error,
            "warnings": ctx.warnings,
            "milvus": {
                "collection": ctx.settings.milvus_collection,
                "server_version": ctx.store.server_version if ctx.store else None,
                "entities": ctx.store.num_entities() if ctx.store else None,
            },
            "retrieval": {
                "fusion_mode": ctx.fusion_mode,
                "embedding_model": ctx.settings.embedding_model,
                "embedding_dim": ctx.settings.embedding_dim,
                "dense_top_k": ctx.settings.dense_top_k,
                "sparse_top_k": ctx.settings.sparse_top_k,
                "rrf_k": ctx.settings.rrf_k,
                "final_top_k": ctx.settings.final_top_k,
            },
            "reranker": {
                "enabled": type(ctx.reranker).__name__ != "NoOpReranker",
                "impl": type(ctx.reranker).__name__,
                "model": ctx.settings.reranker_model,
                "device": getattr(ctx.reranker, "device", None),
                # 加载是否走的纯离线路径，以及实际用的快照目录。
                # 这两个字段是排查"模型已在本地却仍要联网"的直接依据：
                # loaded_offline=true 表示零网络请求；resolved_path 能确认
                # 用的到底是哪份缓存。
                "loaded_offline": getattr(ctx.reranker, "loaded_offline", None),
                "resolved_path": getattr(ctx.reranker, "resolved_path", None),
            },
            "llm": {"chat_model": ctx.settings.chat_model},
        }
        # degraded 时给一个非 200 的语义提示，但状态码仍保持 200 以便监控区分
        return JSONResponse(content=body, status_code=200)

    # ------------------------------------------------------------------
    # 业务路由
    # 注意注册顺序：静态资源挂载在 "/" 会兜住所有未匹配路径，
    # 必须放在 API 路由之后注册。
    # ------------------------------------------------------------------
    app.include_router(chat_router)
    app.include_router(ingest_router)

    # ------------------------------------------------------------------
    # 静态前端
    # ------------------------------------------------------------------
    if WEB_DIR.is_dir():
        # html=True 让 GET / 直接返回 index.html
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
        logger.info("已挂载静态前端目录：%s", WEB_DIR)
    else:  # pragma: no cover - 仅在 web/ 被删掉时触发
        logger.warning("未找到前端目录 %s，界面将不可用（API 不受影响）", WEB_DIR)

    return app
