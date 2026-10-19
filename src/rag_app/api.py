"""HTTP 接口层：应用装配 + 所有路由。

整个 HTTP 面只有一个文件，从上到下依次是：应用创建 → 依赖检查 → 各接口。
路由内部不再经过额外的依赖容器，直接调 ``app.state.rag``（一个 RAG 实例）。

## 关于 SSE

``/chat`` 用 Server-Sent Events 逐 token 推流。事件序列::

    event: sources   -> 检索到的引用列表（先发，前端立刻能渲染引用）
    event: token     -> 文本增量，可能多次
    event: done      -> 正常结束

出错时发 ``event: error`` —— 此时响应头早已发出，改不了 HTTP 状态码，
只能靠事件类型告知前端。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_app import __version__
from rag_app.config import Settings, get_settings
from rag_app.ingest import ingest
from rag_app.rag import NO_CONTEXT_REPLY, RAG

logger = logging.getLogger(__name__)

# 仓库根目录下的前端静态资源
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

# SSE 响应头：必须禁用代理缓冲，否则 token 会被攒起来一次性下发，
# 流式效果完全消失（这是"线上流式不生效"最常见的原因）
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
SSE_MEDIA = "text/event-stream; charset=utf-8"


# ===========================================================================
# 请求 / 响应模型
# ===========================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    include_sources: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class IngestPathRequest(BaseModel):
    path: str = Field(..., min_length=1)
    recursive: bool = True


# ===========================================================================
# 应用
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时创建 RAG 并连接依赖，关闭时释放。

    所有重量级资源（Milvus 连接、2GB 的重排模型）都在这里创建一次 ——
    不是模块导入时（会污染 import），也不是每个请求（重排模型每次加载要几秒）。
    """
    settings = app.state.settings
    rag = RAG(settings)
    app.state.rag = rag
    app.state.startup_error = None

    try:
        rag.start()
    except Exception as exc:
        # 启动失败也照常提供服务：让 /health 能报告原因，
        # 而不是让用户面对一个连不上的端口。
        app.state.startup_error = f"初始化失败：{exc}"
        logger.exception("初始化失败，检索功能不可用")

    yield
    rag.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造 FastAPI 应用。"""
    settings = settings or get_settings()

    app = FastAPI(
        title="本地 RAG 知识库问答",
        description="Milvus 混合检索（稠密 + BM25）+ RRF 融合 + 交叉编码器重排 + Ollama 生成",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    _register_routes(app)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:  # pragma: no cover
        logger.warning("未找到前端目录 %s", WEB_DIR)

    return app


# ===========================================================================
# 路由
# ===========================================================================
def _register_routes(app: FastAPI) -> None:
    """把接口挂到 app 上。全部路由集中在这里，便于一眼看全 HTTP 面。"""

    def get_rag(request: Request) -> RAG:
        """取出 RAG 实例；未就绪时返回 503 并带上初始化失败的原因。"""
        rag: RAG | None = getattr(request.app.state, "rag", None)
        error = getattr(request.app.state, "startup_error", None)
        if rag is None or error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error or "服务尚未就绪",
            )
        return rag

    # ---------------------------------------------------------------- 健康
    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> JSONResponse:
        """返回服务状态、降级原因与当前生效的检索配置。"""
        rag: RAG | None = getattr(request.app.state, "rag", None)
        error = getattr(request.app.state, "startup_error", None)
        cfg = request.app.state.settings

        # warnings 会显示在前端顶部的告警条里，所以必须是可读的中文原因
        warnings = [rag.rerank_warning] if (rag and rag.rerank_warning) else []

        return JSONResponse({
            "status": "ok" if (rag and not error) else "degraded",
            "version": __version__,
            "ready": bool(rag and not error),
            "error": error,
            "warnings": warnings,
            "milvus": {
                "collection": cfg.milvus_collection,
                "server_version": rag.store.server_version if rag else None,
                "chunks": rag.store.count() if (rag and not error) else None,
            },
            "retrieval": {
                "embedding_model": cfg.embedding_model,
                "dense_top_k": cfg.dense_top_k,
                "sparse_top_k": cfg.sparse_top_k,
                "rrf_k": cfg.rrf_k,
                "final_top_k": cfg.final_top_k,
            },
            "reranker": {
                "enabled": bool(rag and rag.rerank_enabled),
                "model": cfg.reranker_model,
            },
            "llm": {"chat_model": cfg.chat_model},
        })

    # ---------------------------------------------------------------- 对话（SSE）
    @app.post("/chat", tags=["chat"])
    async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        """流式对话：先推引用，再逐 token 推答案。"""
        rag = get_rag(request)

        async def stream() -> AsyncIterator[str]:
            try:
                # 1) 检索
                hits = await rag.aretrieve(payload.message)

                if payload.include_sources:
                    yield _sse("sources", {
                        "count": len(hits),
                        "sources": [h.to_dict() for h in hits],
                    })

                # 2) 检索为空：直接给兜底文案，不调用模型
                if not hits:
                    yield _sse("token", {"text": NO_CONTEXT_REPLY})
                    yield _sse("done", {"answer_chars": len(NO_CONTEXT_REPLY)})
                    return

                # 3) 流式生成
                total = 0
                async for delta in rag.astream(payload.message, hits):
                    # 客户端断开就停止，避免白跑几分钟推理
                    if await request.is_disconnected():
                        logger.info("客户端已断开，终止生成")
                        return
                    total += len(delta)
                    yield _sse("token", {"text": delta})

                yield _sse("done", {"answer_chars": total})

            except Exception as exc:
                logger.exception("流式对话失败")
                yield _sse("error", {"message": f"服务端处理失败：{type(exc).__name__}"})

        return StreamingResponse(stream(), media_type=SSE_MEDIA, headers=SSE_HEADERS)

    # ---------------------------------------------------------------- 对话（非流式）
    @app.post("/chat/sync", tags=["chat"])
    async def chat_sync(payload: ChatRequest, request: Request) -> dict:
        """一次性返回完整答案，便于脚本与自动化测试。"""
        rag = get_rag(request)

        hits = await rag.aretrieve(payload.message)
        if not hits:
            return {"answer": NO_CONTEXT_REPLY, "sources": []}

        answer = await rag.answer(payload.message, hits)
        return {"answer": answer, "sources": [h.to_dict() for h in hits]}

    # ---------------------------------------------------------------- 纯检索
    @app.post("/search", tags=["chat"])
    async def search(payload: SearchRequest, request: Request) -> dict:
        """只检索不生成。排查"答案不对"时先看这里：

        这里的结果就不相关 → 问题在检索（分块/嵌入/融合/重排）；
        这里对但答案不对 → 问题在提示词或生成模型。
        """
        rag = get_rag(request)

        hits = await rag.aretrieve(payload.query)
        return {
            "query": payload.query,
            "count": len(hits),
            "results": [h.to_dict() for h in hits],
        }

    # ---------------------------------------------------------------- 入库
    @app.post("/ingest", tags=["ingest"])
    async def ingest_path(payload: IngestPathRequest, request: Request) -> dict:
        """把服务端某个路径导入知识库。同一文件重复入库是幂等的。"""
        rag = get_rag(request)

        target = Path(payload.path)
        if not target.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"路径不存在：{payload.path}")

        report = await ingest(rag.settings, rag.store, rag.embedder,
                              target, payload.recursive)
        return report.to_dict()

    @app.post("/ingest/upload", tags=["ingest"])
    async def ingest_upload(file: UploadFile, request: Request) -> dict:
        """上传文件后入库。落盘再解析，便于事后排查问题文件。"""
        rag = get_rag(request)

        if not file.filename:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "缺少文件名")

        # 只取文件名部分，剥掉任何目录成分，防止目录穿越
        safe_name = Path(file.filename).name
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "文件超过 50MB")

        upload_dir = Path(rag.settings.data_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / safe_name
        target.write_bytes(content)
        logger.info("已保存上传文件：%s", target)

        report = await ingest(rag.settings, rag.store, rag.embedder,
                              target, recursive=False)
        return report.to_dict()

    # ---------------------------------------------------------------- 概况
    @app.get("/stats", tags=["ingest"])
    async def stats(request: Request) -> dict:
        """知识库规模与当前生效的检索配置。"""
        rag = get_rag(request)
        cfg = rag.settings
        return {
            "collection": cfg.milvus_collection,
            "server_version": rag.store.server_version,
            "chunks": rag.store.count(),
            "embedding_model": cfg.embedding_model,
            "embedding_dim": cfg.embedding_dim,
            "reranker_enabled": rag.rerank_enabled,
        }


# ===========================================================================
# SSE 工具
# ===========================================================================
def _sse(event: str, payload: object) -> str:
    """编码一个 SSE 帧。

    ``ensure_ascii=False`` 是刻意的：中文直接以 UTF-8 输出而不是 \\uXXXX 转义，
    响应体积能小约 3 倍，curl 调试时也肉眼可读。
    """
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
