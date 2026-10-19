"""聊天相关路由：流式对话、非流式对话、纯检索调试。

三个接口各有明确用途：
  * ``POST /chat``       流式，面向 Web UI，是主接口；
  * ``POST /chat/sync``  非流式，便于脚本调用与自动化测试；
  * ``POST /search``     只做检索不做生成，用于观察混合检索与重排的中间结果。
    这是排查"答案不对"的第一步 —— 先确认检索对不对，再怀疑模型。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_app.api.sse import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_SOURCES,
    EVENT_TOKEN,
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
    sse_frame,
)
from rag_app.deps import AppContext, get_context
from rag_app.llm.prompts import NO_CONTEXT_REPLY

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """对话请求。"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户问题",
        examples=["混合检索里的 RRF 是怎么工作的？"],
    )
    include_sources: bool = Field(
        default=True,
        description="是否在回答前先推送检索来源（前端据此渲染引用卡片）",
    )


class SearchRequest(BaseModel):
    """纯检索请求。"""

    query: str = Field(..., min_length=1, max_length=4000, description="查询文本")


class SearchResponse(BaseModel):
    """纯检索响应，暴露完整的分支名次与分数，便于调试排序。"""

    query: str
    count: int
    fusion_mode: str
    results: list[dict]


class ChatSyncResponse(BaseModel):
    """非流式对话响应。"""

    answer: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _ensure_ready(ctx: AppContext) -> None:
    """检查检索链路是否可用，不可用则返回 503 并说明原因。

    Args:
        ctx: 应用上下文。

    Raises:
        HTTPException: 上下文未就绪（503），携带具体的初始化失败原因。
    """
    if not ctx.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ctx.startup_error or "检索服务尚未就绪",
        )


def _sources_payload(ctx: AppContext, chunks: list) -> dict:
    """构造 sources 事件的负载。"""
    return {
        "count": len(chunks),
        "fusion_mode": ctx.fusion_mode,
        "sources": [c.to_public_dict() for c in chunks],
    }


# ---------------------------------------------------------------------------
# POST /chat —— 流式（SSE）
# ---------------------------------------------------------------------------
@router.post(
    "/chat",
    summary="流式对话（SSE）",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE 事件流"},
        503: {"description": "检索服务未就绪"},
    },
)
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    ctx: AppContext = Depends(get_context),
) -> StreamingResponse:
    """执行 RAG 并流式返回答案。

    事件序列::

        event: sources   -> 检索到的引用列表（先发，让前端立刻渲染引用）
        event: token     -> 文本增量，可能多次
        event: done      -> 结束

    出错时发送 ``event: error``，而不是直接断开连接 ——
    因为响应头早已发出，无法再改 HTTP 状态码，只能通过事件类型告知前端。
    """
    _ensure_ready(ctx)
    assert ctx.retriever is not None and ctx.chat is not None

    async def event_stream() -> AsyncIterator[str]:
        """SSE 事件生成器。"""
        try:
            # ---- 1) 检索 ----
            chunks = await ctx.retriever.retrieve(payload.message)

            if payload.include_sources:
                # 先推送来源：用户能更早看到"找到了哪些材料"，
                # 即使生成较慢，界面也不是空白的
                yield sse_frame(EVENT_SOURCES, _sources_payload(ctx, chunks))

            # ---- 2) 检索为空：直接给兜底文案，不调用模型 ----
            # 这一步既省一次推理，也避免模型在没有依据时硬编答案
            if not chunks:
                logger.info("检索为空，返回兜底文案")
                yield sse_frame(EVENT_TOKEN, {"text": NO_CONTEXT_REPLY})
                yield sse_frame(EVENT_DONE, {"answer_chars": len(NO_CONTEXT_REPLY)})
                return

            # ---- 3) 流式生成 ----
            total_chars = 0
            async for delta in ctx.chat.astream_answer(payload.message, chunks):
                # 客户端已断开时及时停止，避免继续消耗模型算力。
                # 对本地 7B 模型来说这一点很重要：一次无人接收的生成
                # 会白占几十秒 GPU/CPU。
                if await request.is_disconnected():
                    logger.info("客户端已断开，终止生成")
                    return

                total_chars += len(delta)
                yield sse_frame(EVENT_TOKEN, {"text": delta})

            yield sse_frame(EVENT_DONE, {"answer_chars": total_chars})

        except Exception as exc:
            # 记录完整堆栈，但只把简要信息发给前端（不泄露内部细节）
            logger.exception("流式对话处理失败")
            yield sse_frame(EVENT_ERROR, {"message": f"服务端处理失败：{type(exc).__name__}"})

    return StreamingResponse(
        event_stream(),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# POST /chat/sync —— 非流式
# ---------------------------------------------------------------------------
@router.post(
    "/chat/sync",
    summary="非流式对话",
    response_model=ChatSyncResponse,
)
async def chat_sync(
    payload: ChatRequest,
    ctx: AppContext = Depends(get_context),
) -> ChatSyncResponse:
    """一次性返回完整回答。

    便于自动化测试与命令行调试；面向用户的界面应使用 ``/chat``。
    """
    _ensure_ready(ctx)
    assert ctx.retriever is not None and ctx.chat is not None

    chunks = await ctx.retriever.retrieve(payload.message)

    if not chunks:
        return ChatSyncResponse(answer=NO_CONTEXT_REPLY, sources=[])

    answer = await ctx.chat.answer(payload.message, chunks)
    return ChatSyncResponse(
        answer=answer,
        sources=[c.to_public_dict() for c in chunks],
    )


# ---------------------------------------------------------------------------
# POST /search —— 只检索，不生成
# ---------------------------------------------------------------------------
@router.post(
    "/search",
    summary="纯检索（调试用）",
    response_model=SearchResponse,
)
async def search_only(
    payload: SearchRequest,
    ctx: AppContext = Depends(get_context),
) -> SearchResponse:
    """只执行检索链路，返回带完整打分明细的结果。

    这个接口是排查问题的关键工具：
      * 若这里的结果就不相关，问题在检索（分块、嵌入、BM25、融合）；
      * 若这里的结果正确但最终答案不对，问题在提示词或生成模型。

    返回中的 ``dense_rank`` / ``sparse_rank`` 能直接看出某条结果是
    语义命中的还是关键词命中的，``is_hybrid_hit`` 则标记两者兼顾的高质量结果。
    """
    _ensure_ready(ctx)
    assert ctx.retriever is not None

    chunks = await ctx.retriever.retrieve(payload.query)

    return SearchResponse(
        query=payload.query,
        count=len(chunks),
        fusion_mode=ctx.fusion_mode,
        results=[c.to_public_dict() for c in chunks],
    )
