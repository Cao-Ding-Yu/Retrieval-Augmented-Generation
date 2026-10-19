"""入库相关路由：目录入库、文件上传入库、知识库统计。

## 关于安全的说明

``POST /ingest`` 接受服务端本地路径，属于**高权限操作**：一旦暴露到公网，
等于把服务器文件系统的读取能力开放出去。因此：

  * 该接口默认只应在本地开发/内网使用；
  * 正式部署必须加鉴权（API Key / OAuth），或干脆不暴露该路由；
  * 上传接口对文件名做了净化，防止 ``../`` 逃逸。

这些约束写在这里，是因为"能跑通"和"能安全上线"是两件事，
面试中也常被追问。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from rag_app.deps import AppContext, get_context

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

# 单次上传的文件大小上限（50MB）。
# 目的是防止超大文件把内存吃满 —— UploadFile 虽然会落盘，
# 但后续 PDF 解析是整份读进内存的。
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# 上传文件存放目录（相对 DATA_DIR）
UPLOAD_SUBDIR = "uploads"


class IngestPathRequest(BaseModel):
    """按路径入库的请求。"""

    path: str = Field(
        ...,
        min_length=1,
        description="服务端本地文件或目录路径",
        examples=["./data"],
    )
    recursive: bool = Field(default=True, description="目录是否递归子目录")


class IngestResponse(BaseModel):
    """入库结果。"""

    files_scanned: int
    files_loaded: int
    chunks_written: int
    skipped: list[str]


class StatsResponse(BaseModel):
    """知识库统计。"""

    collection: str
    server_version: str
    fusion_mode: str
    embedding_model: str
    embedding_dim: int
    entities: int
    rerank_enabled: bool


def _ensure_ready(ctx: AppContext) -> None:
    """检查上下文可用性。"""
    if not ctx.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ctx.startup_error or "服务尚未就绪",
        )


def _to_response(report) -> IngestResponse:
    """把 IngestReport 转成响应模型。

    显式转换而不是直接返回 dataclass，是为了让 OpenAPI 文档里
    有完整的字段说明（FastAPI 需要 Pydantic 模型才能生成 schema）。
    """
    return IngestResponse(
        files_scanned=report.files_scanned,
        files_loaded=report.files_loaded,
        chunks_written=report.chunks_written,
        skipped=report.skipped,
    )


@router.post("/ingest", summary="按服务端路径入库", response_model=IngestResponse)
async def ingest_path(
    payload: IngestPathRequest,
    ctx: AppContext = Depends(get_context),
) -> IngestResponse:
    """把服务端某个文件或目录导入知识库。

    同一文件重复入库是**幂等**的：会先按 doc_id 删除旧分块再写入新分块，
    因此不会产生重复数据，文件更新后也不会残留旧版本内容。
    """
    _ensure_ready(ctx)
    assert ctx.ingest is not None

    target = Path(payload.path)
    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"路径不存在：{payload.path}",
        )

    report = await ctx.ingest.ingest_path(target, recursive=payload.recursive)
    return _to_response(report)


@router.post("/ingest/upload", summary="上传文件并入库", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(..., description="待入库的 PDF / TXT / MD 文件"),
    ctx: AppContext = Depends(get_context),
) -> IngestResponse:
    """接收上传文件，落盘后入库。

    落盘而不是直接在内存里解析，原因有两个：
      1. 入库服务按"路径"组织，落盘后可以复用同一套逻辑；
      2. 文件保留在磁盘上便于事后排查（原始文件比解析结果更好定位问题）。
    """
    _ensure_ready(ctx)
    assert ctx.ingest is not None

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少文件名",
        )

    # 净化文件名：只取最后一段，丢弃任何目录成分。
    # 若不做这一步，`../../etc/passwd` 这类文件名会造成目录穿越。
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    # 读取内容并校验大小
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件超过上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )

    upload_dir = Path(ctx.settings.data_dir) / UPLOAD_SUBDIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / safe_name

    # 以二进制写入，避免编码转换破坏 PDF
    target.write_bytes(content)
    logger.info("已保存上传文件：%s（%d 字节）", target, len(content))

    report = await ctx.ingest.ingest_path(target, recursive=False)
    return _to_response(report)


@router.get("/stats", summary="知识库统计", response_model=StatsResponse)
async def stats(ctx: AppContext = Depends(get_context)) -> StatsResponse:
    """返回集合规模与当前生效的检索配置。

    这些信息在演示时很有用：能直观说明"库里有多少分块"、
    "当前是服务端 BM25 还是客户端回退"、"重排是否开启"。
    """
    _ensure_ready(ctx)
    assert ctx.store is not None

    return StatsResponse(
        collection=ctx.settings.milvus_collection,
        server_version=ctx.store.server_version,
        fusion_mode=ctx.fusion_mode,
        embedding_model=ctx.settings.embedding_model,
        embedding_dim=ctx.settings.embedding_dim,
        entities=ctx.store.num_entities(),
        # NoOpReranker 意味着重排未生效（可能是配置关闭，也可能是加载失败）
        rerank_enabled=type(ctx.reranker).__name__ != "NoOpReranker",
    )
