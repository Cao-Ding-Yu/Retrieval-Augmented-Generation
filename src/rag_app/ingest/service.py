"""
入库编排：chunks -> 向量化 -> 写入 Milvus

流程顺序与关键决策：
    1. **先删后插实现幂等更新**：按 doc_id 删除该文档的旧分块，再写新分块
        同一文件重复入库不会产生重复数据，文件内容变更后也不会残留旧版本分块
    2. **按文档逐个提交**：不把所有文档攒成一个大批次，大语料下内存占用可控
    3. **批量向量化**：调用嵌入模型按固定小批次进行，避免一次性输入导致超时或溢出
    4. **阻塞调用放到线程池**：Ollama 的 Python 客户端是同步的，
        直接放在 async 函数里会阻塞事件循环，使 SSE 流式响应失去意义
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from rag_app.config import Settings
from rag_app.ingest.loader import load_directory
from rag_app.ingest.splitter import split_documents
from rag_app.models import IngestReport
from rag_app.retrieval.milvus_store import MilvusStore

logger = logging.getLogger(__name__)

# 单次调用嵌入模型的文本条数上限
EMBED_BATCH_SIZE = 32


class IngestService:
    """
    文档入库

    Args:
        store: Milvus 存储封装，负责实际写入与按文档删除
        embedder: 提供 embed_documents 的嵌入器（OllamaEmbeddings）
        settings: 全局配置
    """

    def __init__(self, store: MilvusStore, embedder: Any, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    async def ingest_path(self, path: str | Path, recursive: bool = True) -> IngestReport:
        """
        入库一个文件或目录

        异步接口，内部把阻塞操作统一切到线程池执行，确保不阻塞 FastAPI 的事件循环

        Args:
            path: 文件或目录路径
            recursive: 目录是否递归查找

        Returns:
            本次入库的统计报告
            class IngestReport:
                files_scanned → 扫描到的文件数
                files_loaded → 成功解析的文件数
                chunks_written → 写入的分块总数
                skipped → 被跳过的文件及原因
        """
        # 加载与切分整体丢到线程
        chunks, skipped, files_scanned, files_loaded = await asyncio.to_thread(
            self._load_and_split, Path(path), recursive
        )

        # 创建入库报告
        report = IngestReport(files_scanned=files_scanned, skipped=skipped, files_loaded=files_loaded, chunks_written=0)

        # 分块结果为空
        if not chunks:
            logger.warning("路径 %s 未产出任何分块，入库结束", path)
            return report

        # 按 doc_id 分组：同一文档分块一起处理
        grouped: dict[str, list[Document]] = defaultdict(list)
        for chunk in chunks:
            grouped[str(chunk.metadata.get("doc_id", ""))].append(chunk)

        # 逐文档提交，隔离单文档失败
        for doc_id, doc_chunks in grouped.items():
            try:    # 写入数据并返回写入chunk条数
                written = await self._ingest_one_document(doc_id, doc_chunks)
                report.chunks_written += written
            except Exception as e:
                logger.exception("文档 %s 入库失败", doc_id)
                report.skipped.append(f"doc_id={doc_id[:8]}…: 写入失败({e})")

        logger.info(
            "入库完成：扫描 %d 文件，成功 %d 文档，写入 %d 分块",
            report.files_scanned,
            report.files_loaded,
            report.chunks_written,
        )
        return report

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------
    def _load_and_split(self, path: Path, recursive: bool) -> tuple[list[Document], list[str], int, int]:
        """
        加载 + 切分

        Returns:
            (分块列表, 跳过原因, 扫描到的文件数, 成功加载的文件数)
        """
        # 文档文件读取
        documents, skipped = load_directory(path, recursive=recursive)
        # 文档片段分割
        chunks = split_documents(
            documents,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        loaded = {d.metadata.get("doc_id") for d in documents}
        files_scanned = len(loaded) + len(skipped)
        return chunks, skipped, files_scanned, len(loaded)

    async def _ingest_one_document(self, doc_id: str, chunks: list[Document]) -> int:
        """
        写入单个文档的全部分块，返回写入条数

        步骤： 向量化 -> 组装行 -> 删除旧分块 -> 插入
        """
        # 1) 文本向量化
        texts = [c.page_content for c in chunks]
        vectors: list[list[float]] = await asyncio.to_thread(self._embed_texts, texts)

        # 2) 组装行数据
        rows: list[dict[str, Any]] = list()
        for chunk, vector in zip(chunks, vectors, strict=True):
            meta = chunk.metadata
            rows.append(
                {
                    "text": chunk.page_content,
                    "doc_id": str(meta.get("doc_id", doc_id)),
                    "source": str(meta.get("source", "")),
                    "chunk_index": int(meta.get("chunk_index", -1)),
                    "dense": vector,
                }
            )

        # 3) 先删后插：保证重复入库不会产生重复分块
        deleted = await asyncio.to_thread(self._store.delete_by_doc_id, doc_id)
        if deleted: logger.debug("文档 %s 覆盖写入，先删除了 %d 条旧分块", doc_id[:8], deleted)
        written = await asyncio.to_thread(self._store.insert, rows)
        return written

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """分批调用嵌入模型，返回与输入等长的向量列表"""
        vectors: list[list[float]] = list()
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors.extend(self._embedder.embed_documents(batch))
        return vectors
