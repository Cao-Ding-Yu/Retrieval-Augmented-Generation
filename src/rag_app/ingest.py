"""
入库：文件 → 分块 → 向量 → Milvus

两处值得注意的设计：
1. 幂等更新机制：基于文件路径的 SHA1 散列生成唯一 doc_id，采用先删后插策略实现文档的幂等更新。
    确保重复入库不产生冗余数据，文件修改后可清除旧版本分块，保障数据一致性。
2. 异步非阻塞架构：针对读盘、PDF 解析、向量化及 Milvus 写入等 I/O 密集型同步操作，统一交由线程池执行。
    避免阻塞 FastAPI 事件循环，保障 SSE 流式输出期间系统的并发处理能力。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import Settings
from rag_app.store import MilvusStore

logger = logging.getLogger(__name__)

# 支持的文本类后缀 / PDF后缀
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
PDF_SUFFIXES = {".pdf"}

# 单次调用嵌入模型的文本条数
EMBED_BATCH = 32

# 切分标识符（按照优先级）
SEPARATORS = [
    "\n\n", "\n",
    "。", "！", "？", "；",
    ". ", "! ", "? ", "; ",
    "，", ", ", " ",
    "",
]


@dataclass(slots=True)
class IngestReport:
    """一次入库的结果统计。"""

    files_loaded: int = 0
    chunks_written: int = 0
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files_loaded": self.files_loaded,
            "chunks_written": self.chunks_written,
            "skipped": self.skipped,
        }


async def ingest(
    settings: Settings,
    store: MilvusStore,
    embedder,
    path: str | Path,
    recursive: bool = True,
) -> IngestReport:
    """把一个文件或目录导入知识库。

    Args:
        settings: 全局配置（用到 chunk_size / chunk_overlap / embedding_dim）。
        store: Milvus 存储。
        embedder: 需提供 ``embed_documents(list[str]) -> list[list[float]]``。
        path: 文件或目录。
        recursive: 目录是否递归子目录。

    Returns:
        IngestReport。
    """
    target = Path(path)
    report = IngestReport()

    # ---- 1. 加载 + 切分（CPU 密集，放线程池）----
    documents, skipped = await asyncio.to_thread(_load, target, recursive)
    report.skipped.extend(skipped)
    if not documents:
        logger.warning("路径 %s 未产出任何可入库内容", target)
        return report

    chunks = await asyncio.to_thread(_split, documents, settings)

    # ---- 2. 按文档分组：同一文档的分块要一起处理（先删旧、再写新）----
    grouped: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.metadata.get("doc_id", ""))].append(chunk)
    report.files_loaded = len(grouped)

    for doc_id, doc_chunks in grouped.items():
        try:
            written = await _ingest_one(store, embedder, doc_id, doc_chunks)
            report.chunks_written += written
        except Exception as exc:
            # 单个文档失败不该中断整批
            logger.exception("文档 %s 入库失败", doc_id)
            report.skipped.append(f"doc_id={doc_id[:8]}…: {exc}")

    logger.info(
        "入库完成：成功 %d 个文档，写入 %d 个分块", report.files_loaded, report.chunks_written
    )
    return report


async def _ingest_one(store: MilvusStore, embedder, doc_id: str, chunks: list[Document]) -> int:
    """写入单个文档的全部分块：向量化 → 先删旧 → 再插入。

    删除依据是 ``doc_id``：它由文件路径哈希得到，不随内容变化，
    所以"重新入库同一个文件"能正确替换掉它之前的全部分块。
    """
    texts = [c.page_content for c in chunks]

    # 向量化（阻塞，放线程池）
    vectors: list[list[float]] = await asyncio.to_thread(_embed, embedder, texts)

    rows = [
        {
            "text": chunk.page_content,
            "doc_id": str(chunk.metadata.get("doc_id", doc_id)),
            "source": str(chunk.metadata.get("source", "")),
            "chunk_index": int(chunk.metadata.get("chunk_index", -1)),
            "dense": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    deleted = await asyncio.to_thread(store.delete_doc, doc_id)
    if deleted:
        logger.debug("文档 %s 覆盖写入，先删了 %d 条旧分块", doc_id[:8], deleted)

    return await asyncio.to_thread(store.insert, rows)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------
def _doc_id(path: Path) -> str:
    """由绝对路径派生稳定的文档 ID（sha1）。

    用路径哈希而不是 uuid：同一文件重复入库得到同一个 id，
    才能实现"先删后插"的幂等更新（文件内容改了也能正确替换旧分块）。
    """
    return hashlib.sha1(str(path.resolve()).lower().encode("utf-8")).hexdigest()


def _iter_files(root: Path, recursive: bool) -> list[Path]:
    """列出受支持的文件（排序保证入库顺序可复现）。"""
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in root.glob(pattern)
        if p.is_file()
        and p.suffix.lower() in (TEXT_SUFFIXES | PDF_SUFFIXES)
        and not any(part.startswith(".") for part in p.parts)
    ]
    return sorted(files)


def _load(root: Path, recursive: bool) -> tuple[list[Document], list[str]]:
    """解析文件成 Document 列表。单个文件失败只跳过，不中断整批。"""
    docs: list[Document] = []
    skipped: list[str] = []

    for file in _iter_files(root, recursive):
        try:
            loaded = _load_file(file)
        except Exception as exc:
            logger.exception("加载失败：%s", file)
            skipped.append(f"{file.name}: 解析异常({exc})")
            continue

        if not loaded:
            skipped.append(f"{file.name}: 未提取到文本")
            continue
        docs.extend(loaded)

    return docs, skipped


def _load_file(file: Path) -> list[Document]:
    """解析单个文件。PDF 按页拆，便于引用定位到具体页码。"""
    doc_id = _doc_id(file)
    suffix = file.suffix.lower()

    if suffix in PDF_SUFFIXES:
        return _load_pdf(file, doc_id)
    if suffix in TEXT_SUFFIXES:
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 中文环境的老文件常是 GBK，兜底一次避免整批被单个文件打断
            logger.warning("%s 非 UTF-8，回退 GBK", file.name)
            text = file.read_text(encoding="gbk", errors="replace")
        text = text.strip()
        if not text:
            return []
        return [Document(page_content=text,
                         metadata={"doc_id": doc_id, "source": str(file)})]
    return []


def _load_pdf(file: Path, doc_id: str) -> list[Document]:
    """按页解析 PDF（纯图片页没有文字层，会跳过）。"""
    from pypdf import PdfReader

    docs: list[Document] = []
    for page_number, page in enumerate(PdfReader(str(file)).pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:  # pragma: no cover - 取决于 PDF 内部结构
            logger.warning("%s 第 %d 页解析失败：%s", file.name, page_number, exc)
            continue
        if text:
            docs.append(Document(page_content=text,
                                 metadata={"doc_id": doc_id, "source": str(file)}))
    return docs


def _split(documents: list[Document], settings: Settings) -> list[Document]:
    """切分文档，并为每个分块补上文档内序号。"""
    splitter = RecursiveCharacterTextSplitter(
        length_function=len,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
        strip_whitespace=True,
    )
    chunks = splitter.split_documents(documents)

    # chunk_index 是"在该文档内"的序号，用于按顺序回看原文
    counters: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id", ""))
        chunk.metadata["chunk_index"] = counters[doc_id]
        counters[doc_id] += 1

    logger.info("切分：%d 个文档 → %d 个分块", len(documents), len(chunks))
    return chunks


def _embed(embedder, texts: list[str]) -> list[list[float]]:
    """分批调用嵌入模型。"""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embedder.embed_documents(texts[start : start + EMBED_BATCH]))
    return vectors
