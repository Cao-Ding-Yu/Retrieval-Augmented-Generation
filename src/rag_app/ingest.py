"""
入库：文件 → 分块 → 向量 → Milvus

两处值得注意的设计：
1. 幂等更新机制：基于文件路径的 SHA1 散列生成唯一 doc_id，采用先删后插策略实现文档的幂等更新
    确保重复入库不产生冗余数据，文件修改后可清除旧版本分块，保障数据一致性
2. 异步非阻塞架构：针对读盘、PDF 解析、向量化及 Milvus 写入等 I/O 密集型同步操作，统一交由线程池执行
    避免阻塞 FastAPI 事件循环，保障 SSE 流式输出期间系统的并发处理能力
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
    """入库结果统计"""
    files_loaded: int = 0
    chunks_written: int = 0
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files_loaded": self.files_loaded,
            "chunks_written": self.chunks_written,
            "skipped": self.skipped,
        }


async def ingest(settings: Settings, store: MilvusStore, embedder, path: str | Path) -> IngestReport:
    """
    递归查找目录下文件并导入知识库

    Args:
        settings: 全局配置
        store: Milvus 操作包装
        embedder: 提供嵌入操作
        path: 文件或目录

    Returns:
        IngestReport
    """
    target = Path(path)
    report = IngestReport()

    # ---- 1. 加载 ----
    documents, skipped = await asyncio.to_thread(_load, target)
    report.skipped.extend(skipped)
    if not documents:
        logger.warning("路径 %s 未产出可入库内容", target)
        return report

    # ---- 2. 切分 ----
    chunks = await asyncio.to_thread(_split, documents, settings)

    # ---- 3. 按文档分组：同一文档的分块要一起处理（先删旧、再写新）----
    grouped: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.metadata.get("doc_id"))].append(chunk)
    report.files_loaded = len(grouped)

    for doc_id, doc_chunks in grouped.items():
        try:
            written = await _ingest_one(store, embedder, doc_id, doc_chunks)
            report.chunks_written += written
        except Exception as exc:
            logger.exception("文档 %s 入库失败", doc_id)
            report.skipped.append(f"doc_id={doc_id[:8]}…: {exc}")

    logger.info("入库完成：成功 %d 个文档，写入 %d 个分块", report.files_loaded, report.chunks_written)
    return report


async def _ingest_one(store: MilvusStore, embedder, doc_id: str, chunks: list[Document]) -> int:
    """写入单个文档的分块：向量化 → 先删旧 → 再插入"""
    texts = [c.page_content for c in chunks]

    vectors: list[list[float]] = await asyncio.to_thread(_embed, embedder, texts)

    rows = [
        {
            "text": chunk.page_content, # 文本内容
            "doc_id": str(chunk.metadata.get("doc_id", doc_id)), # 文档id
            "source": str(chunk.metadata.get("source", "")), # 文档路径
            "chunk_index": int(chunk.metadata.get("chunk_index", -1)), # 切块索引
            "page": int(chunk.metadata.get("page") or 0), # 切块所属页码
            "dense": vector, # 文本向量
        } for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    deleted = await asyncio.to_thread(store.delete_doc, doc_id)
    if deleted: logger.debug("文档 %s 覆盖写入，先删了 %d 条旧分块", doc_id[:8], deleted)

    return await asyncio.to_thread(store.insert, rows)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------
def _doc_id(path: Path) -> str:
    """
    由绝对路径派生稳定的文档 ID（sha1哈希）
    相较于 uuid：同一文件重复入库得到同一个 id，能实现"先删后插"的幂等更新
    """
    return hashlib.sha1(str(path.resolve()).lower().encode("utf-8")).hexdigest()


def _iter_files(root: Path) -> list[Path]:
    """
    列出待入库文件列表

    目录递归查找；只保留受支持的文件后缀
    排序是为入库顺序可复现
    """
    if root.is_file(): return [root]

    files: list[Path] = list()
    for p in root.glob("**/*"):
        # 排除文件夹目录与不合法文件
        if not p.is_file() or p.suffix.lower() not in (TEXT_SUFFIXES | PDF_SUFFIXES): continue
        # 排除以'.'开头的隐藏或配置文件
        if any(part.startswith(".") for part in p.relative_to(root).parts): continue
        files.append(p)
    return sorted(files)


def _load(root: Path) -> tuple[list[Document], list[str]]:
    """加载文档列表中的文件"""
    docs: list[Document] = list()
    skipped: list[str] = list()

    for file in _iter_files(root):
        try: loaded = _load_file(file)
        except Exception as exc:
            logger.exception("加载失败：%s", file)
            skipped.append(f"{file.name}: 解析异常({exc})")
            continue
        if not loaded:  # 文件提取内容为空
            skipped.append(f"{file.name}: 未提取到文本")
            continue
        docs.extend(loaded)
    return docs, skipped


def _load_file(file: Path) -> list[Document]:
    """解析单个文件（PDF 按页拆，便于引用定位到具体页码）"""
    doc_id = _doc_id(file)
    suffix = file.suffix.lower()

    if suffix in PDF_SUFFIXES:  # 文件为pdf类
        return _load_pdf(file, doc_id)
    if suffix in TEXT_SUFFIXES: # 文件为文档类
        try: text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError: # 可能使用的是GBK编码
            logger.warning("%s 非 UTF-8，回退 GBK", file.name)
            # 遇到无法解析的字符替换成�，防止报错
            text = file.read_text(encoding="gbk", errors="replace")
        text = text.strip()
        if not text: return list()
        return [
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "source": str(file)
                }
            )
        ]
    return list()


def _load_pdf(file: Path, doc_id: str) -> list[Document]:
    """按页解析 PDF"""
    from pypdf import PdfReader

    docs: list[Document] = list()
    for page_number, page in enumerate(PdfReader(str(file)).pages, start=1):
        try: text = (page.extract_text() or "").strip()
        except Exception as exc:
            logger.warning("%s 第 %d 页解析失败：%s", file.name, page_number, exc)
            continue
        if text:  # 解析内容不为空
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": doc_id,
                        "source": str(file),
                        "page": page_number,
                    }
                )
            )
    return docs


def _split(documents: list[Document], settings: Settings) -> list[Document]:
    """切分文档，并为每个分块补上文档内序号"""
    splitter = RecursiveCharacterTextSplitter(
        length_function=len,                    # 按字符切分
        chunk_size=settings.chunk_size,         # 分块最大长度
        chunk_overlap=settings.chunk_overlap,   # 重叠窗口大小
        separators=SEPARATORS,                  # 分割字符列表
        keep_separator=True,                    # 保留切分标点
        strip_whitespace=True,                  # 去除块尾占位符
    )
    chunks = splitter.split_documents(documents)

    # chunk_index 是"在该文档内"的序号，用于按顺序定位原文
    counters: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id"))
        chunk.metadata["chunk_index"] = counters[doc_id]
        counters[doc_id] += 1
    logger.info("切分：%d 个文档 → %d 个分块", len(documents), len(chunks))
    return chunks


def _embed(embedder, texts: list[str]) -> list[list[float]]:
    """分批调用嵌入模型"""
    vectors: list[list[float]] = list()
    for start in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embedder.embed_documents(texts[start : start + EMBED_BATCH]))
    return vectors
