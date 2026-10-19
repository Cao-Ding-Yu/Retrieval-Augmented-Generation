"""
文档加载：把磁盘上的文件解析成带元数据的 LangChain Document。

设计考虑：
    * **稳定 doc_id**：用文件绝对路径的 sha1 派生，而不是用自增序号或 uuid
        同一文件重复入库得到同一个 doc_id，可以实现"先删后插"的幂等更新，在增量场景下正确识别"文档被替换"
    * **页码拆分**：PDF 按页产出 Document，便于把引用精确定位到页码，而非单文件级引用
    * **失败不中断**：单个文件解析失败只记录并跳过，不影响整批入库

改进方向：
    PDF 解析可以引入 OCR 或在线解析工具来覆盖图文数据解析

加载结构：
    Document(
        page_content='正文文本',
        metadata={
            'doc_id': 'hash_id',
            'source': '文件相对路径',
            'page': 0
        }
    )
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 支持的文本类文件后缀
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
# 支持的 PDF 后缀
PDF_SUFFIXES = {".pdf"}
# 支持的总后缀集合
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def compute_doc_id(path: Path) -> str:
    """
    由文件绝对路径派生稳定的文档 ID
    使用 resolve() 归一化路径，保证同一文件传入 doc_id 一致

    Args:
        path: 文件路径

    Returns:
        40 位十六进制 sha1 哈希值
    """
    normalized = str(path.resolve()).lower()  # 标准化文件绝对路径
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _read_text_file(path: Path) -> str:
    """读取纯文本文件，优先 UTF-8，失败后退回 GBK """
    try: return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("文件 %s 非 UTF-8 编码，回退到 GBK 解码", path.name)
        return path.read_text(encoding="gbk", errors="replace")


def load_pdf(path: Path, doc_id: str) -> list[Document]:
    """
    按页解析 PDF，逐页产出 Document 便于定位文档所在页数

    Args:
        path: PDF 文件路径
        doc_id: 稳定文档 ID

    Returns:
        非空页面解析成的 [Document] 列表
    """
    # 延迟导入：pypdf 体量较大，只有真正处理 PDF 时才需要
    from pypdf import PdfReader

    reader = PdfReader(str(path))   # 文档阅读器
    docs: list[Document] = list()   # 解析结果列表

    for page_number, page in enumerate(reader.pages, start=1):
        try: text = page.extract_text().strip() or ""
        except Exception as e:  # 防止单页解析失败中断解析进程
            logger.warning("PDF %s 第 %d 页解析失败：%s", path.name, page_number, e)
            continue
        # 纯图片页没有文字；如需覆盖应引入 OCR
        if not text: continue
        # 解析封装为Document并追加至列表
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "source": str(path),
                    "page": page_number,
                },
            )
        )

    return docs


def load_file(path: Path) -> list[Document]:
    """
    加载单个文件，返回解析的 Document 列表

    Args:
        path: 文件路径

    Returns:
        Document 列表（不支持的格式返回空列表）
    """
    # 文件后缀预处理，用于选择文件解析方式
    suffix = path.suffix.lower()
    # 将文档绝对路径生成的sha1哈希值作为文档id
    doc_id = compute_doc_id(path)
    # 文档属于 PDF 类型
    if suffix in PDF_SUFFIXES:
        return load_pdf(path, doc_id)
    # 文档属于 文本文件 类型
    if suffix in TEXT_SUFFIXES:
        text = _read_text_file(path).strip()
        if not text: return []
        return [
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "source": str(path),
                    "page": 0,
                },
            )
        ]
    return []


def iter_files(root: Path, recursive: bool = True) -> list[Path]:
    """
    列出目录下所有支持的文件

    Args:
        root: 目录或单个文件
        recursive: 是否递归子目录

    Returns:
        排序后的文件路径列表（排序保证入库顺序可复现）
    """
    # 如果root是文件，则直接返回
    if root.is_file(): return [root]
    # 如果root是目录，则默认递归检查目录内所有文件，筛选合法文件排序后返回
    pattern = "**/*" if recursive else "*"
    files = [
        # path提取：包括文件和目录
        p for p in root.glob(pattern)
        # 筛选条件1：是文件而非目录
        if p.is_file()
        # 筛选条件2：文件后缀合法
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        # 筛选条件3：非隐藏目录（.hidden_dir）
        and not any(part.startswith(".") for part in p.parts)
    ]
    # 排序让同批入库数据顺序稳定，便于复现与排查
    return sorted(files)


def load_directory(root: Path, recursive: bool = True) -> tuple[list[Document], list[str]]:
    """
    加载目录下所有受支持文件

    Args:
        root: 目录或单个文件路径
        recursive: 是否递归子目录

    Returns:
        (成功加载的 Document 列表, 跳过原因说明列表)
        跳过原因形如 "xxx.pdf: 未提取到文本"
    """
    # 成功加载的文档集合
    documents: list[Document] = list()
    # 失败加载的说明列表
    skipped: list[str] = list()

    for path in iter_files(root, recursive=recursive):
        try: loaded = load_file(path)
        except Exception as e:  # 捕获加载异常，防止程序中断
            logger.exception("加载文件失败：%s", path)
            skipped.append(f"{path.name}: 解析异常({e})")
            continue
        if not loaded:
            skipped.append(f"{path.name}: 未提取到文本")
            continue
        documents.extend(loaded)
    return documents, skipped

