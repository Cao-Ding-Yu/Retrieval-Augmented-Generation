"""
文本切分
切分策略直接决定检索上限：块太大则语义被稀释、命中后噪声多；块太小则单块信息不足、答案被切碎
这里的选择与理由：
    * **递归字符切分 + 中文标点分隔符**：因此显式加入中文标点，让切分优先发生在句子/分句边界上
    * **保留句读（keep_separator）**：让标点留在块内，避免出现缺尾标点的碎片
    * **记录 start_index**：把分块在原文中的字符偏移写进元数据，便于前端高亮原文位置，也便于调试

分割结构：
    {
        "page_content": "切分后的文本",
        "metadata": {
            "doc_id": "hash_id",
            "source": "文件相对路径",
            "page": 0,
            "start_index": 100,
            "chunk_index": 0,
        },
        "type": "Document"
    }
"""

from __future__ import annotations

import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# 分隔符按优先级从高到低排列
DEFAULT_SEPARATORS: list[str] = [
    "\n\n",     # 段落
    "\n",       # 换行
    "。",        # 中文句号
    "！",        # 中文叹号
    "？",        # 中文问号
    "；",        # 中文分号
    ". ",       # 英文句号（带空格，避免把小数点/缩写切开）
    "! ",       # 英文叹号（带空格，避免把小数点/缩写切开）
    "? ",       # 英文问号（带空格，避免把小数点/缩写切开）
    "; ",       # 英文分号（带空格，避免把小数点/缩写切开）
    "，",        # 中文逗号
    ", ",       # 英文逗号
    " ",        # 空格
    "",         # 兜底：逐字符切分，保证任何输入都能推进
]


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    构造文本切分器

    Args:
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 重叠窗口大小（字符数）

    Returns:
        RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        # 字符计数方式
        length_function=len,
        # 文本块长度上限
        chunk_size=chunk_size,
        # 重叠窗口大小
        chunk_overlap=chunk_overlap,
        # 分隔符递归切割列表
        separators=DEFAULT_SEPARATORS,
        # 保留标点，避免块尾缺失句读
        keep_separator=True,
        # 记录分块在原文中的起始偏移
        add_start_index=True,
        # 连续空白折叠
        strip_whitespace=True,
    )


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    切分文档，并为每个分块补齐溯源的元数据

    补的元数据：
      * ``chunk_index``：分块在**该文档内**的序号
      * ``start_index``：由切分器写入的原文偏移

    Args:
        documents: 待切分的 Document 列表
        chunk_size: 目标块大小
        chunk_overlap: 重叠窗口大小

    Returns:
        带完整溯源元数据的切分后 Document 列表
    """
    # 创建分割器并执行分割
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # 按 doc_id 维护计数器，生成文档内序号
    counters: dict[str, int] = dict()
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id", ""))
        index = counters.get(doc_id, 0)
        chunk.metadata["chunk_index"] = index
        counters[doc_id] = index + 1

    # 打印分割日志
    logger.info(
        "切分完成：%d 个输入文档 -> %d 个分块（chunk_size=%d, overlap=%d）",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
