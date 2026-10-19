"""跨层共享的领域模型。

把"检索到的分块"定义成显式数据结构，而不是到处传裸字典，好处是：
  * 类型可检查，IDE 能补全；
  * 溯源信息（来源、分块序号、各阶段排名/分数）随数据一起流动，
    便于前端展示引用、也便于排查"为什么这条被排到前面"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    """一条被检索命中的文本分块，携带完整的溯源与打分信息。"""

    # ---- 内容与出处 ----
    text: str
    """分块正文，最终会拼进提示词。"""

    doc_id: str = ""
    """文档级 ID（同一文档的所有分块共享）。用于按文档删除/去重。"""

    source: str = ""
    """来源标识，通常是文件路径或文件名，用于前端展示引用。"""

    chunk_index: int = -1
    """该分块在原文中的序号，便于按顺序回看上下文。"""

    # ---- 打分与排名（用于可观测性与调试）----
    score: float = 0.0
    """当前生效的综合得分。重排开启时为重排分数，否则为融合分数。"""

    dense_rank: int | None = None
    """在稠密分支中的名次（0 起）。未命中该分支时为 None。"""

    sparse_rank: int | None = None
    """在 BM25 稀疏分支中的名次（0 起）。未命中该分支时为 None。"""

    rrf_score: float | None = None
    """RRF 融合分数。"""

    rerank_score: float | None = None
    """交叉编码器给出的相关性分数。"""

    @property
    def is_hybrid_hit(self) -> bool:
        """是否被稠密与稀疏两个分支同时命中。

        同时命中通常意味着该分块既语义相关、又包含关键词，是质量信号，
        前端可以标出来。
        """
        return self.dense_rank is not None and self.sparse_rank is not None

    def to_public_dict(self) -> dict[str, Any]:
        """转为可安全返回给前端的字典。

        刻意不包含全文（正文可能很长），只给出摘要与溯源信息，
        正文单独通过 snippets 字段按需提供。
        """
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf_score": round(self.rrf_score, 6) if self.rrf_score is not None else None,
            "rerank_score": (
                round(self.rerank_score, 4) if self.rerank_score is not None else None
            ),
            "is_hybrid_hit": self.is_hybrid_hit,
            # 引用片段：截断避免响应体过大，前端展示引用卡片足够
            "snippet": self.text[:280] + ("…" if len(self.text) > 280 else ""),
        }

    def to_dict(self) -> dict[str, Any]:
        """完整字典（含正文），用于日志与调试。"""
        return asdict(self)


@dataclass(slots=True)
class IngestReport:
    """一次入库任务的结果统计"""
    # 扫描到的文件数
    files_scanned: int = 0
    # 成功解析的文件数
    files_loaded: int = 0
    # 写入 Milvus 的分块总数
    chunks_written: int = 0
    # 被跳过的文件及原因
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
