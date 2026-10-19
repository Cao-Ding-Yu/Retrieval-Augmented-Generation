"""检索质量指标（纯函数，零外部依赖）。

## 为什么需要这些指标

RAG 的效果由两段决定：**检索**（把对的材料找出来）和**生成**（据此作答）。
RAGAS 那类 LLM 评判指标衡量的是后半段，但它有两个问题：

  1. 需要额外装依赖、且要一个够强的"裁判模型"（本地 7B~14B 常常给不出稳定 JSON）；
  2. 慢 —— 每个样本要跑好几次 LLM 调用。

而**检索指标**不需要任何模型，几十毫秒就能算完，却能回答一个更根本的问题：
"我调 `CHUNK_SIZE`、开/关重排、换 `RRF_K`，检索真的变好了吗？"

本项目把检索指标独立成模块的用意就在这里：**先让检索可量化，再去追生成质量**。
没有这层，调参只能靠"感觉好像好了一点"。

## 相关性判定

以**来源文件**（`source` 的 basename）为相关性单位，而不是分块 ID。
原因：同一个文件会被切成很多块，用分块 ID 做标注既繁琐又不稳定
（改一次 `chunk_size` 全部失效）。按文件标注则稳健得多，也更贴近
"用户期望答案出自哪篇文档"这个真实判断。

## 指标口径

设一次查询召回了 top-k 个分块，把它们映射成来源文件集合（去重、保序）:

  * ``hit_rate@k``：top-k 里**至少命中一个**期望来源的查询占比。看"有没有找到"。
  * ``recall@k``  ：命中的期望来源数 / 期望来源总数，再对查询取平均。看"找全没有"。
  * ``mrr@k``     ：第一个命中位置的倒数（1/rank）对查询取平均。看"排得靠不靠前"。
  * ``ndcg@k``    ：二值相关性下的归一化折损累计增益。看"整体排序质量"。

四个指标一起看才有意义：`hit_rate` 高但 `mrr` 低，说明找到了但排在后面
（重排没做好）；`hit_rate` 低则是召回本身有问题。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


def _basename(path: str) -> str:
    """取路径的文件名部分，并统一小写以便比较。

    检索结果里的 ``source`` 是完整路径（可能是 Windows 反斜杠），
    而金标准里通常只写文件名，因此两边都要归一化后再比。
    """
    normalized = (path or "").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].strip().lower()


def _dedupe_ordered(items: Iterable[str]) -> list[str]:
    """按首次出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


@dataclass(slots=True)
class QueryOutcome:
    """单个查询的评估结果，保留中间量便于排查。"""

    question: str
    retrieved_sources: list[str]
    """top-k 召回映射成的来源文件（去重保序）。"""

    expected_sources: list[str]
    hit: bool
    """top-k 里是否命中至少一个期望来源。"""

    recall: float
    """命中数 / 期望数。期望为空时记为 1.0（视为无需召回，不计入扣分）。"""

    reciprocal_rank: float
    """第一个命中的倒数排名；未命中为 0.0。"""

    ndcg: float
    """二值相关性下的 nDCG。"""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "hit": self.hit,
            "recall": round(self.recall, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "ndcg": round(self.ndcg, 4),
            "retrieved_sources": self.retrieved_sources,
            "expected_sources": self.expected_sources,
        }


def evaluate_query(
    question: str,
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int,
) -> QueryOutcome:
    """对单个查询计算各项指标。

    Args:
        question: 查询原文（仅用于报告）。
        retrieved_sources: 按相关性降序的召回来源（可以是完整路径）。
        expected_sources: 金标准里期望命中的来源（通常是文件名）。
        k: 截断到前 k 条再评估。

    Returns:
        :class:`QueryOutcome`。
    """
    top_k = _dedupe_ordered(_basename(s) for s in retrieved_sources[:k])
    expected = _dedupe_ordered(_basename(s) for s in expected_sources)

    if not expected:
        # 没有标注期望来源的样本无法判断对错，按"不扣分"处理，
        # 而不是当成失败 —— 否则一条空标注会把整体指标拉低，产生误导。
        return QueryOutcome(
            question=question,
            retrieved_sources=top_k,
            expected_sources=[],
            hit=True,
            recall=1.0,
            reciprocal_rank=1.0,
            ndcg=1.0,
        )

    expected_set = set(expected)

    # ---- 命中与排名 ----
    first_hit_rank = 0  # 0 表示未命中；排名从 1 开始计
    for index, source in enumerate(top_k, start=1):
        if source in expected_set:
            first_hit_rank = index
            break

    hit = first_hit_rank > 0
    matched = len(expected_set & set(top_k))
    recall = matched / len(expected_set)
    reciprocal_rank = 1.0 / first_hit_rank if hit else 0.0

    # ---- nDCG（二值相关性）----
    # DCG = Σ rel_i / log2(i+1)；IDCG 用"理想排序"即前 min(len(expected), k) 个都命中。
    dcg = 0.0
    for index, source in enumerate(top_k, start=1):
        if source in expected_set:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return QueryOutcome(
        question=question,
        retrieved_sources=top_k,
        expected_sources=expected,
        hit=hit,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        ndcg=ndcg,
    )


@dataclass(slots=True)
class RetrievalReport:
    """一批查询的汇总结果。"""

    k: int
    total: int
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "recall": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
        }

    def summary_line(self) -> str:
        """一行摘要，便于在实验间快速对比。"""
        return (
            f"k={self.k} 样本={self.total} "
            f"命中率={self.hit_rate:.3f} 召回={self.recall:.3f} "
            f"MRR={self.mrr:.3f} nDCG={self.ndcg:.3f}"
        )

    def misses(self) -> list[QueryOutcome]:
        """未命中的查询，用于逐条排查检索为何失败。"""
        return [o for o in self.outcomes if not o.hit]


def summarize(outcomes: Sequence[QueryOutcome], k: int) -> RetrievalReport:
    """把逐条结果汇总成报告。

    Args:
        outcomes: :func:`evaluate_query` 的结果列表。
        k: 本次评估的截断值（仅用于报告展示）。

    Returns:
        :class:`RetrievalReport`。样本为空时各项指标为 0，避免除零。
    """
    n = len(outcomes)
    if n == 0:
        return RetrievalReport(k=k, total=0, hit_rate=0.0, recall=0.0, mrr=0.0, ndcg=0.0)

    return RetrievalReport(
        k=k,
        total=n,
        hit_rate=sum(1.0 for o in outcomes if o.hit) / n,
        recall=sum(o.recall for o in outcomes) / n,
        mrr=sum(o.reciprocal_rank for o in outcomes) / n,
        ndcg=sum(o.ndcg for o in outcomes) / n,
        outcomes=list(outcomes),
    )
