"""评估执行器：把评估集跑过真实检索链路，产出指标。

分两层，刻意解耦：

  * :func:`run_retrieval_eval` —— 只调检索，**不调用 LLM**。秒级完成，
    因此可以放心地在每次调参后跑一遍；
  * :func:`collect_records` —— 跑完整 RAG（检索 + 生成），产出供 RAGAS 用的记录。
    慢，只在需要评估生成质量时跑。

这个划分是刻意的：绝大多数调参（分块、top_k、RRF k、是否重排）影响的是**检索**，
而检索指标的反馈速度快两个数量级。先优化检索，再谈生成。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from rag_app.eval.dataset import GoldenSample
from rag_app.eval.ragas_eval import EvalRecord
from rag_app.eval.retrieval_metrics import (
    RetrievalReport,
    evaluate_query,
    summarize,
)

logger = logging.getLogger(__name__)


async def run_retrieval_eval(
    retriever: Any,
    samples: Sequence[GoldenSample],
    k: int,
) -> RetrievalReport:
    """在评估集上跑检索并汇总指标（不调用 LLM）。

    Args:
        retriever: ``HybridRetriever``，需提供 ``async retrieve(query)``。
        samples: 金标准样本。``expected_sources`` 为空的样本会被跳过。
        k: 截断到前 k 条再评估；通常设为 ``FINAL_TOP_K``。

    Returns:
        :class:`RetrievalReport`。

    Raises:
        ValueError: 没有任何带 ``expected_sources`` 的样本。
    """
    usable = [s for s in samples if s.expected_sources]
    if not usable:
        raise ValueError(
            "评估集里没有任何样本标注了 expected_sources，无法计算检索指标。\n"
            "请在 JSONL 里为样本补上期望命中的文件名。"
        )

    outcomes = []
    for sample in usable:
        chunks = await retriever.retrieve(sample.question)
        # 检索结果已按相关性降序，直接映射成来源列表
        retrieved_sources = [c.source for c in chunks]

        outcome = evaluate_query(
            question=sample.question,
            retrieved_sources=retrieved_sources,
            expected_sources=sample.expected_sources,
            k=k,
        )
        outcomes.append(outcome)
        logger.debug(
            "样本完成：命中=%s 检索到=%s 期望=%s",
            outcome.hit,
            outcome.retrieved_sources,
            outcome.expected_sources,
        )

    report = summarize(outcomes, k)
    logger.info("检索评估完成：%s", report.summary_line())
    return report


async def collect_records(
    retriever: Any,
    chat: Any,
    samples: Sequence[GoldenSample],
    limit: int | None = None,
) -> list[EvalRecord]:
    """跑完整 RAG 链路，收集 RAGAS 需要的记录。

    Args:
        retriever: ``HybridRetriever``。
        chat: ``RAGChat``，需提供 ``async answer(question, chunks)``。
        samples: 金标准样本。
        limit: 只跑前 N 条。**强烈建议首次用 limit=1** 验证 RAGAS 的 API 与
            裁判模型是否可用，再跑全量（每条要调用数次 LLM，很慢）。

    Returns:
        :class:`EvalRecord` 列表。
    """
    selected = list(samples[:limit] if limit else samples)
    records: list[EvalRecord] = []

    for index, sample in enumerate(selected, start=1):
        logger.info("生成评估进度：%d/%d", index, len(selected))

        chunks = await retriever.retrieve(sample.question)
        if not chunks:
            # 检索为空时没有上下文可评，跳过而不是塞空列表 ——
            # RAGAS 对空 retrieved_contexts 通常直接报错
            logger.warning("样本检索为空，已跳过：%s", sample.question[:40])
            continue

        response = await chat.answer(sample.question, chunks)

        records.append(
            EvalRecord(
                question=sample.question,
                response=response,
                retrieved_contexts=[c.text for c in chunks],
                reference=sample.reference,
            )
        )

    logger.info("已收集 %d 条评估记录", len(records))
    return records
