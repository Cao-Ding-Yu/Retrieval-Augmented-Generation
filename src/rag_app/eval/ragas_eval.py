"""RAGAS 集成（可选依赖，惰性导入）。

## 为什么单独放一个模块、而且默认不导入

RAGAS 是**可选**依赖：不装它，本项目全部核心功能照常工作。
因此这里所有 ragas 的 import 都推迟到真正调用时才发生，
避免"没装 ragas 就连服务都起不来"。

## 重要：关于本模块的验证状态

RAGAS 的 API 在 0.2 → 0.3 → 0.4 之间改动过多次（类名、构造方式、指标命名都有变化）。
本模块**未经本机实测**（开发环境未安装 ragas），因此刻意做了两件事把风险包住：

  1. **所有 ragas 调用都收敛在本文件**里少数几个函数中，
     版本不匹配时只需改这一处，不影响其它任何代码；
  2. 指标类通过 :func:`_resolve_metric` 按多个候选名依次尝试，
     并对缺失的指标给出明确的告警而不是静默跳过。

首次使用前请：
  1. ``pip install -e ".[eval]"``
  2. 先跑一条样本（``--limit 1``）确认 API 匹配，
     再跑全量 —— 别一上来就全量，RAGAS 每个样本要调用好几次 LLM。

## 本地模型当裁判的现实问题

RAGAS 的 LLM 评判指标要求裁判模型稳定输出**结构化 JSON**。
7B~14B 的本地模型经常做不到，表现为大量样本解析失败（返回 NaN）。
实践中的处理顺序：
  1. 先用本模块的检索指标（``retrieval_metrics``）—— 它不需要模型；
  2. 再只开 ``faithfulness`` 这类最简单的指标试水；
  3. 确认稳定后再加 ``context_precision`` / ``context_recall``。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class RagasUnavailableError(RuntimeError):
    """ragas 未安装或版本不兼容。"""


@dataclass(slots=True)
class EvalRecord:
    """一条用于 RAGAS 评估的完整记录。

    字段名与 RAGAS 的 ``SingleTurnSample`` 对齐：

      * ``question``          -> ``user_input``
      * ``response``          -> ``response``
      * ``retrieved_contexts``-> ``retrieved_contexts``
      * ``reference``         -> ``reference``
    """

    question: str
    response: str
    retrieved_contexts: list[str] = field(default_factory=list)
    reference: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "response": self.response,
            "retrieved_contexts": self.retrieved_contexts,
            "reference": self.reference,
            "context_count": len(self.retrieved_contexts),
        }


# 指标名 -> 多个候选类名。
# 按顺序尝试，命中第一个可用的即可 —— 这样跨 0.2/0.3/0.4 都能跑起来。
METRIC_CANDIDATES: dict[str, tuple[str, ...]] = {
    # 答案是否完全由检索到的上下文支撑（衡量幻觉）
    "faithfulness": ("Faithfulness",),
    # 答案与问题的相关程度
    "answer_relevancy": ("ResponseRelevancy", "AnswerRelevancy"),
    # 检索到的上下文里，相关内容排序得有多好
    "context_precision": (
        "LLMContextPrecisionWithReference",
        "ContextPrecision",
    ),
    # 参考答案里的信息有多少被检索到的上下文覆盖
    "context_recall": ("LLMContextRecall", "ContextRecall"),
}

# 默认只开 faithfulness：它对裁判模型的要求最低，本地 7B~14B 最可能跑通。
DEFAULT_METRICS: tuple[str, ...] = ("faithfulness",)


def _import_ragas() -> dict[str, Any]:
    """惰性导入 ragas 的关键组件。

    Returns:
        含 ``evaluate`` / ``EvaluationDataset`` / ``SingleTurnSample`` /
        ``llm_wrapper`` / ``embeddings_wrapper`` / ``metrics_module`` 的字典。

    Raises:
        RagasUnavailableError: 未安装 ragas。
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except ImportError as exc:  # pragma: no cover - 取决于是否装了 extras
        raise RagasUnavailableError(
            "未安装 ragas。请先执行：pip install -e \".[eval]\"\n"
            "（或 pip install ragas）"
        ) from exc

    # 模型包装类的路径在版本间变动过，两处都试一下
    llm_wrapper = None
    embeddings_wrapper = None
    try:
        from ragas.llms import LangchainLLMWrapper as llm_wrapper  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        pass
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper as embeddings_wrapper  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        pass

    import ragas.metrics as metrics_module

    return {
        "evaluate": evaluate,
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "llm_wrapper": llm_wrapper,
        "embeddings_wrapper": embeddings_wrapper,
        "metrics_module": metrics_module,
    }


def _resolve_metric(metrics_module: Any, name: str) -> Any | None:
    """按候选类名解析指标对象；都找不到时返回 None 并告警。"""
    for candidate in METRIC_CANDIDATES.get(name, (name,)):
        metric_cls = getattr(metrics_module, candidate, None)
        if metric_cls is not None:
            try:
                return metric_cls()
            except TypeError:  # pragma: no cover - 少数指标构造需要参数
                logger.warning("指标 %s 构造需要参数，已跳过", candidate)
                return None

    logger.warning(
        "当前 ragas 版本找不到指标 %s（已尝试 %s），已跳过",
        name,
        METRIC_CANDIDATES.get(name, (name,)),
    )
    return None


def build_dataset(records: Sequence[EvalRecord]) -> Any:
    """把评估记录转成 RAGAS 数据集。

    Args:
        records: 评估记录。

    Returns:
        ragas 的 EvaluationDataset。

    Raises:
        RagasUnavailableError: 未安装 ragas。
        ValueError: records 为空。
    """
    if not records:
        raise ValueError("评估记录为空")

    ragas = _import_ragas()
    sample_cls = ragas["SingleTurnSample"]
    dataset_cls = ragas["EvaluationDataset"]

    samples = []
    for record in records:
        # 只传非空字段：参考为空时传空串会让 context_recall 类指标报错，
        # 直接不传该键更安全（RAGAS 会视作缺失，跳过相关指标）
        payload: dict[str, Any] = {
            "user_input": record.question,
            "response": record.response,
            "retrieved_contexts": record.retrieved_contexts,
        }
        if record.reference:
            payload["reference"] = record.reference
        samples.append(sample_cls(**payload))

    return dataset_cls(samples=samples)


def evaluate_with_ragas(
    records: Sequence[EvalRecord],
    llm: Any,
    embeddings: Any | None = None,
    metric_names: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """用 RAGAS 评估一批记录。

    Args:
        records: 评估记录（问题 / 回答 / 检索到的上下文 / 参考答案）。
        llm: 裁判模型。传 LangChain 的 ChatOllama 即可，本函数会自行包装。
            直接用本地模型时建议 ``temperature=0``。
        embeddings: 嵌入模型，供 answer_relevancy 等指标使用。
        metric_names: 要计算的指标名，取值见 :data:`METRIC_CANDIDATES`。

    Returns:
        形如 ``{"metrics": {...}, "per_sample": [...], "skipped": [...]}``
        的字典。``metrics`` 是各指标的平均分。

    Raises:
        RagasUnavailableError: 未安装 ragas。
    """
    ragas = _import_ragas()

    # ---- 包装裁判模型 ----
    # RAGAS 需要 ragas 自己的 LLM/Embeddings 接口，LangChain 对象要包一层
    wrapped_llm = llm
    llm_wrapper = ragas["llm_wrapper"]
    if llm_wrapper is not None:
        try:
            wrapped_llm = llm_wrapper(llm)
        except Exception:  # pragma: no cover - 包装失败时退回原对象
            logger.warning("包装裁判模型失败，将直接传入原对象", exc_info=True)

    wrapped_embeddings = embeddings
    embeddings_wrapper = ragas["embeddings_wrapper"]
    if embeddings is not None and embeddings_wrapper is not None:
        try:
            wrapped_embeddings = embeddings_wrapper(embeddings)
        except Exception:  # pragma: no cover
            logger.warning("包装嵌入模型失败，将直接传入原对象", exc_info=True)

    # ---- 解析指标 ----
    metrics = []
    skipped: list[str] = []
    for name in metric_names:
        metric = _resolve_metric(ragas["metrics_module"], name)
        if metric is None:
            skipped.append(name)
        else:
            metrics.append(metric)

    if not metrics:
        raise RagasUnavailableError(
            f"没有任何可用的 RAGAS 指标（请求了 {list(metric_names)}）。"
            "请检查 ragas 版本，或改用内置的检索指标（rag-app-cli eval）。"
        )

    dataset = build_dataset(records)
    logger.info("开始 RAGAS 评估：%d 条样本，指标=%s", len(records), [m for m in metric_names])

    # ---- 执行评估 ----
    result = ragas["evaluate"](
        dataset=dataset,
        metrics=metrics,
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )

    # result 在不同版本里可能是 EvaluationResult（有 to_pandas）或 dict
    per_sample: list[dict] = []
    averages: dict[str, float] = {}

    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        per_sample = frame.to_dict(orient="records")
        for column in frame.columns:
            if column in {"user_input", "response", "retrieved_contexts", "reference"}:
                continue
            series = frame[column]
            try:
                averages[column] = float(series.mean())
            except (TypeError, ValueError):  # pragma: no cover - 非数值列
                continue
    else:  # pragma: no cover - 老版本返回 dict
        averages = {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}

    return {"metrics": averages, "per_sample": per_sample, "skipped": skipped}
