"""评估子系统：检索指标（无依赖）与可选的 RAGAS 集成。

分两条线，按依赖与速度切分：

  * :mod:`rag_app.eval.retrieval_metrics` —— 命中率 / 召回 / MRR / nDCG。
    纯函数、零依赖、秒级完成，适合每次调参后回归。
  * :mod:`rag_app.eval.ragas_eval` —— LLM 评判的生成质量指标（需安装可选依赖
    ``ragas``，且需要裁判模型）。

:mod:`rag_app.eval.dataset` 负责金标准 JSONL 的读写，
:mod:`rag_app.eval.runner` 负责把评估集跑过真实链路。
"""

from rag_app.eval.dataset import (
    GoldenSample,
    GoldenSetError,
    load_golden_set,
    save_golden_set,
)
from rag_app.eval.retrieval_metrics import (
    QueryOutcome,
    RetrievalReport,
    evaluate_query,
    summarize,
)

__all__ = [
    "GoldenSample",
    "GoldenSetError",
    "QueryOutcome",
    "RetrievalReport",
    "evaluate_query",
    "load_golden_set",
    "save_golden_set",
    "summarize",
]
