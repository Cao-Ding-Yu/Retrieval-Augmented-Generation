"""金标准问答集的加载与校验。

## 格式

一份 JSONL（每行一个 JSON 对象），字段：

====================  ======  ==========================================
字段                   必填    说明
====================  ======  ==========================================
``question``          是      测试问题
``expected_sources``  否      期望命中的来源文件名列表（用于检索指标）
``reference``         否      参考答案（RAGAS 的 context_recall 等指标需要）
====================  ======  ==========================================

示例::

    {"question": "RRF 是什么？", "expected_sources": ["rrf.md"], "reference": "倒数排名融合……"}

## 为什么用 JSONL 而不是 CSV

  * 一行一条，方便手工增删、也方便 diff（这正是标注工作流的实际形态）；
  * 天然支持列表字段（``expected_sources``），CSV 里得靠分隔符硬凑；
  * 追加写入不会破坏已有内容。

## 严肃提醒：``expected_sources`` 必须来自真实语料

标注时请打开 ``data/`` 里实际存在的文件，照抄其文件名。
凭印象写一个不存在的文件名，会让该样本**永远无法命中**，
从而把指标拉低 —— 而这种错误从指标本身看不出来，只会让人误以为检索有问题。
加载时会校验文件名是否出现在已知来源集合里（若调用方提供了该集合）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GoldenSample:
    """一条金标准样本。"""

    question: str
    expected_sources: list[str] = field(default_factory=list)
    """期望命中的来源文件名。为空表示该样本只参与生成质量评估，不参与检索指标。"""

    reference: str = ""
    """参考答案。RAGAS 的 context_recall / factual_correctness 会用到。"""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_sources": self.expected_sources,
            "reference": self.reference,
        }


class GoldenSetError(ValueError):
    """金标准文件格式错误。"""


def parse_sample(raw: dict, line_number: int) -> GoldenSample:
    """把一行 JSON 解析成 :class:`GoldenSample`。

    Args:
        raw: 已解析的 JSON 对象。
        line_number: 行号，仅用于报错定位。

    Returns:
        GoldenSample。

    Raises:
        GoldenSetError: 缺少 question，或字段类型不对。
    """
    if not isinstance(raw, dict):
        raise GoldenSetError(f"第 {line_number} 行不是 JSON 对象")

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise GoldenSetError(f"第 {line_number} 行缺少非空的 question 字段")

    sources = raw.get("expected_sources") or []
    if isinstance(sources, str):
        # 容错：只写了一个文件名的常见手误
        sources = [sources]
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise GoldenSetError(
            f"第 {line_number} 行的 expected_sources 必须是字符串或字符串列表"
        )

    reference = raw.get("reference") or ""
    if not isinstance(reference, str):
        raise GoldenSetError(f"第 {line_number} 行的 reference 必须是字符串")

    return GoldenSample(
        question=question.strip(),
        expected_sources=[s.strip() for s in sources if s.strip()],
        reference=reference.strip(),
    )


def load_golden_set(
    path: str | Path,
    known_sources: Iterable[str] | None = None,
) -> list[GoldenSample]:
    """从 JSONL 文件加载金标准集。

    Args:
        path: JSONL 文件路径。
        known_sources: 已知来源文件名集合。提供时会校验 ``expected_sources``
            是否真的存在 —— 拼错的文件名会让样本永远无法命中，
            而这种错误仅凭指标看不出来，所以宁可在加载阶段就报出来。

    Returns:
        GoldenSample 列表。

    Raises:
        FileNotFoundError: 文件不存在。
        GoldenSetError: 某行格式错误，或引用了不存在的来源文件。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"金标准文件不存在：{file_path}")

    samples: list[GoldenSample] = []
    # 归一化已知来源：只取 basename 并小写，与检索指标的比较口径保持一致
    known = (
        {s.replace("\\", "/").rsplit("/", 1)[-1].strip().lower() for s in known_sources}
        if known_sources is not None
        else None
    )

    with file_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            # 允许空行与 # 注释行，方便在文件里写分组说明
            if not line or line.startswith("#"):
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenSetError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc

            sample = parse_sample(raw, line_number)

            if known is not None:
                missing = [
                    s
                    for s in sample.expected_sources
                    if s.replace("\\", "/").rsplit("/", 1)[-1].strip().lower() not in known
                ]
                if missing:
                    raise GoldenSetError(
                        f"第 {line_number} 行引用了语料中不存在的来源：{missing}。\n"
                        "请核对 data/ 目录下的实际文件名（只写文件名即可，不必写完整路径）。"
                    )

            samples.append(sample)

    if not samples:
        raise GoldenSetError(f"金标准文件里没有任何有效样本：{file_path}")

    logger.info("已加载金标准集：%d 条（%s）", len(samples), file_path)
    return samples


def save_golden_set(samples: Iterable[GoldenSample], path: str | Path) -> None:
    """把样本写回 JSONL（便于手工标注后保存）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    logger.info("已写出 %s", target)
