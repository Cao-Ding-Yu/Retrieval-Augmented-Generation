"""评估模块的单元测试。

分两块：

  * **检索指标**（``retrieval_metrics``）—— 纯函数，可以精确断言到小数点。
    这部分必须测透，因为它是所有调参决策的依据：指标算错会让人朝着
    错误方向优化，而指标本身不会"报错"。
  * **金标准加载**（``dataset``）—— 重点测各类畸形输入是否给出可读的报错。
    标注文件是人写的，出错是常态。

RAGAS 部分不在单元测试范围内：它需要额外依赖与裁判模型，
且本模块对它的调用是惰性导入的防御式封装。
"""

from __future__ import annotations

import json
import math

import pytest

from rag_app.eval.dataset import (
    GoldenSetError,
    load_golden_set,
    parse_sample,
    save_golden_set,
)
from rag_app.eval.retrieval_metrics import (
    evaluate_query,
    summarize,
)

# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------


class TestPathNormalization:
    """检索结果的 source 是完整路径，标注里只有文件名，两边要先归一化。"""

    def test_windows_path_matches_basename(self):
        """Windows 反斜杠路径应能与纯文件名匹配上。"""
        outcome = evaluate_query(
            question="q",
            retrieved_sources=["C:\\projects\\docs\\manual.pdf"],
            expected_sources=["manual.pdf"],
            k=5,
        )
        assert outcome.hit is True

    def test_posix_path_matches_basename(self):
        outcome = evaluate_query(
            question="q",
            retrieved_sources=["/home/user/docs/manual.pdf"],
            expected_sources=["manual.pdf"],
            k=5,
        )
        assert outcome.hit is True

    def test_matching_is_case_insensitive(self):
        """Windows 文件名大小写不敏感，比较时统一小写。"""
        outcome = evaluate_query(
            question="q",
            retrieved_sources=["Manual.PDF"],
            expected_sources=["manual.pdf"],
            k=5,
        )
        assert outcome.hit is True


class TestHitAndRecall:
    def test_perfect_hit_scores_one(self):
        outcome = evaluate_query("q", ["a.md"], ["a.md"], k=5)

        assert outcome.hit is True
        assert outcome.recall == 1.0
        assert outcome.reciprocal_rank == 1.0
        assert outcome.ndcg == 1.0

    def test_complete_miss_scores_zero(self):
        outcome = evaluate_query("q", ["x.md", "y.md"], ["a.md"], k=5)

        assert outcome.hit is False
        assert outcome.recall == 0.0
        assert outcome.reciprocal_rank == 0.0
        assert outcome.ndcg == 0.0

    def test_partial_recall(self):
        """期望两个文件、只命中一个时，召回为 0.5。"""
        outcome = evaluate_query("q", ["a.md"], ["a.md", "b.md"], k=5)

        assert outcome.hit is True
        assert outcome.recall == 0.5

    def test_mrr_reflects_position(self):
        """命中但排在第二位时，MRR 应为 1/2。"""
        outcome = evaluate_query("q", ["x.md", "a.md"], ["a.md"], k=5)

        assert outcome.hit is True
        assert outcome.reciprocal_rank == pytest.approx(0.5)


class TestTruncationAndDedupe:
    def test_k_truncates_before_scoring(self):
        """排在 k 之外的命中不计分 —— 这正是 top-k 指标的意义。"""
        outcome = evaluate_query("q", ["x.md", "y.md", "a.md"], ["a.md"], k=2)

        assert outcome.hit is False
        assert outcome.retrieved_sources == ["x.md", "y.md"]

    def test_duplicate_retrieved_sources_are_deduped(self):
        """同一文件被切成多块、多条同时命中时，只应按一个来源计入。

        若不去重，"同一文档召回 5 块"会被当成命中 5 个来源，
        让 nDCG 虚高、也让 top-k 的名额被同一文档占满。
        """
        outcome = evaluate_query(
            "q",
            ["a.md", "a.md", "a.md", "b.md"],
            ["a.md", "b.md"],
            k=4,
        )

        assert outcome.retrieved_sources == ["a.md", "b.md"]
        assert outcome.recall == 1.0

    def test_dedupe_preserves_first_occurrence_order(self):
        """去重必须保序，否则名次信息被破坏、MRR 就算错了。"""
        outcome = evaluate_query("q", ["b.md", "a.md", "b.md"], ["a.md"], k=3)

        assert outcome.retrieved_sources == ["b.md", "a.md"]
        assert outcome.reciprocal_rank == pytest.approx(0.5)


class TestNdcg:
    def test_ideal_ordering_scores_one(self):
        """期望文件正好占据前几位时，nDCG 应为 1。"""
        outcome = evaluate_query("q", ["a.md", "b.md"], ["a.md", "b.md"], k=2)

        assert outcome.ndcg == pytest.approx(1.0)

    def test_late_hit_is_penalised(self):
        """命中位置越靠后，nDCG 越低；数值应与手算一致。"""
        outcome = evaluate_query("q", ["x.md", "a.md"], ["a.md"], k=2)

        # DCG = 1/log2(3)；IDCG = 1/log2(2) = 1
        assert outcome.ndcg == pytest.approx(1.0 / math.log2(3))

    def test_partial_hits_lower_than_perfect(self):
        perfect = evaluate_query("q", ["a.md", "b.md"], ["a.md", "b.md"], k=2)
        partial = evaluate_query("q", ["a.md", "x.md"], ["a.md", "b.md"], k=2)

        assert partial.ndcg < perfect.ndcg

    def test_ndcg_bounded_by_one(self):
        """多召回不应让 nDCG 超过 1 —— 超过说明 IDCG 算错了。"""
        outcome = evaluate_query("q", ["a.md", "b.md", "c.md"], ["a.md"], k=3)

        assert outcome.ndcg <= 1.0


class TestUnlabelledSamples:
    def test_missing_expectation_is_not_penalised(self):
        """未标注期望来源的样本按"不扣分"处理。

        若把它算作失败，一条空标注就会拉低整体指标，
        而真实原因只是"没标"，不是"检索错了" —— 会误导优化方向。
        """
        outcome = evaluate_query("q", ["whatever.md"], [], k=5)

        assert outcome.hit is True
        assert outcome.recall == 1.0
        assert outcome.ndcg == 1.0


class TestSummarize:
    def test_averages_across_queries(self):
        outcomes = [
            evaluate_query("q1", ["a.md"], ["a.md"], k=5),   # 命中，MRR=1
            evaluate_query("q2", ["x.md"], ["a.md"], k=5),   # 未命中，MRR=0
        ]

        report = summarize(outcomes, k=5)

        assert report.total == 2
        assert report.hit_rate == pytest.approx(0.5)
        assert report.mrr == pytest.approx(0.5)
        assert report.recall == pytest.approx(0.5)

    def test_empty_input_does_not_divide_by_zero(self):
        report = summarize([], k=5)

        assert report.total == 0
        assert report.hit_rate == 0.0
        assert report.ndcg == 0.0

    def test_misses_only_returns_failures(self):
        outcomes = [
            evaluate_query("q1", ["a.md"], ["a.md"], k=5),
            evaluate_query("q2", ["x.md"], ["a.md"], k=5),
        ]

        misses = summarize(outcomes, k=5).misses()

        assert [o.question for o in misses] == ["q2"]

    def test_summary_line_contains_all_metrics(self):
        report = summarize([evaluate_query("q", ["a.md"], ["a.md"], k=5)], k=5)
        line = report.summary_line()

        for label in ("命中率", "召回", "MRR", "nDCG"):
            assert label in line

    def test_to_dict_is_json_serialisable(self):
        report = summarize([evaluate_query("q", ["a.md"], ["a.md"], k=5)], k=5)
        json.dumps(report.to_dict())  # 不抛异常即可


# ---------------------------------------------------------------------------
# 金标准加载
# ---------------------------------------------------------------------------


def write_jsonl(tmp_path, lines: list[str]):
    """把若干行写入临时 JSONL 文件并返回路径。"""
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestParseSample:
    def test_minimal_valid_sample(self):
        sample = parse_sample({"question": "  什么是 RRF？  "}, line_number=1)

        assert sample.question == "什么是 RRF？"  # 首尾空白被清理
        assert sample.expected_sources == []
        assert sample.reference == ""

    def test_single_string_source_is_coerced_to_list(self):
        """常见手误：expected_sources 只写了一个字符串。应容错而非报错。"""
        sample = parse_sample(
            {"question": "q", "expected_sources": "a.md"},
            line_number=1,
        )
        assert sample.expected_sources == ["a.md"]

    def test_missing_question_raises_with_line_number(self):
        with pytest.raises(GoldenSetError, match="第 7 行"):
            parse_sample({"expected_sources": ["a.md"]}, line_number=7)

    def test_blank_question_raises(self):
        with pytest.raises(GoldenSetError, match="question"):
            parse_sample({"question": "   "}, line_number=1)

    def test_wrong_source_type_raises(self):
        with pytest.raises(GoldenSetError, match="expected_sources"):
            parse_sample({"question": "q", "expected_sources": [1, 2]}, line_number=1)

    def test_wrong_reference_type_raises(self):
        with pytest.raises(GoldenSetError, match="reference"):
            parse_sample({"question": "q", "reference": 123}, line_number=1)


class TestLoadGoldenSet:
    def test_loads_valid_file(self, tmp_path):
        path = write_jsonl(
            tmp_path,
            [
                '{"question": "q1", "expected_sources": ["a.md"]}',
                '{"question": "q2", "expected_sources": ["b.md"], "reference": "答案"}',
            ],
        )

        samples = load_golden_set(path)

        assert len(samples) == 2
        assert samples[1].reference == "答案"

    def test_blank_and_comment_lines_are_ignored(self, tmp_path):
        """空行与 # 注释应被跳过，方便在文件里写分组说明。"""
        path = write_jsonl(
            tmp_path,
            [
                "# 这一组是精确关键词型查询",
                "",
                '{"question": "q1", "expected_sources": ["a.md"]}',
                "   ",
            ],
        )

        assert len(load_golden_set(path)) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden_set(tmp_path / "nope.jsonl")

    def test_invalid_json_reports_line_number(self, tmp_path):
        path = write_jsonl(tmp_path, ['{"question": "q"}', "{ 这不是 json"])

        with pytest.raises(GoldenSetError, match="第 2 行"):
            load_golden_set(path)

    def test_empty_file_raises(self, tmp_path):
        path = write_jsonl(tmp_path, ["# 只有注释"])

        with pytest.raises(GoldenSetError, match="没有任何有效样本"):
            load_golden_set(path)

    def test_unknown_source_is_rejected(self, tmp_path):
        """标注了语料里不存在的文件时必须报错。

        这是本模块最重要的一个校验：拼错文件名会让该样本**永远无法命中**，
        但指标只是"看起来低了一点"，从数据上完全看不出根因。
        """
        path = write_jsonl(
            tmp_path,
            ['{"question": "q", "expected_sources": ["typo.md"]}'],
        )

        with pytest.raises(GoldenSetError, match="不存在"):
            load_golden_set(path, known_sources={"C:\\data\\real.md"})

    def test_known_source_check_accepts_paths(self, tmp_path):
        """已知来源是完整路径、标注是文件名时，也应校验通过。"""
        path = write_jsonl(
            tmp_path,
            ['{"question": "q", "expected_sources": ["real.md"]}'],
        )

        samples = load_golden_set(path, known_sources={"C:\\data\\real.md"})

        assert len(samples) == 1

    def test_known_source_check_matches_case_insensitively(self, tmp_path):
        path = write_jsonl(
            tmp_path,
            ['{"question": "q", "expected_sources": ["REAL.MD"]}'],
        )

        assert len(load_golden_set(path, known_sources={"real.md"})) == 1

    def test_validation_skipped_when_no_known_sources(self, tmp_path):
        """没提供已知来源时不做校验（例如集合是空的）。"""
        path = write_jsonl(
            tmp_path,
            ['{"question": "q", "expected_sources": ["whatever.md"]}'],
        )

        assert len(load_golden_set(path)) == 1


class TestRoundTrip:
    def test_save_then_load_preserves_content(self, tmp_path):
        path = write_jsonl(
            tmp_path,
            ['{"question": "q1", "expected_sources": ["a.md"], "reference": "中文答案"}'],
        )
        samples = load_golden_set(path)

        out = tmp_path / "out.jsonl"
        save_golden_set(samples, out)
        reloaded = load_golden_set(out)

        assert reloaded[0].question == "q1"
        assert reloaded[0].reference == "中文答案"

    def test_saved_file_keeps_chinese_readable(self, tmp_path):
        """JSONL 应以 UTF-8 明文写出中文，而不是 \\uXXXX 转义。

        转义后文件不可读，人工标注与 diff 都会变得痛苦。
        """
        path = write_jsonl(tmp_path, ['{"question": "中文问题", "expected_sources": []}'])
        samples = load_golden_set(path)

        out = tmp_path / "out.jsonl"
        save_golden_set(samples, out)

        assert "中文问题" in out.read_text(encoding="utf-8")
