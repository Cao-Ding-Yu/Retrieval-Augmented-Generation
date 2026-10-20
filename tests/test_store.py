"""RRF 融合的单元测试。

这是全项目最值得测的一块：融合算法决定了最终把哪些内容交给 LLM，
算错了表现是"答案质量莫名下降"，很难从日志里看出来。
好在它是纯函数，可以精确断言到小数点。
"""

from __future__ import annotations

import pytest

from rag_app.store import _to_hits, fuse
from tests.conftest import make_hit


class TestScoreFormula:
    """RRF(d) = Σ 1/(k + rank)，rank 从 1 起算。"""

    def test_single_branch_scores_match_manual_calculation(self):
        hits = fuse([make_hit("a"), make_hit("b")], [], k=60)
        scores = {h.id: h.score for h in hits}

        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)

    def test_two_branch_scores_are_summed(self):
        # a 在稠密排第 1、在 BM25 排第 2；b 反之，两者同分
        hits = fuse([make_hit("a"), make_hit("b")], [make_hit("b"), make_hit("a")], k=60)
        scores = {h.id: h.score for h in hits}

        assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)

    def test_ranks_are_recorded_per_branch(self):
        hits = fuse([make_hit("a")], [make_hit("x"), make_hit("a")], k=60)
        by_id = {h.id: h for h in hits}

        assert by_id["a"].dense_rank == 0
        assert by_id["a"].sparse_rank == 1
        assert by_id["x"].dense_rank is None
        assert by_id["x"].sparse_rank == 0


class TestCoreProperty:
    """RRF 相对单分支排序的核心优势。"""

    def test_multi_branch_hit_outranks_single_branch_top(self):
        """两路都排第 2，应优于只在单路排第 1。

        数学上是恒成立的：2/(k+2) > 1/(k+1) 对任意 k>0。
        这就是"多路证据一致性比单路极端高分更可信"的形式化表达。
        """
        hits = fuse(
            [make_hit("only_dense"), make_hit("both")],
            [make_hit("only_sparse"), make_hit("both")],
            k=60,
        )

        assert hits[0].id == "both"
        assert hits[0].is_hybrid_hit is True

    @pytest.mark.parametrize("k", [1, 10, 60, 1000])
    def test_property_holds_for_any_k(self, k: int):
        hits = fuse(
            [make_hit("solo"), make_hit("both")],
            [make_hit("solo2"), make_hit("both")],
            k=k,
        )

        assert hits[0].id == "both"

    def test_larger_k_compresses_gap(self):
        small = fuse([make_hit("first"), make_hit("second")], [], k=1)
        large = fuse([make_hit("first"), make_hit("second")], [], k=1000)

        gap_small = small[0].score - small[1].score
        gap_large = large[0].score - large[1].score
        assert gap_large < gap_small


class TestDeterminism:
    def test_ties_are_broken_deterministically(self):
        """完全对称的输入必须给出稳定顺序，不能依赖字典遍历顺序。"""
        first = fuse([make_hit("a"), make_hit("b")], [make_hit("b"), make_hit("a")], k=60)
        second = fuse([make_hit("b"), make_hit("a")], [make_hit("a"), make_hit("b")], k=60)

        assert [h.id for h in first] == [h.id for h in second]

    def test_output_sorted_by_score_desc(self):
        hits = fuse([make_hit("a"), make_hit("b")], [make_hit("b")], k=60)

        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    def test_both_branches_empty(self):
        assert fuse([], [], k=60) == []

    def test_empty_branch_is_ignored(self):
        hits = fuse([make_hit("a")], [], k=60)

        assert len(hits) == 1
        assert hits[0].sparse_rank is None

    def test_duplicate_in_same_branch_counted_once(self):
        """同一分支内重复出现只按最优名次计一次分。

        若不处理，"同一文档召回多次"会被累加多次而虚假上浮。
        """
        hits = fuse([make_hit("a"), make_hit("a"), make_hit("b")], [], k=60)
        by_id = {h.id: h for h in hits}

        assert by_id["a"].score == pytest.approx(1 / 61)
        assert by_id["a"].dense_rank == 0
        assert len(hits) == 2

    def test_non_positive_k_raises(self):
        with pytest.raises(ValueError, match="必须为正数"):
            fuse([make_hit("a")], [], k=0)


class TestPagePropagation:
    """PDF 页码要从 Milvus 行一路传到前端 payload。

    页码和 chunk_index 是两个概念：一页可能切出多块，分块序号还跨页连续。
    所以引用"第 10 页"不能拿 chunk_index+1 冒充。
    """

    def test_hit_page_defaults_to_none(self):
        assert make_hit("a").page is None

    def test_to_dict_exposes_page(self):
        hit = make_hit("a")
        hit.page = 7

        assert hit.to_dict()["page"] == 7

    def test_to_hits_reads_page(self):
        rows = [[{"id": 1, "distance": 0.5, "entity": {
            "text": "t", "doc_id": "d", "source": "s.pdf",
            "chunk_index": 2, "page": 7,
        }}]]

        hit = _to_hits(rows)[0]

        assert hit.page == 7
        assert hit.chunk_index == 2

    def test_zero_page_means_not_a_pdf(self):
        """写入时非 PDF 来源写 0，读出来必须还原成 None。"""
        rows = [[{"id": 1, "distance": 0.5, "entity": {
            "text": "t", "doc_id": "d", "source": "a.txt",
            "chunk_index": 0, "page": 0,
        }}]]

        assert _to_hits(rows)[0].page is None

    def test_missing_page_field_tolerated(self):
        """老集合补字段前写入的行没有该字段，不能因此报错。"""
        rows = [[{"id": 1, "distance": 0.5, "entity": {
            "text": "t", "doc_id": "d", "source": "a.txt", "chunk_index": 0,
        }}]]

        assert _to_hits(rows)[0].page is None

    def test_null_page_tolerated(self):
        """补字段后存量行读出 None（服务端要求新增字段 nullable）。"""
        rows = [[{"id": 1, "distance": 0.5, "entity": {
            "text": "t", "doc_id": "d", "source": "a.txt",
            "chunk_index": 0, "page": None,
        }}]]

        assert _to_hits(rows)[0].page is None
