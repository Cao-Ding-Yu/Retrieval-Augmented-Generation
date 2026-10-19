"""RRF 融合的单元测试。

这是整个项目里最值得写测试的一块：融合算法决定了最终把哪些内容交给 LLM，
一旦它算错，表现是"答案质量莫名下降"，很难从日志里看出来。
把它写成纯函数并逐条断言，等于给检索质量上了保险。

实现位于 ``rag_app.retrieval.milvus_store``（与两条融合路径放在一起，
便于对照"自己融合"与"交给 Milvus 的 RRFRanker 融合"）。

覆盖点：
  1. 分数计算公式正确（与手工推导对比）
  2. "双分支命中"优于"单分支头名"（RRF 的核心价值）
  3. k 参数的作用方向正确
  4. 结果确定可复现（相同输入必得相同输出）
  5. 边界与异常输入
"""

from __future__ import annotations

import pytest

from rag_app.retrieval.milvus_store import (
    DEFAULT_RRF_K,
    fuse_two_branch,
    reciprocal_rank_fusion,
)


class TestScoreFormula:
    """验证打分公式本身：RRF(d) = Σ 1/(k + rank)。"""

    def test_single_branch_scores_match_manual_calculation(self):
        """单分支时分数应严格等于 1/(k + rank)，rank 从 1 起算。"""
        k = 60
        hits = reciprocal_rank_fusion({"dense": ["a", "b", "c"]}, k=k)

        expected = {
            "a": 1 / (k + 1),
            "b": 1 / (k + 2),
            "c": 1 / (k + 3),
        }

        assert {h.key: h.score for h in hits} == pytest.approx(expected)

    def test_two_branch_scores_are_summed(self):
        """双分支命中时两个贡献应相加。"""
        k = 60
        # a 在 dense 排第 1、在 sparse 排第 2
        hits = reciprocal_rank_fusion(
            {"dense": ["a", "b"], "sparse": ["b", "a"]},
            k=k,
        )
        by_key = {h.key: h for h in hits}

        assert by_key["a"].score == pytest.approx(1 / (k + 1) + 1 / (k + 2))
        assert by_key["b"].score == pytest.approx(1 / (k + 2) + 1 / (k + 1))

    def test_ranks_are_recorded_per_branch(self):
        """每个命中的分支名次应被完整记录，供上层解释排序依据。"""
        hits = reciprocal_rank_fusion(
            {"dense": ["a", "b"], "sparse": ["c", "a"]},
            k=60,
        )
        by_key = {h.key: h for h in hits}

        assert by_key["a"].ranks == {"dense": 0, "sparse": 1}
        assert by_key["b"].ranks == {"dense": 1}
        assert by_key["c"].ranks == {"sparse": 0}

    def test_rank_in_helper_returns_none_for_missing_branch(self):
        """未命中某分支时 rank_in 应返回 None，而不是抛 KeyError。"""
        hits = reciprocal_rank_fusion({"dense": ["a"]}, k=60)
        assert hits[0].rank_in("dense") == 0
        assert hits[0].rank_in("sparse") is None


class TestRrfCoreProperty:
    """验证 RRF 相对单分支排序的核心优势。"""

    def test_multi_branch_hit_outranks_single_branch_top(self):
        """在两个分支都排第二，应优于只在单一分支排第一。

        这是 RRF 存在的根本理由：**多路证据的一致性**比单路的极端高分更可信。
        数学上恒成立：2/(k+2) > 1/(k+1) 对所有 k>0 成立。
        """
        k = 60
        hits = reciprocal_rank_fusion(
            {
                "dense": ["only_dense", "both"],
                "sparse": ["only_sparse", "both"],
            },
            k=k,
        )

        assert hits[0].key == "both", "双分支命中的结果应排在首位"
        assert hits[0].score > hits[1].score

    def test_property_holds_across_k_values(self):
        """上述性质应对任意合法 k 都成立，而不是只在 k=60 时凑巧成立。"""
        for k in (1, 10, 60, 100, 1000):
            hits = reciprocal_rank_fusion(
                {"dense": ["solo", "both"], "sparse": ["solo2", "both"]},
                k=k,
            )
            assert hits[0].key == "both", f"k={k} 时性质被破坏"


class TestKParameter:
    """验证 k 的作用方向：越大越"平均化"。"""

    def test_larger_k_compresses_score_gap(self):
        """k 越大，头名与次名之间的分数差距越小。"""
        rankings = {"dense": ["first", "second"]}

        small_k = reciprocal_rank_fusion(rankings, k=1)
        large_k = reciprocal_rank_fusion(rankings, k=1000)

        gap_small = small_k[0].score - small_k[1].score
        gap_large = large_k[0].score - large_k[1].score

        assert gap_large < gap_small

    def test_k_must_be_positive(self):
        """k 非正数应直接报错，而不是产生除零或负分。"""
        with pytest.raises(ValueError, match="必须为正数"):
            reciprocal_rank_fusion({"dense": ["a"]}, k=0)

        with pytest.raises(ValueError, match="必须为正数"):
            reciprocal_rank_fusion({"dense": ["a"]}, k=-5)

    def test_default_k_is_sixty(self):
        """默认 k 应为业界常用的 60，避免被无意改动。"""
        assert DEFAULT_RRF_K == 60

        explicit = reciprocal_rank_fusion({"dense": ["a", "b"]}, k=60)
        default = reciprocal_rank_fusion({"dense": ["a", "b"]})
        assert [h.score for h in explicit] == [h.score for h in default]


class TestDeterminism:
    """验证结果可复现 —— 这是能被测试、能被调试的前提。"""

    def test_identical_input_gives_identical_output(self):
        rankings = {"dense": ["a", "b", "c"], "sparse": ["c", "b", "a"]}

        first = reciprocal_rank_fusion(rankings, k=60)
        second = reciprocal_rank_fusion(rankings, k=60)

        assert [(h.key, h.score) for h in first] == [(h.key, h.score) for h in second]

    def test_ties_are_broken_deterministically(self):
        """完全对称的输入（分数相同）必须给出稳定顺序，不能依赖字典遍历顺序。

        a 与 b 在两条分支上的名次正好互换，因此总分完全相同。
        此时按"最佳名次"仍相同，最终按 key 字典序 -> a 在前。
        """
        hits = reciprocal_rank_fusion({"dense": ["a", "b"], "sparse": ["b", "a"]}, k=60)

        assert hits[0].score == pytest.approx(hits[1].score)
        assert [h.key for h in hits] == ["a", "b"]

    def test_output_sorted_by_score_descending(self):
        """输出必须严格按分数降序，否则上层截断会取错候选。"""
        hits = reciprocal_rank_fusion(
            {"dense": ["a", "b", "c"], "sparse": ["b"]},
            k=60,
        )
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    """边界情况。"""

    def test_empty_rankings_returns_empty(self):
        assert reciprocal_rank_fusion({}, k=60) == []

    def test_empty_branch_is_ignored(self):
        """空分支不应影响另一分支的结果。"""
        hits = reciprocal_rank_fusion({"dense": ["a", "b"], "sparse": []}, k=60)
        assert [h.key for h in hits] == ["a", "b"]
        assert all("sparse" not in h.ranks for h in hits)

    def test_duplicate_key_in_same_branch_counted_once(self):
        """同一分支内重复出现的 key 只按最优名次计一次分。

        若不处理，重复项会被累加多次而虚假上浮，
        导致某条结果仅因为"在列表里出现两次"就排到前面。
        """
        hits = reciprocal_rank_fusion({"dense": ["a", "a", "b"]}, k=60)
        by_key = {h.key: h for h in hits}

        # a 只按 rank 0 计分，而不是 1/(k+1) + 1/(k+2)
        assert by_key["a"].score == pytest.approx(1 / 61)
        assert by_key["a"].ranks["dense"] == 0
        assert len(hits) == 2

    def test_single_document_in_single_branch(self):
        hits = reciprocal_rank_fusion({"dense": ["solo"]}, k=60)
        assert len(hits) == 1
        assert hits[0].score == pytest.approx(1 / 61)


class TestFuseTwoBranchWrapper:
    """验证双分支便捷封装与通用函数行为一致。"""

    def test_wrapper_matches_generic_function(self):
        dense = ["a", "b", "c"]
        sparse = ["c", "a"]

        wrapped = fuse_two_branch(dense, sparse, k=60)
        generic = reciprocal_rank_fusion({"dense": dense, "sparse": sparse}, k=60)

        assert [(h.key, h.score) for h in wrapped] == [(h.key, h.score) for h in generic]

    def test_wrapper_populates_both_branch_names(self):
        """封装后分支名必须是 dense / sparse，供上层检索器按名取用。"""
        hits = fuse_two_branch(["a"], ["a"], k=60)
        assert hits[0].ranks == {"dense": 0, "sparse": 0}
