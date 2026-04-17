"""Unit tests for retrieval-only IR metrics."""

from __future__ import annotations

import math

import pytest

from fastapi_rag_lab.eval.retrieval_metrics import mrr, ndcg_at_k, recall_at_k


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a.md", "b.md", "c.md"]
        expected = ["a.md", "b.md"]
        assert recall_at_k(retrieved, expected, k=3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a.md", "x.md", "y.md"]
        expected = ["a.md", "b.md"]
        assert recall_at_k(retrieved, expected, k=3) == 0.5

    def test_zero_recall(self):
        retrieved = ["x.md", "y.md", "z.md"]
        expected = ["a.md", "b.md"]
        assert recall_at_k(retrieved, expected, k=3) == 0.0

    def test_k_limits_retrieved(self):
        retrieved = ["x.md", "y.md", "a.md", "b.md"]
        expected = ["a.md", "b.md"]
        # k=2 only looks at first two, which are misses
        assert recall_at_k(retrieved, expected, k=2) == 0.0

    def test_empty_expected_returns_one(self):
        assert recall_at_k(["a.md"], [], k=3) == 1.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["a.md"], k=3) == 0.0


class TestMRR:
    def test_first_hit_at_rank_one(self):
        assert mrr(["a.md", "b.md"], ["a.md"]) == 1.0

    def test_first_hit_at_rank_two(self):
        assert mrr(["x.md", "a.md", "b.md"], ["a.md"]) == 0.5

    def test_first_hit_at_rank_three(self):
        assert mrr(["x.md", "y.md", "a.md"], ["a.md"]) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert mrr(["x.md", "y.md"], ["a.md"]) == 0.0

    def test_multiple_expected_returns_first_found(self):
        # b.md at rank 2 is found before a.md at rank 3
        assert mrr(["x.md", "b.md", "a.md"], ["a.md", "b.md"]) == 0.5

    def test_empty_retrieved(self):
        assert mrr([], ["a.md"]) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        retrieved = ["a.md", "b.md", "x.md"]
        expected = ["a.md", "b.md"]
        assert ndcg_at_k(retrieved, expected, k=3) == 1.0

    def test_reversed_ranking(self):
        # Relevant docs at positions 2,3 instead of 1,2
        retrieved = ["x.md", "a.md", "b.md"]
        expected = ["a.md", "b.md"]
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(retrieved, expected, k=3) == pytest.approx(dcg / idcg)

    def test_no_relevant_docs(self):
        assert ndcg_at_k(["x.md", "y.md"], ["a.md"], k=2) == 0.0

    def test_empty_expected_returns_one(self):
        assert ndcg_at_k(["a.md"], [], k=3) == 1.0

    def test_k_limits_evaluation(self):
        # Relevant doc at position 3, but k=2
        retrieved = ["x.md", "y.md", "a.md"]
        expected = ["a.md"]
        assert ndcg_at_k(retrieved, expected, k=2) == 0.0

    def test_single_relevant_at_top(self):
        retrieved = ["a.md", "x.md", "y.md"]
        expected = ["a.md"]
        assert ndcg_at_k(retrieved, expected, k=3) == 1.0
