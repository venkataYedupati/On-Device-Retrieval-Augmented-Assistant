from __future__ import annotations

import unittest

from on_device_assistant.benchmarking import ndcg_at_k


class BenchmarkingTests(unittest.TestCase):
    def test_ndcg_at_k_scores_perfect_ranking_as_one(self) -> None:
        self.assertEqual(ndcg_at_k([True, True, False], 3), 1.0)

    def test_ndcg_at_k_penalizes_late_relevant_results(self) -> None:
        self.assertLess(ndcg_at_k([False, True, True], 3), 1.0)


if __name__ == "__main__":
    unittest.main()
