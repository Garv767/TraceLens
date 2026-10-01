import unittest
from rank_bm25 import BM25Okapi

class TestBM25Retrieval(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "def binary_search(arr, target):",
            "def quicksort(arr):",
            "def breadth_first_search(graph, root):"
        ]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def test_term_frequency_scoring(self):
        query = "binary search array".lower().split()
        scores = self.bm25.get_scores(query)
        self.assertEqual(len(scores), len(self.corpus))
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_tokenization_case_insensitivity(self):
        q1 = "BINARY SEARCH".lower().split()
        q2 = "binary search".lower().split()
        scores1 = self.bm25.get_scores(q1)
        scores2 = self.bm25.get_scores(q2)
        self.assertTrue((scores1 == scores2).all())

if __name__ == '__main__':
    unittest.main()
