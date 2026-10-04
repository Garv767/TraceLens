import re
import unittest
from rank_bm25 import BM25Okapi

class TestBM25Retrieval(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "def binary_search(arr, target):",
            "def quicksort(arr):",
            "def breadth_first_search(graph, root):"
        ]
        self.tokenized_corpus = [re.findall(r'[a-zA-Z0-9]+', doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def test_term_frequency_scoring(self):
        query = re.findall(r'[a-zA-Z0-9]+', "binary search array".lower())
        scores = self.bm25.get_scores(query)
        self.assertEqual(len(scores), len(self.corpus))
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_tokenization_case_insensitivity(self):
        q1 = re.findall(r'[a-zA-Z0-9]+', "BINARY SEARCH".lower())
        q2 = re.findall(r'[a-zA-Z0-9]+', "binary search".lower())
        scores1 = self.bm25.get_scores(q1)
        scores2 = self.bm25.get_scores(q2)
        self.assertTrue((scores1 == scores2).all())

if __name__ == '__main__':
    unittest.main()
