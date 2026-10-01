import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

class DenseRetriever:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus_ids = []
        self.bm25 = None
        self.corpus_texts = []

    def build_index(self, corpus):
        self.corpus_texts = [doc["text"] for doc in corpus.values()]
        self.corpus_ids = list(corpus.keys())
        
        # Dense Index
        embeddings = self.model.encode(self.corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        # Sparse Index
        tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_sparse(self, query, k=100):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        return {self.corpus_ids[idx]: bm25_scores[idx] for idx in top_indices}
