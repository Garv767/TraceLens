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

    def search(self, queries, k=100, final_k=10):
        query_embeddings = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        dense_scores, dense_indices = self.index.search(query_embeddings, k)
        
        results = []
        for i, query in enumerate(queries):
            dense_q_res = {}
            for j in range(k):
                idx = dense_indices[i][j]
                if idx != -1:
                    dense_q_res[self.corpus_ids[idx]] = float(dense_scores[i][j])
                    
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
            bm25_q_res = {self.corpus_ids[idx]: bm25_scores[idx] for idx in top_bm25_indices}
            
            # Reciprocal Rank Fusion (RRF)
            # Calibrated RRF constant for code snippet lengths
            rrf_k = 60
            fused_scores = {}
            dense_ranked = sorted(dense_q_res.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(dense_ranked):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
                
            sparse_ranked = sorted(bm25_q_res.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sparse_ranked):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
                
            top_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:final_k]
            results.append({doc_id: score for doc_id, score in top_fused})
        return results
