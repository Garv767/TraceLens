import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import json
import os
from rank_bm25 import BM25Okapi

class DenseRetriever:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name) if cross_encoder_name else None
        self.index = None
        self.corpus_ids = []
        self.bm25 = None
        self.corpus_texts = []

    def build_index(self, corpus):
        # corpus is a dict {id: {"text": text}}
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
        # queries is a list of strings
        query_embeddings = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        dense_scores, dense_indices = self.index.search(query_embeddings, k)
        
        results = []
        for i, query in enumerate(queries):
            # Dense Results
            dense_q_res = {}
            for j in range(k):
                idx = dense_indices[i][j]
                if idx != -1:
                    doc_id = self.corpus_ids[idx]
                    dense_q_res[doc_id] = float(dense_scores[i][j])
                    
            # Sparse Results
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # Get top k BM25 indices
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
            bm25_q_res = {self.corpus_ids[idx]: bm25_scores[idx] for idx in top_bm25_indices}
            
            # Reciprocal Rank Fusion (RRF)
            rrf_k = 60
            fused_scores = {}
            
            # Rank Dense
            dense_ranked = sorted(dense_q_res.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(dense_ranked):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
                
            # Rank Sparse
            sparse_ranked = sorted(bm25_q_res.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sparse_ranked):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
                
            # Top K fused
            top_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Cross-Encoder Reranking
            if self.cross_encoder and len(top_fused) > 0:
                cross_inputs = [[query, self.corpus_texts[self.corpus_ids.index(doc_id)]] for doc_id, _ in top_fused]
                cross_scores = self.cross_encoder.predict(cross_inputs)
                
                reranked = [(top_fused[idx][0], float(cross_scores[idx])) for idx in range(len(top_fused))]
                reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:final_k]
                
                q_res = {doc_id: score for doc_id, score in reranked}
            else:
                q_res = {doc_id: score for doc_id, score in top_fused[:final_k]}
                
            results.append(q_res)
            
        return results

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.retriever = DenseRetriever(model_name)
        self.meta = ModelMeta(
            name="TraceLens/PrePostPipelineEncoder",
            revision="1.0.0",
            release_date="2026-10-04",
            languages=["eng-Latn"],
            framework=["Sentence Transformers"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
            loader=None,
            n_parameters=33000000,
            memory_usage_mb=120,
            max_tokens=512,
            embed_dim=384,
            license="mit",
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
        )

    def encode(self, sentences, task_name=None, prompt_type: PromptType = None, **kwargs):
        # MTEB evaluate passes texts here. We can just encode them.
        return self.retriever.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
