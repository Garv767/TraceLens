import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

class DenseRetriever:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus_ids = []

    def build_index(self, corpus):
        # corpus is a dict {id: {"text": text}}
        texts = [doc["text"] for doc in corpus.values()]
        self.corpus_ids = list(corpus.keys())
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, queries, k=10):
        # queries is a list of strings
        query_embeddings = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_embeddings, k)
        
        results = []
        for i in range(len(queries)):
            q_res = {}
            for j in range(k):
                idx = indices[i][j]
                if idx != -1:
                    doc_id = self.corpus_ids[idx]
                    q_res[doc_id] = float(scores[i][j])
            results.append(q_res)
        return results

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.retriever = DenseRetriever(model_name)
        self.meta = ModelMeta(
            name="PrePostPipelineEncoder",
            revision="1.0.0",
            release_date=None,
            languages=["eng-Latn"],
            framework=["Sentence Transformers"],
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
