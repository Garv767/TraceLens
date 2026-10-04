import numpy as np

class RocchioPRF:
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.15, top_k_relevant=3, top_k_non_relevant=10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k_relevant = top_k_relevant
        self.top_k_non_relevant = top_k_non_relevant

    def apply(self, original_query_embedding, document_embeddings, retrieved_indices):
        """
        Applies Rocchio algorithm for Pseudo-Relevance Feedback.
        """
        if len(retrieved_indices) == 0:
            return original_query_embedding

        relevant_indices = retrieved_indices[:self.top_k_relevant]
        non_relevant_indices = retrieved_indices[-self.top_k_non_relevant:] if len(retrieved_indices) > self.top_k_relevant else []

        relevant_docs = document_embeddings[relevant_indices]
        non_relevant_docs = document_embeddings[non_relevant_indices] if len(non_relevant_indices) > 0 else []

        relevant_mean = np.mean(relevant_docs, axis=0) if len(relevant_docs) > 0 else np.zeros_like(original_query_embedding)
        non_relevant_mean = np.mean(non_relevant_docs, axis=0) if len(non_relevant_docs) > 0 else np.zeros_like(original_query_embedding)

        modified_query = (self.alpha * original_query_embedding) + (self.beta * relevant_mean) - (self.gamma * non_relevant_mean)
        return modified_query
