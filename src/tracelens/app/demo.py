import streamlit as st
import time
import sys
import os
import json
import numpy as np

# Add src to path so we can import tracelens
sys.path.insert(0, os.path.abspath('src'))
from tracelens.retrieval.pipeline import DenseRetriever
from tracelens.indexing.incremental import IncrementalIndexer
from tracelens.categorize.prf import RocchioPRF

st.set_page_config(page_title="TraceLens Demo", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    :root {
        --accent: #6366f1;
        --bg: #ffffff;
        --text: #1e293b;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #0f172a;
            --text: #f8fafc;
        }
    }
    .stTextInput input {
        font-family: 'Inter', sans-serif;
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
    }
    .stTextInput input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }
    code {
        font-family: 'JetBrains Mono', monospace;
    }
    .result-container {
        background-color: rgba(99, 102, 241, 0.05);
        border-left: 4px solid var(--accent);
        padding: 16px;
        margin-bottom: 20px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Default Corpus
DEFAULT_CORPUS = {
    "doc1": {"text": "def bfs(graph, start):\n    queue = [start]\n    visited = set([start])\n    while queue:\n        node = queue.pop(0)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)"},
    "doc2": {"text": "def dfs(graph, node, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(node)\n    for neighbor in graph[node]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited"},
    "doc3": {"text": "import heapq\n\ndef dijkstra(graph, start):\n    distances = {node: float('infinity') for node in graph}\n    distances[start] = 0\n    queue = [(0, start)]\n    while queue:\n        current_distance, current_node = heapq.heappop(queue)\n        if current_distance > distances[current_node]:\n            continue\n        for neighbor, weight in graph[current_node].items():\n            distance = current_distance + weight\n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n                heapq.heappush(queue, (distance, neighbor))\n    return distances"},
    "doc4": {"text": "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]\n    for i in range(n + 1):\n        for w in range(capacity + 1):\n            if i == 0 or w == 0:\n                dp[i][w] = 0\n            elif weights[i-1] <= w:\n                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])\n            else:\n                dp[i][w] = dp[i-1][w]\n    return dp[n][capacity]"},
    "doc5": {"text": "def binary_search(arr, x):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (high + low) // 2\n        if arr[mid] < x:\n            low = mid + 1\n        elif arr[mid] > x:\n            high = mid - 1\n        else:\n            return mid\n    return -1"},
    "doc6": {"text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"}
}

if "corpus" not in st.session_state:
    st.session_state.corpus = DEFAULT_CORPUS
if "feedback" not in st.session_state:
    st.session_state.feedback = {"relevant": set(), "non_relevant": set()}

@st.cache_resource(show_spinner="Loading Embedding Model & Indexing...")
def load_engine(corpus):
    engine = DenseRetriever(model_name="BAAI/bge-small-en-v1.5", cross_encoder_name=None)
    engine.build_index(corpus)
    return engine

@st.cache_resource(show_spinner="Loading Cross-Encoder Reranker (May take a moment)...")
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

engine = load_engine(st.session_state.corpus)

st.title("🔍 TraceLens Code Search")
st.markdown("Agentic Code Intelligence for Competitive Programming")

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")

# Corpus Uploader
st.sidebar.markdown("### 📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload JSON Corpus (Format: {id: {text: code}})", type=["json"])
if uploaded_file is not None:
    try:
        new_corpus = json.load(uploaded_file)
        if st.sidebar.button("Rebuild Index"):
            st.session_state.corpus = new_corpus
            st.cache_resource.clear()
            st.rerun()
    except Exception as e:
        st.sidebar.error("Invalid JSON format.")

search_mode = st.sidebar.selectbox("Retrieval Mode", ["Hybrid (BM25 + Dense RRF)", "Dense Only", "Sparse Only"])
enable_reranker = st.sidebar.checkbox("Enable Cross-Encoder Reranking", value=False)
if enable_reranker:
    cross_encoder = load_cross_encoder()

rrf_k = st.sidebar.slider("RRF k constant", min_value=10, max_value=100, value=60)
enable_prf = st.sidebar.checkbox("Apply Rocchio Feedback (PRF)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Index Stats")
st.sidebar.write(f"- **Total Snippets**: {len(st.session_state.corpus)}")
st.sidebar.write(f"- **Embedding Dim**: 384")
st.sidebar.write(f"- **Vectors Engine**: FAISS (FlatIP)")

query = st.text_input("", placeholder="Describe the code you are looking for (e.g., shortest path in a graph, dynamic programming...)")

if query:
    start_time = time.time()
    
    # 1. Base Query Embedding
    query_emb_original = engine.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_emb = query_emb_original
    
    # 2. Rocchio PRF Application
    if enable_prf and (len(st.session_state.feedback["relevant"]) > 0 or len(st.session_state.feedback["non_relevant"]) > 0):
        prf = RocchioPRF(alpha=1.0, beta=0.75, gamma=0.15)
        # We need the embeddings of the feedback docs
        all_embeddings = engine.model.encode([st.session_state.corpus[doc_id]["text"] for doc_id in engine.corpus_ids], convert_to_numpy=True, normalize_embeddings=True)
        
        # Get indices of feedback docs
        rel_indices = [engine.corpus_ids.index(doc) for doc in st.session_state.feedback["relevant"] if doc in engine.corpus_ids]
        non_rel_indices = [engine.corpus_ids.index(doc) for doc in st.session_state.feedback["non_relevant"] if doc in engine.corpus_ids]
        
        # We fake the retrieved_indices input to Rocchio to force it to use our selected relevant/non-relevant
        # Normally PRF takes the top-k retrieved. Here we use explicit user feedback.
        if rel_indices:
            rel_mean = np.mean(all_embeddings[rel_indices], axis=0)
        else:
            rel_mean = np.zeros_like(query_emb[0])
            
        if non_rel_indices:
            non_rel_mean = np.mean(all_embeddings[non_rel_indices], axis=0)
        else:
            non_rel_mean = np.zeros_like(query_emb[0])
            
        modified_query = (1.0 * query_emb[0]) + (0.75 * rel_mean) - (0.15 * non_rel_mean)
        # Normalize the modified query
        modified_query = modified_query / np.linalg.norm(modified_query)
        query_emb = np.array([modified_query])
        st.info("✨ Rocchio Pseudo-Relevance Feedback applied to query vector based on your feedback!")

    # 3. Dense Search
    dense_scores, dense_indices = engine.index.search(query_emb, len(st.session_state.corpus))
    dense_res = {engine.corpus_ids[dense_indices[0][j]]: float(dense_scores[0][j]) for j in range(len(st.session_state.corpus)) if dense_indices[0][j] != -1}
    
    # 4. Sparse Search
    tokenized_query = query.lower().split()
    bm25_scores = engine.bm25.get_scores(tokenized_query)
    bm25_res = {engine.corpus_ids[idx]: float(bm25_scores[idx]) for idx in range(len(st.session_state.corpus))}
    
    # 5. Hybrid RRF
    fused_scores = {}
    dense_ranked = sorted(dense_res.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(dense_ranked):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        
    sparse_ranked = sorted(bm25_res.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(sparse_ranked):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        
    top_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select mode
    if "Dense" in search_mode and "Hybrid" not in search_mode:
        final_ranking = dense_ranked
    elif "Sparse" in search_mode:
        final_ranking = sparse_ranked
    else:
        final_ranking = top_fused
        
    # 6. Cross-Encoder Reranking
    if enable_reranker and len(final_ranking) > 0:
        cross_inputs = [[query, st.session_state.corpus[doc_id]["text"]] for doc_id, _ in final_ranking]
        cross_scores = cross_encoder.predict(cross_inputs)
        reranked = [(final_ranking[idx][0], float(cross_scores[idx])) for idx in range(len(final_ranking))]
        final_ranking = sorted(reranked, key=lambda x: x[1], reverse=True)

    latency = time.time() - start_time
    st.markdown(f"<div style='font-size: 0.9em; color: gray;'>Found {len(final_ranking)} results in {int(latency*1000)} ms | Mode: {search_mode}</div>", unsafe_allow_html=True)
    st.markdown("---")
        
    for i, (doc_id, score) in enumerate(final_ranking):
        d_score = dense_res.get(doc_id, 0)
        s_score = bm25_res.get(doc_id, 0)
        
        with st.container():
            st.markdown(f"<div class='result-container'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 0.5])
            with col1:
                st.markdown(f"### Rank {i+1} - `{doc_id}`")
                st.code(st.session_state.corpus[doc_id]["text"], language="python")
            with col2:
                st.markdown("#### Retrieval Metrics")
                if enable_reranker:
                    st.metric("Cross-Encoder Score", f"{score:.4f}")
                elif "Hybrid" in search_mode:
                    st.metric("RRF Score", f"{score:.4f}")
                elif "Dense" in search_mode:
                    st.metric("Dense Score", f"{score:.4f}")
                else:
                    st.metric("BM25 Score", f"{score:.4f}")
                    
                st.progress(min(max(d_score, 0), 1.0), text=f"Dense Cosine: {d_score:.2f}")
                st.progress(min(max(s_score / 20.0, 0), 1.0), text=f"BM25 Score: {s_score:.2f}")
            with col3:
                st.markdown("#### Feedback")
                is_rel = doc_id in st.session_state.feedback["relevant"]
                is_non_rel = doc_id in st.session_state.feedback["non_relevant"]
                
                if st.button("👍", key=f"up_{doc_id}", type="primary" if is_rel else "secondary"):
                    st.session_state.feedback["relevant"].add(doc_id)
                    st.session_state.feedback["non_relevant"].discard(doc_id)
                    st.rerun()
                if st.button("👎", key=f"down_{doc_id}", type="primary" if is_non_rel else "secondary"):
                    st.session_state.feedback["non_relevant"].add(doc_id)
                    st.session_state.feedback["relevant"].discard(doc_id)
                    st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
