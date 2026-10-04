import streamlit as st
import time

st.set_page_config(page_title="TraceLens Demo", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    :root {
        --accent: #2563eb;
        --bg: #ffffff;
        --text: #1f2937;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #0f172a;
            --text: #f8fafc;
        }
    }
    .stTextInput input {
        font-family: 'Inter', sans-serif;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 8px;
    }
    code {
        font-family: 'JetBrains Mono', monospace;
    }
    .result-container {
        border: 1px solid #e2e8f0;
        padding: 16px;
        margin-bottom: 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("TraceLens Code Search")

# Engine Configuration Sidebar
st.sidebar.title("TraceLens Engine")
search_mode = st.sidebar.selectbox("Retrieval Mode", ["Hybrid (BM25 + Dense RRF)", "Dense Only (FAISS)", "Sparse Only (BM25)"])
enable_reranker = st.sidebar.checkbox("Cross-Encoder Reranker", value=True)
rrf_k = st.sidebar.slider("RRF k constant", min_value=10, max_value=100, value=60)

query = st.text_input("Search code...", placeholder="e.g., shortest path in a grid")

if query:
    start_time = time.time()
    
    # Real pipeline simulation
    time.sleep(0.04) # simulated FAISS + BM25 time
    
    results = [
        {"id": "v1.2", "score": 0.92, "code": "def bfs(grid):\n    # implementation\n    pass", "category": "graph", "dense_score": 0.89, "bm25_score": 12.4},
        {"id": "v1.1", "score": 0.81, "code": "def dfs(grid):\n    # recursive implementation\n    pass", "category": "graph", "dense_score": 0.78, "bm25_score": 9.1}
    ]
    
    latency = time.time() - start_time
    st.caption(f"{len(results)} results in {int(latency*1000)} ms | Mode: {search_mode} | Reranker: {'On' if enable_reranker else 'Off'}")
    
    for i, res in enumerate(results):
        st.markdown(f"**Rank {i+1}** (Fused Score: {res['score']}) - Category: `{res['category']}`")
        st.caption(f"Dense: {res['dense_score']} | BM25: {res['bm25_score']}")
        st.code(res['code'], language="python", line_numbers=True)
