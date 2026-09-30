import streamlit as st
import time

st.set_page_config(page_title="TraceLens Demo", layout="wide", initial_sidebar_state="expanded")

# CSS for a developer-tool aesthetic
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

query = st.text_input("Search code...", placeholder="e.g., shortest path in a grid")

if query:
    start_time = time.time()
    
    # Real pipeline simulation
    time.sleep(0.04) # simulated FAISS + BM25 time
    
    results = [
        {"id": "v1.2", "score": 0.92, "code": "def bfs(grid):\n    # implementation\n    pass", "category": "graph"},
        {"id": "v1.1", "score": 0.81, "code": "def dfs(grid):\n    # recursive implementation\n    pass", "category": "graph"}
    ]
    
    latency = time.time() - start_time
    st.caption(f"{len(results)} results in {int(latency*1000)} ms")
    
    for i, res in enumerate(results):
        st.markdown(f"**Rank {i+1}** (Score: {res['score']}) - {res['category']}")
        st.code(res['code'], language="python", line_numbers=True)
