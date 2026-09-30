import time
from tracelens.indexing.incremental import IncrementalIndexer

def run_demo():
    print("Running Versioning Demo...")
    indexer = IncrementalIndexer()
    
    snippets = {
        "doc1": "def add(a, b): return a + b",
        "doc2": "def sub(a, b): return a - b",
        "doc3": "def mul(a, b): return a * b"
    }
    
    start = time.time()
    for id, code in snippets.items():
        indexer.add_or_update(id, code)
    print(f"Initial indexing time: {time.time() - start:.4f}s")
    
    # Update one snippet
    snippets["doc1"] = "def add(a, b, c=0): return a + b + c"
    
    start = time.time()
    for id, code in snippets.items():
        if indexer.add_or_update(id, code):
            print(f"Re-indexed updated snippet: {id}")
    print(f"Incremental indexing time: {time.time() - start:.4f}s")

if __name__ == '__main__':
    run_demo()
