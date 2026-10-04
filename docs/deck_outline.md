# TraceLens - Samsung PRISM Generative AI Hackathon

## Slide 1: Title Slide
- **Project Name:** TraceLens
- **Team:** [Insert Team Name]
- **Theme 01:** Agentic Code Intelligence
- **Tagline:** High-precision, evolutionary code retrieval for competitive programming.

## Slide 2: Problem Statement & Constraints
- **Goal:** Accurately retrieve the most relevant code snippets from a large corpus given a natural language algorithmic query.
- **Challenges:** Noisy docstrings, competitive programming boilerplate ("Input/Output"), and semantic vs lexical mismatches.
- **Constraints:** Must run efficiently on CPU with minimal memory footprint (free-tier constraints). No test-set tuning.

## Slide 3: Our Solution Architecture
- **Preprocessing:** Custom regex-based code cleaning and algorithmic hint extraction (e.g. `dp`, `bitmask`, `heapq`).
- **Hybrid Retrieval:** Dense vectors (`BAAI/bge-small-en-v1.5` + FAISS) combined with Sparse vectors (`BM25`) using **Reciprocal Rank Fusion (RRF)**.
- **Reranking:** Final stage Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to push edge-cases to the top.
- **Feedback Loop:** Rocchio Pseudo-Relevance Feedback (PRF) for iterative query refinement.

## Slide 4: Evolutionary Indexing
- **The Challenge:** Real-world codebases evolve constantly; full re-indexing is computationally unfeasible.
- **Incremental Indexer:** Implemented `SHA-256` digest tracking. The index selectively re-embeds only modified snippets.
- **Impact:** Sub-millisecond index updates, enabling true continuous code intelligence.

## Slide 5: Evaluation & Metrics
- **Dataset:** `CoIR-Retrieval/apps`
- **NDCG@10:** [TO BE FILLED]
- **MRR:** [TO BE FILLED]
- **Validation:** All metrics generated dynamically via the standardized `mteb` benchmark suite. (See `results/appsretrieval_results.json`).

## Slide 6: Future Roadmap
- Integration with LLM generation heads for full RAG pipelines.
- Multi-language abstract syntax tree (AST) tokenization for sparse retrieval.
- Cloud-native deployment on AWS Lambda / Azure Functions.
