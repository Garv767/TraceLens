# MTEB Benchmark Integration

TraceLens utilizes the Massive Text Embedding Benchmark (MTEB) for objective, reproducible evaluation on the `CoIR-Retrieval/apps` dataset.

## How it works
MTEB expects models to inherit from `mteb.models.abs_encoder.AbsEncoder` and expose an `encode()` method.

Because TraceLens is a multi-stage hybrid pipeline (Dense + Sparse + RRF + Cross-Encoder), we wrapped the entire pipeline inside `PrePostPipelineEncoder`. 

```python
from mteb.models.abs_encoder import AbsEncoder

class PrePostPipelineEncoder(AbsEncoder):
    def encode(self, sentences, **kwargs):
        # We pass texts through the sentence-transformer encoder here.
        # Note: True hybrid multi-stage evaluation requires custom MTEB task runners
        # or bypassing `MTEB.run()` to directly measure NDCG across the RRF pipeline.
        pass
```

## Running the Evaluation
To reproduce the baseline dense-only results:
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Run the evaluation script
python run_eval.py
```

This will download the `AppsRetrieval` split and generate the metrics in `results/appsretrieval_results.json`.

## Evaluation Scores
*Scores generated on [DATE], CPU environment.*

- **NDCG@10:** [TO BE FILLED]
- **MRR:** [TO BE FILLED]
