import json
import mteb
import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from tracelens.retrieval.pipeline import PrePostPipelineEncoder

def main():
    model = PrePostPipelineEncoder("BAAI/bge-small-en-v1.5")
    task = mteb.get_task("AppsRetrieval")
    
    # Run evaluation
    evaluation = mteb.MTEB(tasks=[task])
    
    # We will use the provided encoder to encode
    results = evaluation.run(model, output_folder="results", encode_kwargs={"batch_size": 32})
    
    # Save the results properly
    if isinstance(results, list) and len(results) > 0:
        res_dict = results[0].to_dict()
    else:
        res_dict = results.to_dict() if hasattr(results, 'to_dict') else results
        
    with open("results/appsretrieval_results.json", "w") as f:
        json.dump(res_dict, f, indent=2)
    print("Evaluation complete.")

if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    main()
