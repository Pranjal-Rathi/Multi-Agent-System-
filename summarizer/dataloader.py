import os
import json
from typing import List, Dict

def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load a JSONL file and return a list of JSON objects.
    """
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                paper = json.loads(line.strip())
                papers.append(paper)
            except json.JSONDecodeError as e:
                print(f"Error decoding line in {file_path}: {e}")
    return papers

def load_ssn_dataset(dataset_dir: str) -> Dict[str, List[Dict]]:
    """
    Load the train, validation, and test JSONL files from the specified directory.
    
    The function expects the following files:
      - train.jsonl
      - val.jsonl
      - test.jsonl
      
    Returns a dictionary with keys "train", "val", and "test".
    """
    dataset = {}
    for split in ["train", "val", "test"]:
        file_name = f"{split}.jsonl"
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.exists(file_path):
            dataset[split] = load_jsonl(file_path)
            print(f"Loaded {len(dataset[split])} papers from {file_name}.")
        else:
            print(f"File {file_name} not found in {dataset_dir}.")
            dataset[split] = []
    return dataset

if __name__ == '__main__':
    # Update the path according to your dataset directory structure
    dataset_dir = os.path.join("data", "SSN-inductive")
    dataset = load_ssn_dataset(dataset_dir)
    # Print a summary of the first paper from the train split for verification
    if dataset["train"]:
        sample = dataset["train"][0]
        print("Sample paper from train split:")
        print("ID:", sample.get("paper_id"))
        print("Title:", sample.get("title"))
        print("Abstract:", sample.get("abstract")[:200], "...")
