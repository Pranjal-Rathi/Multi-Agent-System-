import os
import json
from dataloader import load_ssn_dataset
from summarizer import summarize_text

def process_papers(dataset: dict, split: str):
    """
    Process and summarize papers from the specified split.
    """
    papers = dataset.get(split, [])
    results = []
    
    for paper in papers:
        paper_id = paper.get("paper_id")
        title = paper.get("title")
        # We'll use the abstract for summarization; you can also combine with other fields if preferred.
        abstract = paper.get("abstract", "")
        
        if not abstract:
            print(f"Skipping paper {paper_id} due to missing abstract.")
            continue
        
        # Generate summary using T5.
        summary = summarize_text(abstract)
        
        # Example: if there is a date field, retrieve it; otherwise use a placeholder.
        date = paper.get("date", "N/A")
        
        result = {
            "paper_id": paper_id,
            "title": title,
            "summary": summary,
            "date": date
        }
        results.append(result)
    
    return results

def save_results(results, output_file: str):
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} summaries to {output_file}.")

if __name__ == '__main__':
    # Update the path based on your dataset directory
    dataset_dir = os.path.join("data", "SSN-inductive")
    dataset = load_ssn_dataset(dataset_dir)
    
    # Process a specific split (e.g., train) – adjust as needed for "val" or "test"
    results = process_papers(dataset, split="train")
    
    # Save the results
    save_results(results, "summarized_train_results.json")
