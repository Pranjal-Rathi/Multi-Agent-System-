# dataloader.py

import os
import json
from typing import List, Dict

def load_papers(json_path: str = None) -> List[Dict[str, str]]:
    """
    Load the JSON list from data/text/extracted_papers.json,
    return a list of {"Title": ..., "Text": ...} dicts.
    """
    # summarizer/data/text/quantitative_biology_papers.json
    # agents\webCrawlerAgent\json_outputs\extracted_papers.json
    current_dir = os.path.dirname(__file__)
    if json_path is None:
        # json_path = os.path.join("agents","webCrawlerAgent","json_outputs", "extracted_papers.json")
        json_path = os.path.abspath(
        os.path.join(current_dir, "..", "webCrawlerAgent", "json_outputs", "extracted_papers.json")
    )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    papers = []
    for item in data:
        title = item.get("Title", "").strip()
        text  = item.get("Text", "").strip()
        date=item.get("Published","").strip()
        if title and text:
            papers.append({"Title": title, "Text": text,"date":date})
    return papers

if __name__ == "__main__":
    papers = load_papers()
    print(f"Loaded {len(papers)} papers")
    # print("First paper preview:", papers[0])
