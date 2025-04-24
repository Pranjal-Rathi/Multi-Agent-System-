# test_summarizer.py

import json
from dataloader import load_papers
from summarizer import summarize_text
from summarizer import chunk_and_summarize


def main():
    # 1. Load data
    papers = load_papers()
    
    # 2. Summarize
    results = []
    # for idx, paper in enumerate(papers, 1):
    #     title = paper["Title"]
    #     text  = paper["Text"]
        
    #     summary = summarize_text(text)
    #     results.append({
    #         "Title":   title,
    #         "Summary": summary
    #     })
        
    #     # Optional progress print
    #     print(f"[{idx}/{len(papers)}] Summarized: {title}")
    for p in papers:
        summary = chunk_and_summarize(
            p["Text"],
            chunk_size=512,
            max_summary_length=150,
            num_beams=4
        )
        results.append({"Title": p["Title"],"date":p["date"] ,"Summary": summary})
        print("summarized:",p["Title"])

    # 3. Save results
    with open("summaries.json", "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} summaries → summaries.json")

if __name__ == "__main__":
    main()
