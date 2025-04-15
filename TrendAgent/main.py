import os
import pandas as pd
from src.preprocess import load_data, combine_fields, preprocess_documents
from src.topic_model import build_topic_model, print_topic_info, visualize_topics_interactive, get_topic_summary

def filter_by_domain(documents, domain_keyword=None):
    """
    Optionally filter documents by a keyword in the text.
    If domain_keyword is None, return all documents.
    """
    if not domain_keyword:
        return documents
    filtered = [doc for doc in documents if domain_keyword.lower() in doc.lower()]
    return filtered

def main():
    # Load and preprocess data
    data = load_data("data/summaries.json")
    documents = combine_fields(data)
    preprocessed_docs = preprocess_documents(documents)
    
    # Filter out empty documents
    filtered_docs = [doc for doc in preprocessed_docs if doc.strip()]
    print(f"Using {len(filtered_docs)} documents for topic modeling (filtered out empty ones).")
    
    if not filtered_docs:
        print("Error: No valid documents found after filtering. Exiting.")
        return
    
    # Optional: Apply domain filtering (for example, filter for "healthcare")
    # domain = "healthcare"  # change or set to None if no filtering is needed
    domain = None  # currently no filtering; update this if required.
    filtered_docs = filter_by_domain(filtered_docs, domain)
    print(f"After domain filtering, {len(filtered_docs)} documents remain.")

    # Build and train the topic model
    topic_model, topics, probs = build_topic_model(filtered_docs)
    
    # Print human-friendly topic info
    print_topic_info(topic_model)
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Visualizations: Save interactive visualizations as HTML
    viz = visualize_topics_interactive(topic_model)
    viz.write_html("output/topic_viz.html")
    topic_model.visualize_barchart(top_n_topics=10).write_html("output/bar_chart.html")
    topic_model.visualize_hierarchy().write_html("output/hierarchy.html")
    topic_model.visualize_heatmap().write_html("output/heatmap.html")
    
    # Save document-level topic info to CSV
    df = topic_model.get_document_info(filtered_docs)
    df.to_csv("output/topics_with_docs.csv", index=False)
    
    # Export JSON summary of topics
    topic_summary = get_topic_summary(topic_model)
    import json
    with open("output/topics_summary.json", "w", encoding="utf-8") as f:
        json.dump(topic_summary, f, indent=2)
    
    # Optionally, save the BERTopic model for later use
    topic_model.save("bertopic_model")
    print("Topic modeling complete. Visualizations and output files saved under 'output/'.")

if __name__ == "__main__":
    main()
