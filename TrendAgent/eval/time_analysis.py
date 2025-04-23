# """
# File: TrendAnalysisAgent/eval/time_analysis.py

# Performs temporal topic modeling using BERTopic's topics_over_time.
# Includes:
# - Date parsing from JSON entries.
# - Manual embedding generation with SentenceTransformer.
# - Interactive and static visualizations of topic evolution.
# - (Optional) A stub for enhanced topic labeling using KeyBERT.
# """

# import os
# import json
# from datetime import datetime
# import pandas as pd
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer
# from bertopic import BERTopic

# from src.preprocess import load_data, combine_fields, preprocess_documents


# def parse_date(date_str):
#     """Parse an ISO date string (YYYY-MM-DD) into a datetime object."""
#     try:
#         return datetime.strptime(date_str, "%Y-%m-%d")
#     except Exception as e:
#         print(f"Error parsing date '{date_str}': {e}")
#         return None


# def load_data_with_dates(filepath):
#     """
#     Loads data from JSON and returns:
#       - documents: combined text (title + summary)
#       - timestamps: list of datetime objects parsed from the date field.
#     Assumes each entry includes a "date" field.
#     """
#     data = load_data(filepath)
#     documents = []
#     timestamps = []
#     for entry in data:
#         title = entry.get("title", "").strip()
#         summary = entry.get("summary", "").strip()
#         date_str = entry.get("date", None)
#         if not date_str:
#             continue  # Skip entries without a valid date.
#         dt = parse_date(date_str)
#         if dt is None:
#             continue
#         combined = title + ". " + summary
#         documents.append(combined)
#         timestamps.append(dt)
#     return documents, timestamps


# def run_temporal_analysis(domain_filter=None):
#     print("📦 Loading and preprocessing data...")
#     data_path = os.path.join("data", "summaries.json")
#     documents, timestamps = load_data_with_dates(data_path)
#     preprocessed_docs = preprocess_documents(documents)
    
#     # Optional: Filter by a domain keyword if provided (timestamps remain as-is)
#     if domain_filter:
#         filtered = [(doc, ts) for doc, ts in zip(preprocessed_docs, timestamps) if domain_filter.lower() in doc.lower()]
#         if not filtered:
#             print("❌ No documents matched the domain filter.")
#             return
#         preprocessed_docs, timestamps = zip(*filtered)
    
#     if not preprocessed_docs:
#         print("❌ No valid documents after filtering.")
#         return

#     print("🤖 Generating embeddings...")
#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = embedding_model.encode(preprocessed_docs, show_progress_bar=True)

#     print("🧠 Building BERTopic model...")
#     # Build the model using the generated embeddings (do not pass timestamps here)
#     topic_model = BERTopic(verbose=True)
#     topics, probs = topic_model.fit_transform(preprocessed_docs, embeddings)

#     print("📈 Computing topics over time...")
#     # Call topics_over_time without the 'time_bin_size' parameter.
#     topics_over_time = topic_model.topics_over_time(preprocessed_docs, timestamps)
    
#     print("📊 Saving interactive visualization...")
#     fig = topic_model.visualize_topics_over_time(topics_over_time)
#     fig.write_html("output/topics_over_time.html")
#     print("Interactive topics over time visualization saved as 'output/topics_over_time.html'.")

#     # Optionally, plot a static line chart using pandas/matplotlib:
#     try:
#         df = pd.DataFrame(topics_over_time)
#         df["Time"] = pd.to_datetime(df["Time"])
#         df.set_index("Time", inplace=True)
#         # Pivot data so that each topic gets its own column for document counts.
#         df_grouped = df.pivot_table(index="Time", columns="Topic", values="Count", fill_value=0)
#         df_grouped.plot(figsize=(10, 6), title="Topic Prevalence Over Time")
#         plt.xlabel("Time")
#         plt.ylabel("Document Count")
#         plt.tight_layout()
#         plt.savefig("output/topics_over_time_line_plot.png")
#         plt.show()
#     except Exception as e:
#         print("⚠️ Error during line plotting:", e)

#     print("✅ Completed trend analysis. Check the output/ directory for results.")


# if __name__ == "__main__":
#     os.makedirs("output", exist_ok=True)
#     run_temporal_analysis(domain_filter=None)

import os
import sys

# Ensure src package importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
File: eval/time_analysis.py
Temporal trend analysis using BERTopic's topics_over_time
with your fine-tuned embeddings.
"""

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from src.preprocess import load_data, preprocess_documents


def load_embedding(model_name: str = "fine_tuned_model3") -> SentenceTransformer:
    """Load the fine-tuned SentenceTransformer model."""
    return SentenceTransformer(model_name)


EMBEDDING_MODEL = load_embedding()


def parse_date(date_str: str) -> datetime | None:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def load_data_with_dates(filepath: str) -> tuple[list[str], list[datetime]]:
    """Load documents and corresponding dates from JSON data."""
    raw_data = load_data(filepath)
    docs, times = [], []
    for entry in raw_data:
        dt = parse_date(entry.get("date", ""))
        if not dt:
            continue
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        text = f"{title}. {summary}" if title or summary else ""
        if text:
            docs.append(text)
            times.append(dt)
    return docs, times


def run_temporal_analysis(
    filepath: str,
    output_dir: str = "output",
    domain_filter: str | None = None
) -> None:
    """Run BERTopic temporal trend analysis and save outputs."""
    os.makedirs(output_dir, exist_ok=True)
    print("📦 Loading data with dates...")
    docs, times = load_data_with_dates(filepath)
    pre_docs = preprocess_documents(docs)

    if domain_filter:
        filtered = [(d, t) for d, t in zip(pre_docs, times) if domain_filter.lower() in d.lower()]
        if not filtered:
            print(f"No documents match filter: '{domain_filter}'")
            return
        pre_docs, times = zip(*filtered)

    print("🤖 Generating embeddings...")
    embeddings = EMBEDDING_MODEL.encode(pre_docs, show_progress_bar=True)

    print("🧠 Fitting BERTopic model...")
    topic_model = BERTopic(embedding_model=EMBEDDING_MODEL, verbose=True)
    topics, probs = topic_model.fit_transform(pre_docs, embeddings)

    print("📈 Calculating topics over time...")
    tot = topic_model.topics_over_time(pre_docs, times)

    # Interactive HTML
    print("💾 Saving interactive HTML...")
    fig = topic_model.visualize_topics_over_time(tot)
    fig.write_html(os.path.join(output_dir, "topics_over_time.html"))

    # Static line plot
    print("💾 Saving static line plot...")
    df = pd.DataFrame(tot)
    # Normalize time column name
    if "Time" in df.columns:
        df = df.rename(columns={"Time": "Timestamp"})
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")

    # Select numeric value column
    numeric_cols = df.select_dtypes(include='number').columns.difference(["Topic"])
    if len(numeric_cols) == 0:
        print("⚠️ No numeric value column found for plotting. Skipping static plot.")
    else:
        # Prefer 'Count' then first numeric
        value_col = "Count" if "Count" in numeric_cols else numeric_cols[0]
        print(f"Using '{value_col}' as value column for plotting.")
        df_pivot = df.pivot_table(
            index=df.index,
            columns="Topic",
            values=value_col,
            fill_value=0
        )
        ax = df_pivot.plot(
            figsize=(10, 6),
            title="Topic Prevalence Over Time"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Document Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "topics_over_time_line.png"))
        plt.close()

    print(f"✅ Trend analysis complete. Outputs in '{output_dir}' directory.")


if __name__ == "__main__":
    data_path = os.path.join("data", "summaries.json")
    run_temporal_analysis(data_path)