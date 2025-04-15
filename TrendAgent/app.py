from flask import Flask, request, jsonify, send_from_directory
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd

# Import our base data functions
from src.preprocess import load_data, combine_fields, preprocess_documents
from src.topic_model import get_topic_summary  # Assuming this function is defined in src/topic_model.py

app = Flask(__name__)

#############################################
# Helper Functions for Temporal Analysis
#############################################

def parse_date(date_str):
    """Parse an ISO date string (YYYY-MM-DD) into a datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

def load_data_with_dates(filepath):
    """
    Loads data from JSON and returns:
      - documents: combined text (title + summary)
      - timestamps: list of datetime objects parsed from the date field.
    Assumes each entry includes a "date" field.
    """
    data = load_data(filepath)
    documents = []
    timestamps = []
    for entry in data:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        date_str = entry.get("date", None)
        if not date_str:
            continue
        dt = parse_date(date_str)
        if dt is None:
            continue
        combined = title + ". " + summary
        documents.append(combined)
        timestamps.append(dt)
    return documents, timestamps

def generate_embeddings(documents):
    """Generates document embeddings using SentenceTransformer."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(documents, show_progress_bar=False)
    return embeddings

#############################################
# Preload Baseline Data for /topics and /documents Endpoints
#############################################

# Load and preprocess data once at startup
DATA_PATH = "data/summaries.json"
data = load_data(DATA_PATH)
documents = combine_fields(data)
preprocessed_docs = preprocess_documents(documents)
# Build BERTopic model on baseline data
baseline_topic_model = BERTopic(verbose=True)
baseline_embeddings = generate_embeddings(preprocessed_docs)
baseline_topics, baseline_probs = baseline_topic_model.fit_transform(preprocessed_docs, baseline_embeddings)
topics_summary = get_topic_summary(baseline_topic_model)

#############################################
# Flask API Endpoints
#############################################

@app.route("/api/topics", methods=["GET"])
def get_topics():
    """
    GET endpoint to retrieve the friendly topic summary.
    Optionally, a 'domain' parameter can be provided to filter results.
    (In this baseline version, we return precomputed topics.)
    """
    domain = request.args.get("domain")
    # For a real implementation, you might re-run the pipeline with filtering.
    # Here we simply return the precomputed topics summary.
    return jsonify(topics_summary)

@app.route("/api/documents", methods=["GET"])
def get_documents():
    """
    GET endpoint to retrieve document-level topic info.
    """
    df_json = baseline_topic_model.get_document_info(preprocessed_docs).to_json(orient="records")
    return jsonify({"documents": df_json})

@app.route("/api/topics-over-time", methods=["GET"])
def topics_over_time_endpoint():
    """
    GET endpoint to compute topics over time.
    Accepts an optional 'domain' query parameter to filter documents.
    Returns JSON data with temporal trends.
    """
    try:
        # Load data with dates
        documents_with_dates, timestamps = load_data_with_dates(DATA_PATH)
        preprocessed = preprocess_documents(documents_with_dates)

        # Optional: domain filtering via query parameter
        domain = request.args.get("domain")
        if domain:
            filtered = [(doc, ts) for doc, ts in zip(preprocessed, timestamps) if domain.lower() in doc.lower()]
            if not filtered:
                return jsonify({"error": "No documents matched the domain filter."}), 404
            preprocessed, timestamps = zip(*filtered)

        if not preprocessed:
            return jsonify({"error": "No valid documents available."}), 404

        # Generate new embeddings for the (possibly filtered) documents
        embeddings = generate_embeddings(preprocessed)

        # Build a new BERTopic model on the fly
        temp_topic_model = BERTopic(verbose=True)
        topics, probs = temp_topic_model.fit_transform(preprocessed, embeddings)
        
        # Compute topics over time (returns a DataFrame)
        tot = temp_topic_model.topics_over_time(preprocessed, timestamps)
        
        # Convert the DataFrame to a list of dictionaries
        tot_dict = tot.to_dict(orient="records")
        
        # Return the topics-over-time data as JSON
        return jsonify(tot_dict)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#############################################
# New Endpoints for "Analyze" and "Visual"
#############################################

@app.route("/api/analyze", methods=["GET"])
def analyze():
    """
    /analyze endpoint: triggers a new topic modeling run on the baseline data.
    Returns a fresh topics summary as JSON.
    """
    try:
        data = load_data(DATA_PATH)
        documents = combine_fields(data)
        preprocessed_docs = preprocess_documents(documents)
        embeddings = generate_embeddings(preprocessed_docs)
        new_topic_model = BERTopic(verbose=True)
        new_topics, new_probs = new_topic_model.fit_transform(preprocessed_docs, embeddings)
        fresh_summary = get_topic_summary(new_topic_model)
        return jsonify(fresh_summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/visual", methods=["GET"])
def serve_visual():
    """
    Returns one of the visualization HTML files from the 'output' directory.
    Usage example:
      /api/visual?file=topics_over_time.html
    """
    try:
        # Get 'file' query parameter. Default could be 'topics_over_time.html'
        filename = request.args.get("file", "topics_over_time.html")
        
        # Build the path to the output directory
        file_path = os.path.join("output", filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"File '{filename}' not found in output directory."}), 404
        
        return send_from_directory("output", filename)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#############################################
# Main Application Run
#############################################

if __name__ == "__main__":
    # Ensure output directory exists (if using visualizations later)
    os.makedirs("output", exist_ok=True)
    app.run(debug=True, port=5000)
