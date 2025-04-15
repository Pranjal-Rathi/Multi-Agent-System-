# # File: TrendAnalysisAgent/src/preprocess.py

# import json
# import re
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# def load_data(filepath):
#     """Load JSON data from the given file path."""
#     with open(filepath, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     return data

# def combine_fields(data):
#     """Combine title and summary fields into one document per entry."""
#     documents = []
#     for entry in data:
#         title = entry.get("title", "")
#         summary = entry.get("summary", "")
#         combined = title.strip() + ". " + summary.strip()
#         documents.append(combined)
#     return documents

# def preprocess(text):
#     """Basic preprocessing: lowercasing, removing non-alphanumeric characters, and stopwords."""
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     words = text.split()
#     filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
#     return " ".join(filtered_words)

# def preprocess_documents(documents):
#     """Apply preprocessing to a list of documents."""
#     return [preprocess(doc) for doc in documents]

# if __name__ == "__main__":
#     # For quick testing; adjust the file path if needed.
#     data = load_data("../data/summaries.json")
#     documents = combine_fields(data)
#     preprocessed_docs = preprocess_documents(documents)
#     print("First preprocessed document:\n", preprocessed_docs[0])

# File: TrendAnalysisAgent/src/preprocess.py

import json
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def load_data(filepath):
    """Load JSON data from the given file path."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def combine_fields(data):
    """Combine title and summary fields into one document per entry."""
    documents = []
    for entry in data:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        combined = title.strip() + ". " + summary.strip()
        documents.append(combined)
    return documents

def preprocess(text):
    """Basic preprocessing: lowercasing, removing non-alphanumeric characters, and stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_words)

def preprocess_documents(documents):
    """Apply preprocessing to a list of documents."""
    return [preprocess(doc) for doc in documents]

if __name__ == "__main__":
    # For quick testing; adjust the file path if needed.
    data = load_data("../data/summaries.json")
    documents = combine_fields(data)
    preprocessed_docs = preprocess_documents(documents)
    print("First preprocessed document:\n", preprocessed_docs[0])
