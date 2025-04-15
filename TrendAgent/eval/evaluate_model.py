# File: eval/evaluate_model.py

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import pickle
import os
from src.preprocess import load_data, combine_fields, preprocess_documents


def calculate_coherence(topic_model, documents, top_n=10):
    topic_words = [
        [word for word, _ in topic_model.get_topic(t)[:top_n]]
        for t in topic_model.get_topics().keys()
        if topic_model.get_topic(t) is not False
    ]
    dictionary = Dictionary([doc.split() for doc in documents])
    corpus = [dictionary.doc2bow(doc.split()) for doc in documents]

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=[doc.split() for doc in documents],
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()


def calculate_diversity(topic_model, top_k=10):
    seen = set()
    total = 0
    for topic_id in topic_model.get_topics().keys():
        words = [word for word, _ in topic_model.get_topic(topic_id)[:top_k]]
        seen.update(words)
        total += len(words)
    return len(seen) / total if total > 0 else 0


def main():
    print("🔍 Evaluating Topic Model...")

    # Load preprocessed documents
    data_path = os.path.join("data", "summaries.json")
    model_path = "bertopic_model"  # folder saved with topic_model.save()

    data = load_data(data_path)
    docs = combine_fields(data)
    preprocessed_docs = preprocess_documents(docs)

    if not preprocessed_docs:
        print("No documents found.")
        return

    # Load the saved BERTopic model
    from bertopic import BERTopic
    topic_model = BERTopic.load(model_path)

    # Calculate metrics
    coherence = calculate_coherence(topic_model, preprocessed_docs)
    diversity = calculate_diversity(topic_model)

    print(f"📊 Coherence Score (c_v): {coherence:.4f}")
    print(f"🧠 Topic Diversity Score: {diversity:.4f}")


if __name__ == "__main__":
    main()
