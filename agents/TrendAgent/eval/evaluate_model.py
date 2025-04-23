import os, sys
# Ensure parent directory is on path so that `src` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

from src.preprocess import load_data, combine_fields, preprocess_documents


def calculate_coherence(topic_model, documents, top_n=10, measure='c_v'):
    tokenized = [doc.split() for doc in documents]
    topic_words = [
        [w for w,_ in topic_model.get_topic(t)[:top_n]]
        for t in topic_model.get_topics().keys() if topic_model.get_topic(t)
    ]
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    cm = CoherenceModel(
        topics=topic_words, texts=tokenized, dictionary=dictionary, coherence=measure
    )
    return cm.get_coherence()


def calculate_diversity(topic_model, top_k=10):
    seen, total = set(), 0
    for t in topic_model.get_topics().keys():
        words = [w for w,_ in topic_model.get_topic(t)[:top_k]]
        seen.update(words)
        total += len(words)
    return (len(seen)/total) if total else 0


def main():
    # Default paths
    model_path = "fine_tuned_bertopic_model2"
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'summaries.json'))

    print(f"🔍 Loading data from {data_path}")
    data = load_data(data_path)
    docs = combine_fields(data)
    pre_docs = preprocess_documents(docs)
    if not pre_docs:
        print("No documents found.")
        return

    print(f"📦 Loading BERTopic model from '{model_path}'... ")
    topic_model = BERTopic.load(model_path)

    print("🧪 Computing quality metrics...")
    cv = calculate_coherence(topic_model, pre_docs, measure='c_v')
    umass = calculate_coherence(topic_model, pre_docs, measure='u_mass')
    diversity = calculate_diversity(topic_model)

    print(f"📊 Coherence (c_v):    {cv:.4f}")
    print(f"📊 Coherence (u_mass): {umass:.4f}")
    print(f"🧠 Diversity:           {diversity:.4f}")

if __name__ == "__main__":
    main()
