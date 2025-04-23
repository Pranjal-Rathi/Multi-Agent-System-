# import streamlit as st
# import pandas as pd
# import os
# import requests
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# from wordcloud import WordCloud

# # Page setup
# st.set_page_config(page_title="Trends Agent", layout="wide")

# # Load embedding model once at startup
# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("fine_tuned_model3")
# EMBEDDING_MODEL = load_embedding_model()

# # Load raw data once
# @st.cache_data
# def load_and_preprocess_data():
#     from src.preprocess import load_data, combine_fields, preprocess_documents
#     data = load_data("data/summaries.json")
#     docs = combine_fields(data)
#     pre_docs = preprocess_documents(docs)
#     return docs, pre_docs
# DOCUMENTS, PREPROCESSED = load_and_preprocess_data()

# # Load or initialize topic model
# @st.cache_resource
# def load_or_init_topic_model():
#     try:
#         return BERTopic.load("fine_tuned_bertopic_model2")
#     except:
#         return None
# st.session_state.setdefault('topic_model', load_or_init_topic_model())

# # Helper to generate embeddings
# def generate_embeddings(texts):
#     return EMBEDDING_MODEL.encode(texts, show_progress_bar=False)

# # Sidebar
# st.sidebar.title("🧠 Trends Agent")
# mode = st.sidebar.radio("Choose Mode", ["Run BERTopic", "View Visualizations", "Explore Documents", "Ask Agent"])

# # Ensure output directory
# output_dir = "output"
# os.makedirs(output_dir, exist_ok=True)

# # 1️⃣ Run BERTopic on preloaded data
# if mode == "Run BERTopic":
#     st.title("📊 Run BERTopic on Preloaded Research Summaries")
#     if st.button("Run BERTopic and Save Model", key="run_bertopic"):
#         with st.spinner("Training BERTopic..."):
#             texts = PREPROCESSED
#             embeddings = generate_embeddings(texts)
#             topic_model = BERTopic(embedding_model=EMBEDDING_MODEL, verbose=True)
#             topics, probs = topic_model.fit_transform(texts, embeddings)
#             topic_model.save("fine_tuned_bertopic_model2")
#             st.session_state['topic_model'] = topic_model
#         st.success("Model trained and saved.")

# # 2️⃣ View Visualizations
# elif mode == "View Visualizations":
#     st.title("📈 Topic Visualizations")
#     # List HTML and PNG files
#     html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
#     png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
#     all_files = html_files + png_files
#     if not all_files:
#         st.warning("No visualizations found. Run BERTopic first.")
#     else:
#         sel = st.selectbox("Select visualization", all_files, key="vis_select")
#         path = os.path.join(output_dir, sel)
#         if sel.endswith('.html'):
#             with open(path, 'r', encoding='utf-8', errors='ignore') as f:
#                 html = f.read()
#             st.components.v1.html(html, height=600)
#         elif sel.endswith('.png'):
#             st.image(path, caption=sel, use_column_width=True)

# # 3️⃣ Explore Documents
# elif mode == "Explore Documents":
#     st.title("🗂 Document Explorer")
#     topic_model = st.session_state.get('topic_model')
#     if topic_model is None:
#         st.info("No saved model found. Run BERTopic first.")
#     else:
#         docs = DOCUMENTS
#         info = topic_model.get_document_info(PREPROCESSED)
#         info['label'] = info['Topic'].apply(lambda t: "Unclassified" if t == -1 else f"Topic {t}")
#         sel = st.multiselect("Filter by topic label", info['label'].unique(), key="doc_sel")
#         filtered = info[info['label'].isin(sel)] if sel else info
#         st.dataframe(filtered)
#         st.download_button("Download assignments CSV", filtered.to_csv(index=False), file_name="doc_topics.csv", key="download_docs")

# # 4️⃣ Ask Agent
# elif mode == "Ask Agent":
#     st.title("💬 Ask Trends Agent")
#     query = st.text_input("Your question...", key="ask_input")
#     if st.button("Ask", key="ask_button") and query:
#         try:
#             resp = requests.get("http://localhost:5000/api/analyze", params={"q": query}).json()
#             st.json(resp)
#         except Exception as e:
#             st.error(f"Backend error: {e}")

# # Word clouds always visible at bottom of Run BERTopic
# if mode == "Run BERTopic" and st.session_state.get('topic_model'):
#     st.subheader("✨ Word Clouds for Top Topics")
#     tm = st.session_state['topic_model']
#     freq = tm.get_topic_freq()
#     top_ids = freq[freq.Topic != -1].Topic.head(4).tolist()
#     cols = st.columns(2)
#     for i, tid in enumerate(top_ids):
#         words = [w for w, _ in tm.get_topic(tid)[:20]]
#         wc = WordCloud(width=300, height=200).generate(" ".join(words))
#         with cols[i % 2]:
#             st.image(wc.to_array(), caption=f"Topic {tid}")
import streamlit as st
import pandas as pd
import os
import requests
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

# Page setup
st.set_page_config(page_title="Trends Agent", layout="wide")

# Load embedding model once at startup
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("fine_tuned_model3")
EMBEDDING_MODEL = load_embedding_model()

# Load raw data once
@st.cache_data
def load_and_preprocess_data():
    from src.preprocess import load_data, combine_fields, preprocess_documents
    data = load_data("data/summaries.json")
    docs = combine_fields(data)
    pre_docs = preprocess_documents(docs)
    return docs, pre_docs
DOCUMENTS, PREPROCESSED = load_and_preprocess_data()

# Pre-generate embeddings for reuse
@st.cache_resource
def get_embeddings():
    return EMBEDDING_MODEL.encode(PREPROCESSED, show_progress_bar=False)
EMBEDDINGS = get_embeddings()

# Load or initialize topic model and populate topics/probs
@st.cache_resource
def load_or_init_topic_model():
    try:
        model = BERTopic.load("fine_tuned_bertopic_model2")
        # Populate topics_ and probs_ using cached embeddings
        topics, probs = model.transform(PREPROCESSED, EMBEDDINGS)
        model.topics_ = topics
        model.probs_ = probs
        return model
    except Exception:
        return None

st.session_state.setdefault('topic_model', load_or_init_topic_model())

# Sidebar
st.sidebar.title("🧠 Trends Agent")
mode = st.sidebar.radio("Choose Mode", ["Run BERTopic", "View Visualizations", "Explore Documents", "Ask Agent"])

# Ensure output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 1️⃣ Run BERTopic on preloaded data
if mode == "Run BERTopic":
    st.title("📊 Run BERTopic on Preloaded Research Summaries")
    if st.button("Run BERTopic and Save Model", key="run_bertopic"):
        with st.spinner("Training BERTopic..."):
            texts = PREPROCESSED
            embeddings = EMBEDDINGS
            topic_model = BERTopic(embedding_model=EMBEDDING_MODEL, verbose=True)
            topics, probs = topic_model.fit_transform(texts, embeddings)
            topic_model.save("fine_tuned_bertopic_model2")
            st.session_state['topic_model'] = topic_model
        st.success("Model trained and saved.")

# 2️⃣ View Visualizations
elif mode == "View Visualizations":
    st.title("📈 Topic Visualizations")
    # List HTML and PNG files
    html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    all_files = html_files + png_files
    if not all_files:
        st.warning("No visualizations found. Run BERTopic first.")
    else:
        sel = st.selectbox("Select visualization", all_files, key="vis_select")
        path = os.path.join(output_dir, sel)
        if sel.endswith('.html'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
            st.components.v1.html(html, height=600)
        elif sel.endswith('.png'):
            st.image(path, caption=sel, use_column_width=True)

# 3️⃣ Explore Documents
elif mode == "Explore Documents":
    st.title("🗂 Document Explorer")
    topic_model = st.session_state.get('topic_model')
    if topic_model is None:
        st.info("No saved model found. Run BERTopic first.")
    else:
        # Retrieve document-topic assignments
        info = topic_model.get_document_info(PREPROCESSED)
        info['Document'] = DOCUMENTS
        info['label'] = info['Topic'].apply(lambda t: "Unclassified" if t == -1 else f"Topic {t}")
        sel = st.multiselect("Filter by topic label", info['label'].unique(), key="doc_sel")
        filtered = info[info['label'].isin(sel)] if sel else info
        st.dataframe(filtered)
        st.download_button("Download assignments CSV", filtered.to_csv(index=False), file_name="doc_topics.csv", key="download_docs")

# 4️⃣ Ask Agent
elif mode == "Ask Agent":
    st.title("💬 Ask Trends Agent")
    query = st.text_input("Your question...", key="ask_input")
    if st.button("Ask", key="ask_button") and query:
        try:
            resp = requests.get("http://localhost:5000/api/analyze", params={"q": query}).json()
            st.json(resp)
        except Exception as e:
            st.error(f"Backend error: {e}")

# Word clouds always visible at bottom of Run BERTopic
if mode == "Run BERTopic" and st.session_state.get('topic_model'):
    st.subheader("✨ Word Clouds for Top Topics")
    tm = st.session_state['topic_model']
    freq = tm.get_topic_freq()
    top_ids = freq[freq.Topic != -1].Topic.head(4).tolist()
    cols = st.columns(2)
    for i, tid in enumerate(top_ids):
        words = [w for w, _ in tm.get_topic(tid)[:20]]
        wc = WordCloud(width=300, height=200).generate(" ".join(words))
        with cols[i % 2]:
            st.image(wc.to_array(), caption=f"Topic {tid}")
