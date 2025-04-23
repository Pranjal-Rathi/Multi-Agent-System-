def run_trend_agent():
    import streamlit as st
    import pandas as pd
    import os
    import sys
    import requests
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from wordcloud import WordCloud
    import importlib.util


    # Ensure the project root (Multi-Agent-System-/) is in sys.path
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)


    # ─── 1. Figure out where we live on disk ───────────────────────────────
    CURRENT_DIR   = os.path.dirname(__file__)                         # .../agents/TrendAgent
    PROJECT_ROOT  = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    SRC_DIR       = os.path.join(PROJECT_ROOT, "agents/TrendAgent/src")
    SRC_PARENT  = CURRENT_DIR                         # where “src/” lives
    output_dir  = os.path.join(CURRENT_DIR, "output")



    # ─── 2. Inject project root on to sys.path ────────────────────────────
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # ─── 3. Debug: show what Python can see ───────────────────────────────
    # st.sidebar.markdown("**🔍 Debug module path**")
    # st.sidebar.write("PROJECT_ROOT:", PROJECT_ROOT)
    # st.sidebar.write("sys.path[0:3]:", sys.path[:3])
    # st.sidebar.write("Contents of PROJECT_ROOT:", os.listdir(PROJECT_ROOT))
    # st.sidebar.write("Contents of Current_dir:", os.listdir(CURRENT_DIR))

    sys.path.insert(0, SRC_PARENT)

    # ─── 4. Try the normal import, else fallback ──────────────────────────
    try:
        from src.preprocess import load_data, combine_fields, preprocess_documents
    except ModuleNotFoundError as e:
        st.sidebar.error(f"Import failed: {e}")
        # fallback: load the file directly
        spec = importlib.util.spec_from_file_location(
            "preprocess",
            os.path.join(SRC_PARENT, "preprocess.py")
        )
        # st.sidebar.write("hiii")
        preprocess = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess)
        load_data, combine_fields, preprocess_documents = (
            preprocess.load_data,
            preprocess.combine_fields,
            preprocess.preprocess_documents
        )

    # ─── 5. Paths for model & data ────────────────────────────────────────
    MODEL_PATH  = os.path.join(CURRENT_DIR, "fine_tuned_model3")
    DATA_PATH   = os.path.join(CURRENT_DIR, "data", "summaries.json")
    os.makedirs("output", exist_ok=True)

    # ─── 6. Caching & loading ─────────────────────────────────────────────
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer(MODEL_PATH)
    EMBEDDING_MODEL = load_embedding_model()

    @st.cache_data
    def load_and_preprocess_data():
        data     = load_data(DATA_PATH)
        docs     = combine_fields(data)
        pre_docs = preprocess_documents(docs)
        return docs, pre_docs

    DOCUMENTS, PREPROCESSED = load_and_preprocess_data()

    @st.cache_resource
    def get_embeddings():
        return EMBEDDING_MODEL.encode(PREPROCESSED, show_progress_bar=False)
    EMBEDDINGS = get_embeddings()

    @st.cache_resource
    def load_or_init_topic_model():
        try:
            model = BERTopic.load(os.path.join(CURRENT_DIR, "fine_tuned_bertopic_model2"))
            topics, probs = model.transform(PREPROCESSED, EMBEDDINGS)
            model.topics_, model.probs_ = topics, probs
            return model
        except Exception:
            return None

    st.session_state.setdefault('topic_model', load_or_init_topic_model())

    # ─── 7. Rest of your Streamlit UI (unchanged) ─────────────────────────
    st.sidebar.title("🧠 Trends Agent")
    mode = st.sidebar.radio("Choose Mode", [
        "Run BERTopic", "View Visualizations", "Explore Documents", "Ask Agent"
    ])

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

    elif mode == "Explore Documents":
        # … same as before …
        ...
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

    else:  # Ask Agent
        # … same as before …
        ...
        st.title("💬 Ask Trends Agent")
        query = st.text_input("Your question...", key="ask_input")
        if st.button("Ask", key="ask_button") and query:
            try:
                resp = requests.get("http://localhost:5000/api/analyze", params={"q": query}).json()
                st.json(resp)
            except Exception as e:
                st.error(f"Backend error: {e}")


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
                
    # ─── 8. Reminder to add __init__.py ──────────────────────────────────
    st.sidebar.info(
        "Make sure there’s an empty __init__.py in your src/ folder "
        "so Python treats it as a package."
    )
