import streamlit as st
import pandas as pd
import os
from bertopic import BERTopic

st.set_page_config(page_title="Trends Agent", layout="wide")

# Sidebar
st.sidebar.title("🧠 Trends Agent")
mode = st.sidebar.radio("Choose Mode", ["Run BERTopic", "View Visualizations", "Ask Agent"])

output_dir = "output"

# 1️⃣ Run BERTopic
if mode == "Run BERTopic":
    st.title("📊 Run BERTopic on Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file with 'text' and 'timestamp' columns", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.dataframe(df.head())

        if st.button("Run BERTopic"):
            with st.spinner("Running topic modeling..."):
                topic_model = BERTopic(verbose=True)
                topics, _ = topic_model.fit_transform(df["text"])
                topic_model.visualize_topics().write_html(os.path.join(output_dir, "topic_viz.html"))
                topic_model.visualize_barchart().write_html(os.path.join(output_dir, "bar_chart.html"))
                topic_model.visualize_heatmap().write_html(os.path.join(output_dir, "heatmap.html"))
                topic_model.visualize_hierarchy().write_html(os.path.join(output_dir, "hierarchy.html"))
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    topic_model.visualize_topics_over_time(df["text"], topics, df["timestamp"]).write_html(
                        os.path.join(output_dir, "topics_over_time.html"))

            st.success("BERTopic completed and visualizations saved!")

# 2️⃣ Visualizations
elif mode == "View Visualizations":
    st.title("📈 Topic Visualizations")
    visual_files = {
        "Bar Chart": "bar_chart.html",
        "Topic Map": "topic_viz.html",
        "Heatmap": "heatmap.html",
        "Hierarchy": "hierarchy.html",
        "Topics Over Time": "topics_over_time.html"
    }

    selected_vis = st.selectbox("Choose a Visualization", list(visual_files.keys()))
    vis_path = os.path.join(output_dir, visual_files[selected_vis])
    if os.path.exists(vis_path):
        with open(vis_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.warning("Visualization not found. Please run BERTopic first.")

# 3️⃣ Natural Language Prompt
elif mode == "Ask Agent":
    st.title("💬 Ask Trends Agent")
    user_query = st.text_input("Ask a question like: What are the top 3 trends in AI this month?")

    if st.button("Ask") and user_query:
        # Mock output for now (you can replace with a real backend call)
        st.markdown("**Top 3 Topics:**")
        st.markdown("1. Instruction-tuned LLMs")
        st.markdown("2. Multimodal Agents")
        st.markdown("3. Evaluation of Tool-Use")
        st.markdown("📈 Topic 2 has seen a 250% increase since last month.")

        st.info("This is a mocked response. You can connect this to a FastAPI backend using LangChain or simple keyword routing.")

