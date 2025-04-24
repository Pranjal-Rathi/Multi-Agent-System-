# agents/summarizerAgent.py

import streamlit as st
from dataloader import load_papers
from summarizer import summarize_text, chunk_and_summarize

def run_summarizer_agent():
    st.title("📝 Summarizer Agent")

    # Sidebar controls
    st.sidebar.header("Summarizer Settings")
    chunk_size = st.sidebar.slider(
        "Chunk Size", 128, 1024, 512, 64,
        help="Max tokens per chunk when splitting long texts"
    )
    max_summary_length = st.sidebar.slider(
        "Max Summary Length", 50, 500, 150, 50,
        help="Max tokens in the generated summary"
    )
    num_beams = st.sidebar.slider(
        "Beam Width (num_beams)", 1, 10, 4, 1,
        help="Beam search width (quality vs speed)"
    )

    mode = st.sidebar.radio(
        "Mode",
        ["Summarize All", "Summarize One", "Custom Text"],
        index=0
    )

    @st.cache_data
    def get_papers():
        return load_papers()

    papers = get_papers()

    if mode == "Summarize All":
        st.header("📚 Summarize All Papers")
        st.markdown(f"Total papers loaded: **{len(papers)}**")
        if st.button("Run Summarization"):
            results = []
            progress = st.progress(0)
            for i, p in enumerate(papers):
                summary = chunk_and_summarize(
                    p["Text"],
                    chunk_size=chunk_size,
                    max_summary_length=max_summary_length,
                    num_beams=num_beams
                )
                results.append((p["Title"], p.get("date", "N/A"), summary))
                progress.progress((i + 1) / len(papers))
            st.success("✅ Summarization complete!")
            for title, date, summ in results:
                with st.expander(title):
                    st.write("**Published:**", date)
                    st.write(summ)

    elif mode == "Summarize One":
        st.header("🔍 Summarize a Single Paper")
        titles = [p["Title"] for p in papers]
        choice = st.selectbox("Select a paper", titles)
        if st.button("Summarize Selected"):
            idx = titles.index(choice)
            paper = papers[idx]
            summary = chunk_and_summarize(
                paper["Text"],
                chunk_size=chunk_size,
                max_summary_length=max_summary_length,
                num_beams=num_beams
            )
            st.subheader(paper["Title"])
            st.write("**Published:**", paper.get("date", "N/A"))
            st.write(summary)

    else:  # Custom Text
        st.header("✍️ Summarize Your Own Text")
        raw_text = st.text_area(
            "Enter text to summarize",
            height=300,
            placeholder="Paste or type your text here…"
        )
        if st.button("Summarize Text"):
            if not raw_text.strip():
                st.warning("Please enter some text above.")
            else:
                if len(raw_text) < chunk_size:
                    summary = summarize_text(
                        raw_text,
                        max_summary_length=max_summary_length,
                        num_beams=num_beams
                    )
                else:
                    summary = chunk_and_summarize(
                        raw_text,
                        chunk_size=chunk_size,
                        max_summary_length=max_summary_length,
                        num_beams=num_beams
                    )
                st.subheader("Generated Summary")
                st.write(summary)
