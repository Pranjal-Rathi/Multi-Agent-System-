import streamlit as st
import pandas as pd
import shutil
import os
from agents.webCrawlerAgent import web_crawler

# Function to clear temporary folders
def clear_temp_folders(download_folder="downloaded_papers", json_folder="json_outputs"):
    for folder in [download_folder, json_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)


# Sidebar agent selector
st.sidebar.title("🤖 Multi-Agent System")
selected_agent = st.sidebar.radio("Choose an Agent", ["Web Crawler Agent", "Trends Agent", "Summarizer Agent"])

# Agent 1: Web Crawler Agent
if selected_agent == "Web Crawler Agent":
    st.title("📄 Web Crawler Agent")
    query = st.text_input("Enter topic to search for papers")
    num_results = st.slider("Number of results", 1, 25, 10)

    papers = []

    if st.button("Fetch Papers"):
        if query.strip():
            clear_temp_folders()
            with st.spinner("Fetching papers from arXiv..."):
                papers = web_crawler.fetch_papers(query, num_results)

            st.success("Fetched papers successfully!")

            if papers:
                st.write("### Results")
                table = "<table><thead><tr><th>Title</th><th>Authors</th><th>Published</th><th>PDF Link</th></tr></thead><tbody>"
                for paper in papers:
                    table += f"<tr><td>{paper['Title']}</td><td>{paper['Authors']}</td><td>{paper['Published']}</td><td><a href='{paper['PDF_URL']}' target='_blank'>PDF</a></td></tr>"
                table += "</tbody></table>"
                st.markdown(table, unsafe_allow_html=True)

                st.download_button("Download CSV", pd.DataFrame(papers).to_csv(index=False), file_name="papers.csv")

        else:
            st.warning("Please enter a valid topic.")

# Agent 2: Trends Agent
elif selected_agent == "Trends Agent":
    from agents.TrendAgent import trends_agent_app
    trends_agent_app.run_trend_agent()  # Delegate UI rendering to agent
    
else:  # Summarizer Agent
    from agents.summarizer import run_summarizer_agent
    run_summarizer_agent()