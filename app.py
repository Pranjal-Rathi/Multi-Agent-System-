import streamlit as st
from agents.web_crawler import fetch_papers
import pandas as pd

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

st.title("Multi-Agent Research Assistant")

st.markdown("""
Welcome! This app uses multiple intelligent agents to:
- Crawl research papers
- Summarize them
- Analyze trends
- Give suggestions and interact with you
""")

query = st.text_input("🔍 Enter a research topic:")

if query and st.button("Fetch Papers"):
    st.info("Fetching papers from arXiv...")
    papers = fetch_papers(query)
    st.success(f"{len(papers)} papers fetched!")

    # Save to CSV
    df = pd.DataFrame(papers)
    df.to_csv("papers.csv", index=False)

    # Display
    for i, paper in enumerate(papers):
        with st.expander(f"📄 Paper {i+1}: {paper['Title']}"):
            st.write(f"**Authors:** {paper['Authors']}")
            st.write(f"**Published:** {paper['Published']}")
            st.write(f"[🔗 PDF Link]({paper['PDF_URL']})")
            if paper['PDF_Path']:
                st.write(f"📁 Downloaded to: `{paper['PDF_Path']}`")




# import streamlit as st

# st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

# st.title("Multi-Agent Research Assistant")

# st.markdown("""
# Welcome! This app uses multiple intelligent agents to:
# - Crawl research papers
# - Summarize them
# - Analyze trends
# - Give suggestions and interact with you
# """)

# query = st.text_input("🔍 Enter a research topic:")
# if query:
#     st.write(f"You entered: **{query}**")
#     st.info("Integrate your crawler/summarizer/trend agents here.")




# import streamlit as st
# from agents.web_crawler import fetch_papers
# from agents.summarizer import summarize_papers
# from agents.trend_analyst import analyze_trends
# from agents.chat_agent import chat_with_user

# st.title("🧠 Multi-Agent Research Assistant")

# # Step 1: Topic Input
# query = st.text_input("Enter a research topic:")
# if query:
#     if st.button("Fetch Papers"):
#         papers = fetch_papers(query)
#         st.session_state['papers'] = papers
#         st.success(f"{len(papers)} papers fetched.")

# # Step 2: Summarizer
# if 'papers' in st.session_state:
#     if st.button("Summarize Papers"):
#         summaries = summarize_papers(st.session_state['papers'])
#         for i, summary in enumerate(summaries):
#             with st.expander(f"Paper {i+1} Summary"):
#                 st.write(summary)

# # Step 3: Trend Analysis
#     if st.button("Analyze Trends"):
#         trends = analyze_trends(st.session_state['papers'])
#         st.pyplot(trends)  # Assuming it returns a matplotlib plot

# # Step 4: Chat Interface
#     st.subheader("Chat with Research Advisor Bot")
#     user_input = st.text_input("Ask something:")
#     if user_input:
#         response = chat_with_user(user_input, summaries)
#         st.write(f"🧠 Advisor: {response}")
