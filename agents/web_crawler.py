# agents/web_crawler.py

import arxiv
import os
import requests

def fetch_papers(user_input, max_results=10):
    """
    Fetches research papers from arXiv based on a user-specified topic.
    Ensures papers are highly relevant by searching within title and abstract.
    Downloads the PDF and returns metadata for each result.
    """
    os.makedirs("downloaded_papers", exist_ok=True)
    results_list = []

    # 🔍 Smart query: limit search to title and abstract only (relevance-focused)
    refined_query = f'(ti:"{user_input}" OR abs:"{user_input}")'

    # arxiv.Search API setup
    search = arxiv.Search(
        query=refined_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    # Loop through results
    for result in search.results():
        title = result.title.strip()
        authors = ", ".join([a.name for a in result.authors])
        published = result.published.strftime("%Y-%m-%d")
        pdf_url = result.pdf_url

        # Sanitize filename for PDF
        safe_title = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in title)
        file_path = f"downloaded_papers/{safe_title[:100]}.pdf"

        # Download PDF
        try:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            else:
                file_path = None
        except Exception as e:
            print(f"❌ Error downloading PDF for {title}: {e}")
            file_path = None

        # Append paper metadata
        results_list.append({
            "Title": title,
            "Authors": authors,
            "Published": published,
            "PDF_URL": pdf_url,
            "PDF_Path": file_path
        })

    return results_list




# # agents/web_crawler.py
# import arxiv
# import os
# import requests

# def fetch_papers(user_input, max_results=5):
#     os.makedirs("downloaded_papers", exist_ok=True)
#     results_list = []

#     refined_query = f'(ti:"{user_input}" OR abs:"{user_input}")'

#     search = arxiv.Search(
#         query=refined_query,
#         max_results=max_results,
#         sort_by=arxiv.SortCriterion.SubmittedDate
#     )

#     for result in search.results():
#         title = result.title.strip()
#         authors = ", ".join([a.name for a in result.authors])
#         published = result.published
#         pdf_url = result.pdf_url

#         safe_title = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in title)
#         file_path = f"downloaded_papers/{safe_title[:100]}.pdf"

#         try:
#             response = requests.get(pdf_url, stream=True)
#             if response.status_code == 200:
#                 with open(file_path, 'wb') as f:
#                     for chunk in response.iter_content(chunk_size=1024):
#                         if chunk:
#                             f.write(chunk)
#             else:
#                 file_path = None
#         except Exception as e:
#             print(f"Error downloading PDF for {title}: {e}")
#             file_path = None

#         results_list.append({
#             "Title": title,
#             "Authors": authors,
#             "Published": published,
#             "PDF_URL": pdf_url,
#             "PDF_Path": file_path
#         })

#     return results_list
