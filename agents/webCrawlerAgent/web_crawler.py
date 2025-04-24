# import arxiv
# import os
# import requests
# import json
# import fitz  # PyMuPDF

# def fetch_papers(user_input, max_results=10):
#     """
#     Fetches research papers from arXiv based on a user-specified topic.
#     Downloads the PDF, extracts text content, and stores metadata + content in a JSON file.
#     """
#     os.makedirs("downloaded_papers", exist_ok=True)
#     os.makedirs("json_outputs", exist_ok=True)
#     results_list = []

#     # 🔍 Smart query: limit search to title and abstract only (relevance-focused)
#     refined_query = f'(ti:"{user_input}" OR abs:"{user_input}")'

#     # arxiv.Search API setup
#     search = arxiv.Search(
#         query=refined_query,
#         max_results=max_results,
#         sort_by=arxiv.SortCriterion.Relevance,
#         sort_order=arxiv.SortOrder.Descending
#     )

#     # Loop through results
#     for result in search.results():
#         title = result.title.strip()
#         authors = ", ".join([a.name for a in result.authors])
#         published = result.published.strftime("%Y-%m-%d")
#         pdf_url = result.pdf_url

#         # Sanitize filename for PDF and JSON
#         safe_title = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in title)
#         file_path = f"downloaded_papers/{safe_title[:100]}.pdf"
#         json_path = f"json_outputs/{safe_title[:100]}.json"

#         content_text = ""

#         # Download PDF
#         try:
#             response = requests.get(pdf_url, stream=True)
#             if response.status_code == 200:
#                 with open(file_path, 'wb') as f:
#                     for chunk in response.iter_content(chunk_size=1024):
#                         if chunk:
#                             f.write(chunk)

#                 # 📄 Extract text using PyMuPDF
#                 with fitz.open(file_path) as doc:
#                     for page in doc:
#                         content_text += page.get_text()

#             else:
#                 file_path = None
#         except Exception as e:
#             print(f"❌ Error downloading or processing PDF for {title}: {e}")
#             file_path = None

#         # Create metadata dict
#         paper_data = {
#             "Title": title,
#             "Authors": authors,
#             "Published": published,
#             "PDF_URL": pdf_url,
#             "PDF_Path": file_path,
#             "Text": content_text.strip()
#         }

#         # Save JSON
#         with open(json_path, 'w', encoding='utf-8') as jf:
#             json.dump(paper_data, jf, ensure_ascii=False, indent=2)

#         # Append to result list (without full content)
#         results_list.append({
#             "Title": title,
#             "Authors": authors,
#             "Published": published,
#             "PDF_URL": pdf_url,
#             "PDF_Path": file_path,
#             "JSON_Path": json_path
#         })

#     return results_list

import arxiv
import os
import requests
import json
import fitz  # PyMuPDF

def fetch_papers(user_input, max_results=10):
    """
    Fetches research papers from arXiv based on a user-specified topic.
    Downloads the PDF, extracts text content, and stores all metadata + content in a single JSON file.
    """
    os.makedirs("downloaded_papers", exist_ok=True)
    os.makedirs("json_outputs", exist_ok=True)
    results_list = []

    refined_query = f'(ti:"{user_input}" OR abs:"{user_input}")'

    search = arxiv.Search(
        query=refined_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    for result in search.results():
        title = result.title.strip()
        authors = ", ".join([a.name for a in result.authors])
        published = result.published.strftime("%Y-%m-%d")
        pdf_url = result.pdf_url

        safe_title = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in title)
        file_path = f"downloaded_papers/{safe_title[:100]}.pdf"

        content_text = ""

        try:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                with fitz.open(file_path) as doc:
                    for page in doc:
                        content_text += page.get_text()
            else:
                file_path = None
        except Exception as e:
            print(f"❌ Error downloading or processing PDF for {title}: {e}")
            file_path = None

        paper_data = {
            "Title": title,
            "Authors": authors,
            "Published": published,
            "PDF_URL": pdf_url,
            "PDF_Path": file_path,
            "Text": content_text.strip()
        }

        results_list.append(paper_data)

    # Save all paper metadata in a single JSON file
    final_json_path = f"json_outputs/extracted_papers.json"
    with open(final_json_path, 'w', encoding='utf-8') as jf:
        json.dump(results_list, jf, ensure_ascii=False, indent=2)
    print("WebCrawler")
    return results_list
fetch_papers("Machine Learning",max_results=10)