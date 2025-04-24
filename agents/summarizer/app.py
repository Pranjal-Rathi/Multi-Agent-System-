# app.py

from flask import Flask, jsonify, request, abort
from dataloader import load_papers
from summarizer import chunk_and_summarize, summarize_text

app = Flask(__name__)

# Default summarization parameters
DEFAULT_CHUNK_SIZE        = 512
DEFAULT_MAX_SUMMARY_LEN   = 150
DEFAULT_NUM_BEAMS         = 4

@app.route("/")
def health_check():
    return jsonify({"status": "running", "message": "CrewAI Summarizer Agent is up."})

@app.route("/summarize_all", methods=["GET"])
def summarize_all():
    """
    Summarize every paper loaded by dataloader.load_papers().
    Query parameters (optional):
      - chunk_size (int)
      - max_summary_length (int)
      - num_beams (int)
    """
    # Parse optional query params
    chunk_size      = int(request.args.get("chunk_size",      DEFAULT_CHUNK_SIZE))
    max_sum_len     = int(request.args.get("max_summary_length", DEFAULT_MAX_SUMMARY_LEN))
    num_beams       = int(request.args.get("num_beams",        DEFAULT_NUM_BEAMS))

    papers = load_papers()
    results = []

    for idx, paper in enumerate(papers):
        text   = paper["Text"]
        title  = paper["Title"]
        date   = paper.get("date", "")

        # You can choose between simple summarize_text or chunk_and_summarize:
        summary = chunk_and_summarize(
            text,
            chunk_size=chunk_size,
            max_summary_length=max_sum_len,
            num_beams=num_beams
        )

        results.append({
            "id":       idx,
            "Title":    title,
            "Published":date,
            "Summary":  summary
        })

    return jsonify({"status": "success", "count": len(results), "data": results})


@app.route("/summarize/<int:paper_id>", methods=["GET"])
def summarize_one(paper_id):
    """
    Summarize a single paper by its index in the loaded list.
    """
    papers = load_papers()
    if paper_id < 0 or paper_id >= len(papers):
        abort(404, description=f"paper_id {paper_id} is out of range")

    paper      = papers[paper_id]
    text       = paper["Text"]
    title      = paper["Title"]
    date       = paper.get("date", "")

    # Optional: allow override via query params
    chunk_size    = int(request.args.get("chunk_size",      DEFAULT_CHUNK_SIZE))
    max_sum_len   = int(request.args.get("max_summary_length", DEFAULT_MAX_SUMMARY_LEN))
    num_beams     = int(request.args.get("num_beams",        DEFAULT_NUM_BEAMS))

    summary = chunk_and_summarize(
        text,
        chunk_size=chunk_size,
        max_summary_length=max_sum_len,
        num_beams=num_beams
    )

    return jsonify({
        "status":   "success",
        "id":       paper_id,
        "Title":    title,
        "Published":date,
        "Summary":  summary
    })


@app.route("/summarize_text", methods=["POST"])
def summarize_custom_text():
    """
    Summarize arbitrary text from the request body.
    Expects JSON: { "text": "...", "chunk_size": int?, "max_summary_length": int?, "num_beams": int? }
    """
    payload = request.get_json(force=True)
    raw_text = payload.get("text", "").strip()
    if not raw_text:
        abort(400, description="Request JSON must contain non-empty 'text' field.")

    chunk_size    = int(payload.get("chunk_size",      DEFAULT_CHUNK_SIZE))
    max_sum_len   = int(payload.get("max_summary_length", DEFAULT_MAX_SUMMARY_LEN))
    num_beams     = int(payload.get("num_beams",        DEFAULT_NUM_BEAMS))

    # If the text is short, you can call summarize_text directly:
    if len(raw_text) < chunk_size // 2:
        summary = summarize_text(
            raw_text,
            max_summary_length=max_sum_len,
            num_beams=num_beams
        )
    else:
        summary = chunk_and_summarize(
            raw_text,
            chunk_size=chunk_size,
            max_summary_length=max_sum_len,
            num_beams=num_beams
        )

    return jsonify({"status": "success", "Summary": summary})


if __name__ == "__main__":
    # debug=True for automatic reload; set to False in production
    app.run(host="0.0.0.0", port=5100, debug=True)
