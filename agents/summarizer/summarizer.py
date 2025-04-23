# summarizer.py

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

checkpoint = "google-t5/t5-small"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
model      = T5ForConditionalGeneration.from_pretrained(checkpoint)
model.eval()

def summarize_text(
    text: str,
    max_input_length: int   = 512,
    max_summary_length: int = 150,
    num_beams: int          = 4,
    early_stopping: bool    = True
) -> str:
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )
    with torch.no_grad():
        outs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_summary_length,
            num_beams=num_beams,
            early_stopping=early_stopping
        )
    return tokenizer.decode(outs[0], skip_special_tokens=True)

def chunk_and_summarize(
    text: str,
    chunk_size: int = 512,
    **summ_kwargs
) -> str:
    # 1. Split into sentences
    sentences = text.replace("\n", " ").split(". ")
    chunks, curr = [], ""
    for sent in sentences:
        candidate = (curr + " " + sent).strip() + "."
        # estimate token count
        if len(tokenizer.encode(candidate, add_special_tokens=False)) < chunk_size:
            curr = candidate
        else:
            chunks.append(curr)
            curr = sent + "."
    if curr:
        chunks.append(curr)

    # 2. Summarize each chunk
    partials = [summarize_text(c, **summ_kwargs) for c in chunks]

    # 3. (Optional) Merge partials into one final summary
    merged = " ".join(partials)
    if len(tokenizer.encode(merged, add_special_tokens=False)) > chunk_size:
        # if merged summary is still too long, do one more pass
        return summarize_text(merged, **summ_kwargs)
    return merged
