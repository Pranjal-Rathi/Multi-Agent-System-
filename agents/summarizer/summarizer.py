from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize T5 model and tokenizer
model_name = "t5-base"  # You may choose t5-small for faster inference during prototyping
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text: str, max_length: int = 150) -> str:
    """
    Generate an abstractive summary for the given text using T5.
    """
    # Prepare the input prompt: the task prefix is 'summarize:'
    input_prompt = f"summarize: {text}"
    inputs = tokenizer.encode(input_prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == '__main__':
    # Example text for summarization – replace with actual abstract or a combination from your dataset.
    sample_text = (
        "In this paper, the weak galerkin finite element method for second order elliptic problems "
        "employing polygonal meshes is discussed. The method presents a new approach to finite element "
        "analysis and offers significant improvements in computational efficiency, making it suitable for "
        "solving complex mathematical models in various applications."
    )
    summary = summarize_text(sample_text)
    print("Generated Summary:")
    print(summary)
