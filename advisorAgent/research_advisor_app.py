# import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Manually define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the model and tokenizer (only once at the start)
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# # Check if the model is in meta state and use .to_empty() to initialize the model properly
# if isinstance(model, torch.nn.Module) and model.device == torch.device("meta"):
#     model = model.to_empty()

# # Now move the model to the correct device (CPU or GPU)
# model = model.to(device)

# # Function to handle chat and generate response
# def generate_response(user_input):
#     # Format the prompt using TinyLlama chat format
#     prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"

#     # Tokenize input
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     # Generate output
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         do_sample=True,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     # Decode and clean up the response
#     decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
#     response = decoded_output.split("<|assistant|>")[-1].strip()
    
#     return response

# # Streamlit interface
# st.title("TinyLlama Chat Assistant")

# # Initialize session state to store chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # User input
# user_input = st.text_input("You: ", "")

# # Button to generate response
# if st.button("Send") and user_input:
#     # Add user input to chat history
#     st.session_state.chat_history.append(f"You: {user_input}")

#     # Get response from model
#     assistant_response = generate_response(user_input)
    
#     # Add assistant's response to chat history
#     st.session_state.chat_history.append(f"Assistant: {assistant_response}")

# # Display chat history
# for message in st.session_state.chat_history:
#     if "You:" in message:
#         st.markdown(f"<div style='color: #2C3E50; background-color: #ECF0F1; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>{message}</div>", unsafe_allow_html=True)
#     else:
#         st.markdown(f"<div style='color: #ECF0F1; background-color: #34495E; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>{message}</div>", unsafe_allow_html=True)


import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# --------------------- Styling ---------------------
st.set_page_config(page_title="TinyLlama Chat Assistant", layout="centered")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------- Sidebar Info ---------------------
with st.sidebar:
    # st.image("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/logo.png", width=100)
    st.markdown("### Agent Info")
    st.write("🔹 Model: TinyLlama-1.1B-Chat")
    st.write(f"🔹 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.write("🔹 Role: General Research Assistant")
    st.markdown("---")
    st.markdown("Part of Multi-Agent Research System")

# --------------------- Setup ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Handle meta state
if isinstance(model, torch.nn.Module) and model.device == torch.device("meta"):
    model = model.to_empty()

model = model.to(device)

# --------------------- Chat Function ---------------------
def generate_response(user_input):
    prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = decoded_output.split("<|assistant|>")[-1].strip()
    return response

# Optional typing effect (for realism)
def type_text(text):
    typed = ""
    placeholder = st.empty()
    for c in text:
        typed += c
        time.sleep(0.01)
        placeholder.markdown(typed)

# --------------------- Session State ---------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --------------------- Header ---------------------
col1, col2 = st.columns([1, 5])
with col1:
    pass
    # st.image("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/logo.png", width=60)
with col2:
    st.markdown("## TinyLlama Chat Assistant")
    st.caption("Your General Research Support Agent")

# --------------------- Clear Chat Button ---------------------
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# --------------------- Chat History ---------------------
for message in st.session_state.chat_history:
    role, content = message.split(": ", 1)
    with st.chat_message("user" if role == "You" else "assistant"):
        st.markdown(content)

# --------------------- Input & Response ---------------------
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.chat_history.append(f"You: {user_input}")
    with st.chat_message("user"):
        st.markdown(user_input)

    assistant_response = generate_response(user_input)
    st.session_state.chat_history.append(f"Assistant: {assistant_response}")

    with st.chat_message("assistant"):
        # st.markdown(assistant_response)
        type_text(assistant_response)  # Optional: enable typing effect
