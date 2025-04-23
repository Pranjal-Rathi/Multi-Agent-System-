# # new_agent.py

# import startup_patch  # Must come FIRST

# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import graphviz
# from io import BytesIO

# # --- Page setup ---
# st.set_page_config(
#     page_title="Research Workflow Generator",
#     page_icon="🧠",
#     layout="wide"
# )
# st.title("🧠 Research Workflow Generator Agent")
# st.markdown(
#     "Enter your research topic and any optional constraints; "
#     "this agent will output both a textual workflow and a visual diagram."
# )

# # --- Load model ---
# MODEL_ID = "tiiuae/falcon-7b-instruct"

# @st.cache_resource
# def load_model(model_id):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
#     return tokenizer, model, device

# tokenizer, model, device = load_model(MODEL_ID)

# # --- Generate workflow text ---
# def generate_workflow_text(topic: str, constraints: str) -> str:
#     prompt = (
#         "You are an expert research assistant. "
#         "Given a research topic and optional constraints, "
#         "provide a clear, numbered, step-by-step workflow plan.\n\n"
#         f"Topic: {topic}\n"
#         f"Constraints: {constraints or 'None'}\n\n"
#         "Workflow:\n"
#     )
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=512,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.9,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
#     return decoded.split("Workflow:")[-1].strip()

# # --- Convert text to Graphviz ---
# def text_to_graphviz(text: str) -> graphviz.Digraph:
#     lines = [line.strip() for line in text.split("\n") if line.strip()]
#     dot = graphviz.Digraph(format="png")
#     for idx, line in enumerate(lines):
#         dot.node(str(idx), label=line, shape="box", style="filled", fillcolor="#E8F4FA")
#         if idx > 0:
#             dot.edge(str(idx-1), str(idx))
#     return dot

# # --- UI inputs ---
# topic = st.text_input("🔬 Research Topic")
# constraints = st.text_area("⚙️ Constraints / Tools (optional)")

# if st.button("Generate Workflow") and topic:
#     with st.spinner("Generating workflow…"):
#         workflow_text = generate_workflow_text(topic, constraints)
#         st.markdown("### 📄 Workflow Steps")
#         st.markdown(f"```text\n{workflow_text}\n```")

#         dot = text_to_graphviz(workflow_text)
#         st.markdown("### 🧩 Visual Workflow (Interactive)")
#         st.graphviz_chart(dot, use_container_width=True)

#         png_bytes = dot.pipe()
#         st.markdown("### 📥 Downloadable Workflow Image")
#         # st.image(png_bytes, caption="Workflow Diagram (PNG)", use_column_width=True)

#         bio = BytesIO(png_bytes)
#         bio.seek(0)
#         st.download_button(
#             "Download PNG",
#             data=bio,
#             file_name="research_workflow.png",
#             mime="image/png"
#         )


import startup_patch  # Must come FIRST

import streamlit as st
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import graphviz
from io import BytesIO

# --- Page setup ---
st.set_page_config(
    page_title="Research Workflow Generator",
    page_icon="🧠",
    layout="wide"
)

# --- Load model ---
MODEL_ID = "tiiuae/falcon-7b-instruct"

@st.cache_resource
def load_model(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model(MODEL_ID)

# --- Generate workflow text ---
def generate_workflow_text(topic: str, constraints: str) -> str:
    prompt = (
        "You are an expert research assistant. "
        "Given a research topic and optional constraints, "
        "provide a clear, numbered, step-by-step workflow plan.\n\n"
        f"Topic: {topic}\n"
        f"Constraints: {constraints or 'None'}\n\n"
        "Workflow:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=350,  # Reduced for faster generation
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return decoded.split("Workflow:")[-1].strip()

# --- Convert text to Graphviz ---
def text_to_graphviz(text: str) -> graphviz.Digraph:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    dot = graphviz.Digraph(format="png")
    for idx, line in enumerate(lines):
        dot.node(str(idx), label=line, shape="box", style="filled", fillcolor="#E8F4FA")
        if idx > 0:
            dot.edge(str(idx-1), str(idx))
    return dot

# --- Optional: Stylize steps with emojis ---
def stylize_steps(text: str) -> str:
    lines = text.strip().split("\n")
    return "\n".join([f"🔹 {line}" for line in lines])

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4f/Iconic_image_for_AI.png", width=100)
    st.markdown("### Input Parameters")
    topic = st.text_input("🔬 Research Topic")
    constraints = st.text_area("⚙️ Constraints / Tools (optional)")
    generate = st.button("Generate Workflow")

# --- Header UI ---
col1, col2 = st.columns([1, 10])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4f/Iconic_image_for_AI.png", width=60)
with col2:
    st.markdown("## Research Workflow Generator")
    st.caption("Your General Research Support Agent powered by Falcon-7B")

with st.expander("🧠 Model Info", expanded=False):
    st.markdown("""
    - **Model**: Falcon-7B Instruct  
    - **Provider**: TII UAE  
    - **Capabilities**: Research planning, task sequencing, constraint handling  
    - **Device**: CPU/GPU (auto-selected)  
    """)

# --- Run logic ---
if generate and topic:
    with st.spinner("Generating research workflow..."):
        # Simulate progress bar during generation
        progress = st.progress(0)
        for i in range(1, 6):
            time.sleep(0.3)  # Fake delay while loading model
            progress.progress(i * 20)
        workflow_text = generate_workflow_text(topic, constraints)
        dot = text_to_graphviz(workflow_text)
        png_bytes = dot.pipe()
        bio = BytesIO(png_bytes)
        bio.seek(0)

    st.success("✅ Workflow generated successfully!")

    # --- Output Tabs ---
    tab1, tab2, tab3 = st.tabs(["📄 Text View", "🧩 Visual Diagram", "📥 Download"])
    with tab1:
        st.markdown("#### Step-by-step Workflow")
        st.markdown(f"```text\n{workflow_text}\n```")

    with tab2:
        st.markdown("#### Visual Workflow")
        st.graphviz_chart(dot, use_container_width=True)

    with tab3:
        st.download_button(
            "Download PNG",
            data=bio,
            file_name="research_workflow.png",
            mime="image/png"
        )
