from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# Disable W&B tracking using Hugging Face args
os.environ["WANDB_DISABLED"] = "true"

# Configuration
MAX_EXAMPLES = 5000
BATCH_SIZE = 16
EPOCHS = 5
MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH = "fine_tuned_model3"

print("📥 Loading dataset...")
dataset = load_dataset("CShorten/ML-ArXiv-Papers")['train']

print(f"📊 Preparing {MAX_EXAMPLES} training examples...")
train_examples = []
for item in dataset.select(range(MAX_EXAMPLES)):
    title = item.get("title", "").strip()
    abstract = item.get("abstract", "").strip()
    if title and abstract:
        train_examples.append(InputExample(texts=[title, abstract]))

print("🧠 Loading pre-trained SentenceTransformer model...")
model = SentenceTransformer(MODEL_NAME)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

print("🚀 Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    show_progress_bar=True
)

print(f"💾 Saving fine-tuned model to {SAVE_PATH}...")
model.save(SAVE_PATH)
print("✅ Done.")