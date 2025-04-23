# import json
# from faker import Faker
# import random
# from datetime import datetime, timedelta

# fake = Faker()
# topics = [
#     "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
#     "Quantum Computing", "Healthcare AI", "Federated Learning", "Graph Neural Networks",
#     "Climate Modeling", "Anomaly Detection", "Generative Models", "Bias in AI",
#     "Multi-Agent Systems", "Edge Computing", "Sustainable AI", "Autonomous Vehicles"
# ]

# # Define a date range (e.g., from Jan 2020 to today)
# start_date = datetime(2020, 1, 1)
# end_date = datetime.today()

# def random_date(start, end):
#     """Generate a random datetime between `start` and `end`"""
#     delta = end - start
#     random_days = random.randint(0, delta.days)
#     return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

# sample_data = []

# for i in range(120):  # Generate 120 mock entries
#     topic = random.choice(topics)
#     title = f"{topic}: {fake.sentence(nb_words=6).rstrip('.')}"
#     summary = fake.paragraph(nb_sentences=4)
#     date = random_date(start_date, end_date)
    
#     sample_data.append({
#         "title": title,
#         "summary": summary,
#         "date": date
#     })

# # Save the JSON file
# with open("data/summaries.json", "w", encoding="utf-8") as f:
#     json.dump(sample_data, f, indent=2, ensure_ascii=False)

# print("✅ Generated 120 mock summaries with dates in data/summaries.json")
import json
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

# Expanded list of 30 “topic seeds”
topics = [
    "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
    "Quantum Computing", "Healthcare AI", "Federated Learning", "Graph Neural Networks",
    "Climate Modeling", "Anomaly Detection", "Generative Models", "Bias in AI",
    "Multi-Agent Systems", "Edge Computing", "Sustainable AI", "Autonomous Vehicles",
    "Explainable AI", "Time Series Forecasting", "Recommender Systems", "Robotics",
    "Cybersecurity", "Drug Discovery", "IoT Analytics", "Smart Grids", "Financial AI",
    "Social Network Analysis", "Human–Computer Interaction", "Bioinformatics",
    "Speech Recognition", "Augmented Reality", "Ethical AI"
]

# Generate random dates between Jan 1, 2020 and today
start_date = datetime(2020, 1, 1)
end_date   = datetime.today()
def random_date(start, end):
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

sample_data = []
N = 300
for _ in range(N):
    # pick 1–2 topics and weave into title
    chosen = random.sample(topics, k=random.choice([1,2]))
    title = " & ".join(chosen) + ": " + fake.sentence(nb_words=6).rstrip(".")
    # seed the summary with those topic words
    summary = fake.paragraph(nb_sentences=3) + " This work explores " + ", ".join(chosen).lower() + "."
    date = random_date(start_date, end_date)
    sample_data.append({
        "title":   title,
        "summary": summary,
        "date":    date
    })

# Write out
with open("data/summaries.json","w",encoding="utf-8") as f:
    json.dump(sample_data, f, indent=2, ensure_ascii=False)

print(f"✅ Generated {N} mock summaries with dates and richer topic variety in data/summaries.json")
