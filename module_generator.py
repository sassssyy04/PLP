from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from bertopic import BERTopic
import torch
import json
import os
from datetime import datetime
import uuid

# === Model Setup ===
peft_model_path = "./phi2-finetuned"
peft_config = PeftConfig.from_pretrained(peft_model_path)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, peft_model_path)

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# === Utility Functions ===
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def save_user_query(session_id, query):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    entry = {"timestamp": datetime.now().isoformat(), "query": query}
    with open(path, "a") as f:
        json.dump(entry, f)
        f.write("\n")

def load_session_queries(session_id):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    queries = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                queries.append(json.loads(line)["query"])
    return queries

def generate_text(prompt, max_tokens=250):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip()

# === Topic Modeling + Module Generation ===
def generate_module_for_session(session_id):
    queries = load_session_queries(session_id)
    if not queries:
        return "No queries found for this session."

    # Topic extraction using BERTopic
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(queries)
    topic_info = topic_model.get_topic_info()
    top_topic = topic_info.iloc[1]["Name"]  # Skip -1 outlier
    topic_summary = top_topic.split(":")[-1].strip()

    # Generate module content
    prompts = {
        "explanation": f"### Prompt:\nWrite a brief explanation about this topic: {topic_summary}\n\n### Response:\n",
        "question": f"### Prompt:\nCreate a thought-provoking question on: {topic_summary}\n\n### Response:\n",
        "mcq": f"### Prompt:\nGenerate one multiple choice question (MCQ) with 4 options and one correct answer on: {topic_summary}\n\n### Response:\n"
    }

    module = {
        "topic": topic_summary,
        "explanation": generate_text(prompts["explanation"]),
        "question": generate_text(prompts["question"]),
        "mcq": generate_text(prompts["mcq"])
    }

    return module

# === Example: Create Session and Generate Module ===
if __name__ == "__main__":
    # Example usage
    session_id = str(uuid.uuid4())  # Simulate a user session

    # Simulate interaction
    queries = [
        "What is transfer learning?",
        "How do neural networks retain knowledge?",
        "Can pre-trained models be reused?"
    ]

    for q in queries:
        save_user_query(session_id, q)

    # Generate module
    module = generate_module_for_session(session_id)

    print(f"\n=== Learning Module for Session: {session_id} ===")
    print(f"📚 Topic: {module['topic']}\n")
    print(f"🧠 Explanation:\n{module['explanation']}\n")
    print(f"❓ Open-ended Question:\n{module['question']}\n")
    print(f"📝 MCQ:\n{module['mcq']}\n")
