from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import gradio as gr
import json, os, uuid
from datetime import datetime
from bertopic import BERTopic
import os 

api_key = os.getenv("OPENAI_API_KEY")
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

# === Session Storage ===
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def get_session_path(session_id):
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def save_user_query(session_id, query):
    entry = {"timestamp": datetime.now().isoformat(), "query": query}
    with open(get_session_path(session_id), "a") as f:
        json.dump(entry, f)
        f.write("\n")

def load_session_queries(session_id):
    path = get_session_path(session_id)
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

session_histories = {}

# === Chat Function with Displayed History ===
def chat_with_history(user_input, chat_state, session_id):
    if session_id not in session_histories:
        session_histories[session_id] = []

    # Save query
    save_user_query(session_id, user_input)

    # Append to history for prompt
    session_histories[session_id].append({"role": "user", "content": user_input})

    prompt = "### Conversation:\n"
    for msg in session_histories[session_id]:
        role = "User" if msg["role"] == "user" else "AI"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "AI:"

    # Generate response
    reply = generate_text(prompt, max_tokens=200)
    session_histories[session_id].append({"role": "ai", "content": reply})

    # Update visible chat history
    chat_state.append((user_input, reply))
    return chat_state, chat_state

from openai import OpenAI

client = OpenAI(api_key = api_key)  # or hardcode your key for dev")  # Uses the default API key from environment or .openai config

def generate_learning_module(session_id):
    queries = load_session_queries(session_id)
    if not queries:
        return "No queries yet for this session!", "", "", ""

    joined_queries = "\n".join(queries)
    prompt = f"""
You are an AI education assistant. A user asked the following questions in a session:

{joined_queries}

Based on this, generate the following:
1. A clear topic name
2. A short explanation (~100 words)
3. An open-ended question to encourage deeper thinking
4. One multiple-choice question with 4 options (A–D) and indicate the correct answer

Return the output in this format:
Topic: ...
Explanation: ...
Open-ended Question: ...
MCQ: ...
"""

    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful teaching assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=600
    )

    output = response.choices[0].message.content
    
    # Parse output into parts (rudimentary)
    topic = explanation = question = mcq = ""
    for line in output.splitlines():
        if line.lower().startswith("topic:"):
            topic = line.split(":", 1)[1].strip()
        elif line.lower().startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()
        elif line.lower().startswith("open-ended question:"):
            question = line.split(":", 1)[1].strip()
        elif line.lower().startswith("mcq:"):
            mcq = line.split(":", 1)[1].strip()
        else:
            if mcq: mcq += "\n" + line

    return topic, explanation, question, mcq



# === UI ===
with gr.Blocks(title="LLM Chat + Learning Module") as app:
    session_id = gr.State(str(uuid.uuid4()))

    with gr.Tab("💬 Chat"):
        gr.Markdown("## 💬 AI Chat with Memory")
        chatbot = gr.Chatbot(label="Chat History")
        user_msg = gr.Textbox(placeholder="Type your message...", show_label=False)
        send_btn = gr.Button("Send")
        state = gr.State([])

        send_btn.click(fn=chat_with_history,
                       inputs=[user_msg, state, session_id],
                       outputs=[chatbot, state])
    
    with gr.Tab("📚 Module Generator"):
        gr.Markdown("## 📚 Generate Learning Module")
        gen_btn = gr.Button("Generate from This Session")
        topic = gr.Textbox(label="🧠 Topic")
        explanation = gr.Textbox(label="📘 Explanation", lines=3)
        question = gr.Textbox(label="❓ Open-ended Question")
        mcq = gr.Textbox(label="📝 MCQ")

        gen_btn.click(fn=generate_learning_module, inputs=session_id, outputs=[topic, explanation, question, mcq])

app.launch()
