from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import gradio as gr
import json
import os
from datetime import datetime

# Load PEFT config and base model
peft_model_path = "./phi2-finetuned"
peft_config = PeftConfig.from_pretrained(peft_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True
)

# Apply LoRA
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Memory (in-session)
chat_history = []

# Log path for topic modeling
LOG_PATH = "user_queries.json"

def save_query_to_log(query):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query
    }
    with open(LOG_PATH, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")

# Main chat function with memory
def chat_with_memory(user_input):
    global chat_history

    # Save query
    save_query_to_log(user_input)

    # Add user input to history
    chat_history.append({"role": "user", "content": user_input})

    # Create prompt from history
    prompt = "### Conversation:\n"
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "AI"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "AI:"

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    ai_response = full_output[len(prompt):].strip()

    chat_history.append({"role": "ai", "content": ai_response})
    return ai_response

# Reset function
def reset_chat():
    global chat_history
    chat_history = []
    return "Memory reset! Start a new conversation."

# Gradio interface
gr.Interface(
    fn=chat_with_memory,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything...", label="Your Message"),
    outputs=gr.Textbox(lines=4, label="AI Response"),
    title="Finetuned AI Chatbot with Memory",
    description="Conversational memory + topic logging for future learning modules.",
    live=False,
    allow_flagging="never",
    theme="default",
    examples=None
).launch()
