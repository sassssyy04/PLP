# app.py
import os, pickle, sqlite3, re, json, requests, torch
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import openai

# ---------- config ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
DB_PATH = "user_queries.db"
# -------------------------------------------------
#  top-of-file constants  (add or replace)
# -------------------------------------------------
PEFT_PATH = "./phi2-finetuned/checkpoint-240500"        # <-- YOUR LoRA
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev")

# ---------- helpers ----------
def safe_parse_json(raw: str):
    if not raw.strip():
        raise ValueError("safe_parse_json received empty string.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        print("Raw content:", raw)
        raise

# ----- models cached in memory on first request -----
INTENT_MODEL = None
VECTORIZER   = None
TOKENIZER    = None
GEN_MODEL    = None
DEVICE       = None

# -------------------------------------------------
#  load_resources()  – swap in the new LoRA
# -------------------------------------------------
def load_resources():
    """
    Lazy-load intent model, vectorizer and the fine-tuned Phi-2 LoRA.
    Runs once, keeps them in global RAM.
    """
    global INTENT_MODEL, VECTORIZER, TOKENIZER, GEN_MODEL

    if INTENT_MODEL:                                   # already loaded
        return

    # ---------- traditional intent / vectorizer ----------
    with open("intent_classifier.pkl", "rb") as f:
        INTENT_MODEL = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        VECTORIZER = pickle.load(f)

    # ---------- NEW LLM: Phi-2 + LoRA ----------
    peft_cfg   = PeftConfig.from_pretrained(PEFT_PATH)
    base_name  = peft_cfg.base_model_name_or_path        # e.g. "microsoft/phi-2"
    TOKENIZER  = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    TOKENIZER.pad_token = TOKENIZER.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=True,
        device_map="auto",
        offload_folder="./offload",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    GEN_MODEL  = PeftModel.from_pretrained(base_model, PEFT_PATH,
                                           offload_folder="./offload").eval()
    GEN_MODEL.to(DEVICE)

def classify_intent(q):
    vec = VECTORIZER.transform([q])
    return INTENT_MODEL.predict(vec)[0]
# -------------------------------------------------
#  generate_with_peft()  – unchanged except paths
# -------------------------------------------------
def generate_with_peft(prompt):
    prompt = f"### Prompt:\n{prompt}\n\n### Response:\n"
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        ids = GEN_MODEL.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=TOKENIZER.eos_token_id
        )
    return TOKENIZER.decode(ids[0], skip_special_tokens=True)

def generate_with_openai(q):
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": q}],
        temperature=0.3, max_tokens=180,
    )
    
    r = r.choices[0].message.content.strip()

    if not r:
        raise ValueError("OpenAI API returned empty message content.")

    return safe_parse_json(r)

def log_to_db(query, response):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS queries (query TEXT, response TEXT)")
    conn.execute("INSERT INTO queries (query, response) VALUES (?,?)", (query, response))
    conn.commit(); conn.close()

def latest_topics(n_topics=4):
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute("SELECT query, response FROM queries ORDER BY ROWID DESC LIMIT 1").fetchone()
    conn.close()
    if not row: return []
    docs  = list(row)
    vec   = TfidfVectorizer(stop_words="english")
    X     = vec.fit_transform(docs)
    lda   = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(X)
    terms = vec.get_feature_names_out()
    return [" ".join(terms[i] for i in comp.argsort()[:-6:-1]) for comp in lda.components_]

def serper_snippets(topics):
    if not SERPER_API_KEY: return []
    headers  = {"X-API-KEY": SERPER_API_KEY}
    out      = []
    for t in topics:
        r = requests.get("https://google.serper.dev/search", headers=headers, params={"q": t})
        if r.status_code == 200:
            res = r.json().get("organic", [])
            if res: out.append(res[0].get("snippet", ""))
    return out
def modules_interactive(query, explanation):
    prompt = (
        f"""I asked: “{query}”. You answered:\n\n{explanation}\n\n"""
        "Now generate a JSON array of modules. Each module must have:\n"
        "  • module_name: string\n"
        "  • description: string\n"
        "  • resources: array of HTML links\n"
        "  • interactive_html: self-contained HTML snippet with interactive MCQs, drag-and-drop, etc.\n"
        "### Rules...\n"  # (trimmed for brevity)
        "Output *only* the raw JSON array—no extra text or markdown."
    )

    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1500  # ← optionally increase this
    )

    content = r.choices[0].message.content.strip()
    # Strip markdown code block if present
    if content.startswith("```json") and content.endswith("```"):
        content = content[len("```json"): -len("```")].strip()

    if not content.startswith("["):
        raise ValueError(f"Invalid or non-JSON response from OpenAI: {content}")

    return safe_parse_json(content)


# ---------- routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    load_resources()                         # lazy-load on first hit
    modules = snippets = topics = response = None

    if request.method == "POST":
        query = request.form["query"].strip()
        if not query:
            flash("Please enter a question."); return redirect(url_for("index"))

        intent   = classify_intent(query)
        response = generate_with_peft(f"For the following error the user has received, think carefully and provide a fix.{query}") if intent in {"Code example","Error help"} \
                   else generate_with_openai(query)
        # Only keep the text after '### Response:'
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        log_to_db(query, response)

        topics    = latest_topics()
        snippets  = serper_snippets(topics)
        modules   = modules_interactive(query, response)

    return render_template("index.html",
                           response=response,
                           topics=topics or [],
                           snippets=snippets or [],
                           modules=modules or [])

# ---------- run ----------
if __name__ == "__main__":
    app.run(debug=True)
