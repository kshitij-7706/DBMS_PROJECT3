from flask import Flask, render_template, request
import mysql.connector
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# -----------------------------
# 1. Database connection
# -----------------------------
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

# ----------------------------------------------------
# 2. LAZY LOAD MODEL (Fix for Render Memory Issue)
# ----------------------------------------------------
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, tokenizer
    if model is None:
        print("🔥 Loading FLAN-T5 model (lazy)...")
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        print("✅ Model loaded successfully!")
    return model, tokenizer

# -----------------------------
# 3. Convert English → SQL
# -----------------------------
def nl_to_sql(nl_query):
    model, tokenizer = load_model()

    prompt = f"Write an SQL query for: {nl_query}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query.strip()

# -----------------------------
# 4. Execute SQL on DB
# -----------------------------
def execute_sql(sql_query):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return columns, rows
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

# -----------------------------
# 5. Web Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["user_query"]

        sql_query = nl_to_sql(user_query)
        print("SQL Query:", sql_query)

        columns, result = execute_sql(sql_query)

        if columns is None:
            return render_template("result.html",
                                   user_query=user_query,
                                   sql_query=sql_query,
                                   error=result)

        return render_template("result.html",
                               user_query=user_query,
                               sql_query=sql_query,
                               columns=columns,
                               result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
