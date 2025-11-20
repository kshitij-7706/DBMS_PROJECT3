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


# -----------------------------
# 2. Load model
# -----------------------------
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# -----------------------------
# 3. Convert English → SQL
# -----------------------------
def nl_to_sql(nl_query):
    input_text = "translate English to SQL: " + nl_query
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        columns, result = execute_sql(sql_query)

        print("SQL Query:", sql_query)
        print("Columns:", columns)
        print("Result:", result)


        if columns is None:
            return render_template("result.html", user_query=user_query, sql_query=sql_query, error=result)
        return render_template("result.html", user_query=user_query, sql_query=sql_query, columns=columns, result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
