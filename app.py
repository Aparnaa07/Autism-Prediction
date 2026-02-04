from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import sqlite3
from datetime import datetime
import matplotlib
matplotlib.use('Agg')   # IMPORTANT for Flask
import matplotlib.pyplot as plt


app = Flask(__name__)

# ---------- MODEL ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- DATABASE ----------
DB_PATH = os.path.join(BASE_DIR, "autism.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id TEXT,
            date TEXT,
            risk REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- ENCODING HELPERS ----------
def encode_gender(val):
    return 1 if val == "Male" else 0

def encode_yes_no(val):
    return 1 if val == "Yes" else 0

def encode_map(val, mapping):
    return mapping.get(val, 0)

# ---------- CATEGORY MAPS (MUST MATCH TRAINING) ----------
ETHNICITY_MAP = {
    "White": 0,
    "Asian": 1,
    "Black": 2,
    "Others": 3
}

COUNTRY_MAP = {
    "India": 0,
    "USA": 1,
    "UK": 2,
    "Others": 3
}

RELATION_MAP = {
    "Parent": 0,
    "Self": 1,
    "Relative": 2,
    "Health Care Professional": 3
}

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    child_id = request.form["child_id"]

    # ✅ EXACTLY 19 FEATURES (MATCHING DATASET)
    features = [
        int(request.form["A1_Score"]),
        int(request.form["A2_Score"]),
        int(request.form["A3_Score"]),
        int(request.form["A4_Score"]),
        int(request.form["A5_Score"]),
        int(request.form["A6_Score"]),
        int(request.form["A7_Score"]),
        int(request.form["A8_Score"]),
        int(request.form["A9_Score"]),
        int(request.form["A10_Score"]),
        int(request.form["age"]),
        encode_gender(request.form["gender"]),
        encode_map(request.form["ethnicity"], ETHNICITY_MAP),
        encode_yes_no(request.form["jaundice"]),
        encode_yes_no(request.form["austim"]),
        encode_map(request.form["contry_of_res"], COUNTRY_MAP),
        encode_yes_no(request.form["used_app_before"]),
        int(request.form["result"]),
        encode_map(request.form["relation"], RELATION_MAP)
    ]

    final_features = np.array([features])

    risk = model.predict_proba(final_features)[0][1]

    # ---------- STORE LONGITUDINAL DATA ----------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO risk_history (child_id, date, risk) VALUES (?, ?, ?)",
        (child_id, datetime.now().strftime("%Y-%m-%d"), float(risk))
    )
    conn.commit()
    conn.close()

    if risk < 0.3:
        level = "Low Risk"
    elif risk < 0.6:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    return render_template(
        "result.html",
        risk=round(risk, 2),
        level=level,
        child_id=child_id
    )

@app.route("/history/<child_id>")
def history(child_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date, risk FROM risk_history WHERE child_id=?",
        (child_id,)
    )
    records = cursor.fetchall()
    conn.close()

    return render_template("history.html", child_id=child_id, records=records)

@app.route("/graph/<child_id>")
def graph(child_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date, risk FROM risk_history WHERE child_id=?",
        (child_id,)
    )
    data = cursor.fetchall()
    conn.close()

    # Safety check
    if len(data) == 0:
        return "No data available to plot graph."

    dates = [d[0] for d in data]
    risks = [d[1] for d in data]

    plt.figure(figsize=(6, 4))
    plt.plot(dates, risks, marker='o')
    plt.xlabel("Date")
    plt.ylabel("Risk Score")
    plt.title(f"Autism Risk Progression ({child_id})")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # ✅ FIX: use absolute static path
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)  # ensure folder exists

    graph_filename = f"{child_id}_risk.png"
    graph_path = os.path.join(static_dir, graph_filename)

    plt.savefig(graph_path)
    plt.close()

    return render_template(
        "graph.html",
        child_id=child_id,
        graph_image=graph_filename
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

