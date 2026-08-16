import os
import traceback

from flask import Flask, render_template, request, jsonify
from utils.load_data import load_alarm_data, load_incident_data
from rag.chatbot import ask_question
from ai.root_cause import analyze_root_cause

app = Flask(__name__)

# Load data when Flask starts
try:
    alarm_df = load_alarm_data()
except Exception as e:
    print(f"ERROR loading alarm data: {e}")
    alarm_df = None

try:
    incident_df = load_incident_data()
except Exception as e:
    print(f"ERROR loading incident data: {e}")
    incident_df = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/api/alarms", methods=["GET"])
def get_alarms():
    if alarm_df is None:
        return jsonify({
            "error": "Alarm data could not be loaded. Check utils/load_data.py and the data file."
        }), 500

    severity = request.args.get("severity", "All")

    if severity == "All":
        filtered_df = alarm_df.copy()
    else:
        filtered_df = alarm_df[
            alarm_df["Severity"].astype(str).str.strip().str.lower()
            == severity.strip().lower()
        ].copy()

    # Convert NaN/NaT values to JSON-safe None
    filtered_df = filtered_df.where(filtered_df.notna(), None)

    records = filtered_df.to_dict(orient="records")

    distribution = (
        alarm_df["Severity"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )

    kpis = {
        "critical": int(
            alarm_df["Severity"].astype(str).str.strip().str.lower().eq("critical").sum()
        ),
        "major": int(
            alarm_df["Severity"].astype(str).str.strip().str.lower().eq("major").sum()
        ),
        "minor": int(
            alarm_df["Severity"].astype(str).str.strip().str.lower().eq("minor").sum()
        ),
        "total": int(len(alarm_df))
    }

    return jsonify({
        "data": records,
        "kpis": kpis,
        "severity_distribution": distribution
    })


@app.route("/api/sop", methods=["POST"])
def sop():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        answer = ask_question(question)
        return jsonify({"answer": str(answer)})
    except Exception as e:
        print("SOP ERROR:")
        traceback.print_exc()
        return jsonify({
            "error": "Unable to process the SOP question.",
            "details": str(e)
        }), 500


@app.route("/api/root-cause", methods=["POST"])
def root_cause():
    if alarm_df is None:
        return jsonify({"error": "Alarm data is not available."}), 500

    if incident_df is None:
        return jsonify({"error": "Incident data is not available."}), 500

    data = request.get_json(silent=True) or {}
    severity = str(data.get("severity", "All")).strip()

    if severity.lower() == "all":
        filtered_df = alarm_df.copy()
    else:
        filtered_df = alarm_df[
            alarm_df["Severity"].astype(str).str.strip().str.lower()
            == severity.lower()
        ].copy()

    if filtered_df.empty:
        return jsonify({
            "error": "No alarms found for the selected severity."
        }), 400

    try:
        result = analyze_root_cause(filtered_df, incident_df)
        return jsonify({"result": str(result)})
    except Exception as e:
        print("ROOT CAUSE ERROR:")
        traceback.print_exc()
        return jsonify({
            "error": "Unable to perform root cause analysis.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    # Render provides the PORT environment variable.
    # Local development falls back to port 5000.
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
