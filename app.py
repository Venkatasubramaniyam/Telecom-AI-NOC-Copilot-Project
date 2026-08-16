import os
import traceback

from flask import Flask, render_template, request, jsonify
from utils.load_data import load_alarm_data, load_incident_data
from rag.chatbot import ask_question
from ai.root_cause import analyze_root_cause

app = Flask(__name__)


# Load data once when Flask starts.
try:
    alarm_df = load_alarm_data()
    print("Alarm data loaded successfully.")
except Exception as e:
    print(f"ERROR loading alarm data: {e}")
    traceback.print_exc()
    alarm_df = None

try:
    incident_df = load_incident_data()
    print("Incident data loaded successfully.")
except Exception as e:
    print(f"ERROR loading incident data: {e}")
    traceback.print_exc()
    incident_df = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/api/alarms", methods=["GET"])
def get_alarms():
    if alarm_df is None:
        return jsonify({
            "error": "Alarm data could not be loaded."
        }), 500

    severity = request.args.get("severity", "All")

    if severity.lower() == "all":
        filtered_df = alarm_df.copy()
    else:
        filtered_df = alarm_df[
            alarm_df["Severity"].astype(str).str.strip().str.lower()
            == severity.strip().lower()
        ].copy()

    filtered_df = filtered_df.where(filtered_df.notna(), None)

    records = filtered_df.to_dict(orient="records")

    distribution = (
        alarm_df["Severity"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )

    severity_values = alarm_df["Severity"].astype(str).str.strip().str.lower()

    kpis = {
        "critical": int(severity_values.eq("critical").sum()),
        "major": int(severity_values.eq("major").sum()),
        "minor": int(severity_values.eq("minor").sum()),
        "total": int(len(alarm_df)),
    }

    return jsonify({
        "data": records,
        "kpis": kpis,
        "severity_distribution": distribution,
    })


@app.route("/api/sop", methods=["POST"])
def sop():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        answer = ask_question(question)
        return jsonify({"answer": str(answer)}), 200

    except Exception as e:
        print("SOP ERROR:")
        traceback.print_exc()

        return jsonify({
            "error": "Unable to process the SOP question.",
            "details": str(e),
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
        print(
            f"ROOT CAUSE START: alarms={len(filtered_df)}, "
            f"history={len(incident_df)}, severity={severity}"
        )

        result = analyze_root_cause(filtered_df, incident_df)

        print("ROOT CAUSE SUCCESS")

        return jsonify({
            "result": str(result)
        }), 200

    except Exception as e:
        print("ROOT CAUSE ERROR:")
        traceback.print_exc()

        return jsonify({
            "error": "Unable to perform root cause analysis.",
            "details": str(e),
        }), 500


@app.errorhandler(500)
def handle_500(error):
    print("FLASK 500 ERROR:")
    traceback.print_exc()

    return jsonify({
        "error": "Internal server error.",
        "details": str(error),
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
