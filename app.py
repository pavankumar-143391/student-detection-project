from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, flash, make_response, redirect, render_template, request, url_for

try:
    import joblib
    import pandas as pd
except ImportError:  # The app can still run in demo mode without ML packages.
    joblib = None
    pd = None

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
DB_PATH = INSTANCE_DIR / "students.db"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "local-development-key-change-before-deploy")

FEATURE_COLUMNS = [
    "Age at enrollment",
    "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
]


@contextmanager
def get_db():
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with get_db() as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                programme TEXT NOT NULL,
                year_of_study INTEGER NOT NULL,
                age INTEGER NOT NULL,
                admission_grade REAL NOT NULL,
                previous_qualification_grade REAL NOT NULL,
                first_sem_enrolled INTEGER NOT NULL,
                first_sem_evaluations INTEGER NOT NULL,
                first_sem_approved INTEGER NOT NULL,
                first_sem_grade REAL NOT NULL,
                first_sem_without_evaluations INTEGER NOT NULL,
                tuition_fees_up_to_date INTEGER NOT NULL,
                debtor INTEGER NOT NULL,
                scholarship_holder INTEGER NOT NULL,
                risk_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                reasons TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        count = db.execute("SELECT COUNT(*) AS total FROM students").fetchone()["total"]
        if count == 0:
            demo_rows = [
                ("STU-2401", "Aarav Mehta", "B.Tech Computer Science", 2, 20, 156, 128, 6, 8, 2, 8.4, 3, 0, 1, 0, .78, "High", ["Low first-semester approval", "Low semester grade", "Fees not up to date"]),
                ("STU-2402", "Diya Sharma", "BBA", 1, 19, 174, 152, 6, 7, 6, 15.8, 0, 1, 0, 0, .12, "Low", ["No major warning signals in the entered data"]),
                ("STU-2403", "Kabir Rao", "B.Tech AI & ML", 2, 21, 142, 118, 7, 9, 4, 11.2, 1, 1, 0, 0, .43, "Medium", ["Moderate first-semester approval", "Grade below the expected range"]),
                ("STU-2404", "Mira Nair", "B.Com", 3, 21, 168, 146, 6, 8, 6, 16.1, 0, 1, 0, 1, .08, "Low", ["No major warning signals in the entered data"]),
                ("STU-2405", "Reyansh Singh", "B.Tech Electronics", 1, 18, 121, 102, 6, 9, 3, 9.6, 2, 0, 1, 0, .66, "High", ["Low first-semester approval", "Fees not up to date", "Several evaluations not completed"]),
                ("STU-2406", "Anaya Iyer", "BCA", 2, 20, 151, 134, 6, 8, 5, 13.1, 1, 1, 0, 1, .29, "Low", ["A few indicators may benefit from routine follow-up"]),
            ]
            for row in demo_rows:
                db.execute(
                    """INSERT INTO students (
                        student_id, name, programme, year_of_study, age, admission_grade,
                        previous_qualification_grade, first_sem_enrolled, first_sem_evaluations,
                        first_sem_approved, first_sem_grade, first_sem_without_evaluations,
                        tuition_fees_up_to_date, debtor, scholarship_holder, risk_probability,
                        risk_level, reasons, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (*row[:15], row[15], row[16], json.dumps(row[17]), datetime.now().strftime("%Y-%m-%d %H:%M")),
                )


def get_model() -> Any | None:
    if joblib is None or not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        app.logger.exception("Unable to load model.joblib")
        return None


def form_features(form: Any) -> dict[str, float]:
    return {
        "Age at enrollment": float(form.get("age", 20)),
        "Admission grade": float(form.get("admission_grade", 140)),
        "Previous qualification (grade)": float(form.get("previous_qualification_grade", 120)),
        "Curricular units 1st sem (enrolled)": float(form.get("first_sem_enrolled", 6)),
        "Curricular units 1st sem (evaluations)": float(form.get("first_sem_evaluations", 8)),
        "Curricular units 1st sem (approved)": float(form.get("first_sem_approved", 4)),
        "Curricular units 1st sem (grade)": float(form.get("first_sem_grade", 11)),
        "Curricular units 1st sem (without evaluations)": float(form.get("first_sem_without_evaluations", 0)),
        "Tuition fees up to date": float(form.get("tuition_fees_up_to_date", 0)),
        "Debtor": float(form.get("debtor", 0)),
        "Scholarship holder": float(form.get("scholarship_holder", 0)),
    }


def explain_signals(features: dict[str, float]) -> list[str]:
    signals: list[str] = []
    enrolled = max(features["Curricular units 1st sem (enrolled)"], 1)
    approved = features["Curricular units 1st sem (approved)"]
    approval_rate = approved / enrolled
    if approval_rate < 0.4:
        signals.append("Fewer than 40% of enrolled first-semester units are approved")
    elif approval_rate < 0.7:
        signals.append("First-semester unit approval is below 70%")
    if features["Curricular units 1st sem (grade)"] < 10:
        signals.append("First-semester average grade is below 10/20")
    if features["Curricular units 1st sem (without evaluations)"] >= 2:
        signals.append("Several enrolled units have no recorded evaluation")
    if features["Tuition fees up to date"] == 0:
        signals.append("Tuition fees are not marked as up to date")
    if features["Debtor"] == 1:
        signals.append("A debtor flag is present in the record")
    if not signals:
        signals.append("No strong warning signals were identified in the entered fields")
    return signals[:4]


def demo_probability(features: dict[str, float]) -> float:
    """Transparent fallback score for demonstration only; it is not a trained ML model."""
    risk = 0.12
    enrolled = max(features["Curricular units 1st sem (enrolled)"], 1)
    approval_rate = features["Curricular units 1st sem (approved)"] / enrolled
    if approval_rate < 0.35:
        risk += 0.28
    elif approval_rate < 0.60:
        risk += 0.17
    elif approval_rate < 0.80:
        risk += 0.07
    grade = features["Curricular units 1st sem (grade)"]
    if grade < 8:
        risk += 0.22
    elif grade < 10:
        risk += 0.15
    elif grade < 12:
        risk += 0.07
    if features["Curricular units 1st sem (without evaluations)"] >= 3:
        risk += 0.12
    elif features["Curricular units 1st sem (without evaluations)"] >= 1:
        risk += 0.05
    if features["Tuition fees up to date"] == 0:
        risk += 0.10
    if features["Debtor"] == 1:
        risk += 0.06
    if features["Admission grade"] < 100:
        risk += 0.04
    return max(0.03, min(risk, 0.95))


def predict_risk(features: dict[str, float]) -> tuple[float, str]:
    model = get_model()
    if model is not None and pd is not None:
        row = pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        probabilities = model.predict_proba(row)[0]
        positive_index = list(model.classes_).index(1)
        return float(probabilities[positive_index]), "Trained machine-learning model"
    return demo_probability(features), "Demo scoring (not trained ML)"


def risk_label(probability: float) -> str:
    if probability >= 0.60:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


@app.route("/")
def dashboard():
    with get_db() as db:
        students = db.execute("SELECT * FROM students ORDER BY risk_probability DESC, created_at DESC").fetchall()
    total = len(students)
    high = sum(row["risk_level"] == "High" for row in students)
    medium = sum(row["risk_level"] == "Medium" for row in students)
    low = sum(row["risk_level"] == "Low" for row in students)
    average = (sum(row["risk_probability"] for row in students) / total * 100) if total else 0
    metrics = {"total": total, "high": high, "medium": medium, "low": low, "average": average}
    model_ready = MODEL_PATH.exists() and get_model() is not None
    try:
        training_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if model_ready and METRICS_PATH.exists() else None
    except (OSError, json.JSONDecodeError):
        training_metrics = None
    return render_template(
        "index.html",
        students=students,
        metrics=metrics,
        model_ready=model_ready,
        model_mode="Trained ML model" if model_ready else "Demo scoring",
        training_metrics=training_metrics,
    )


@app.route("/assess", methods=["POST"])
def assess_student():
    try:
        student_id = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        programme = request.form.get("programme", "").strip()
        if not student_id or not name or not programme:
            flash("Please enter the student ID, name and programme.", "error")
            return redirect(url_for("dashboard") + "#assessment")
        features = form_features(request.form)
        probability, _mode = predict_risk(features)
        level = risk_label(probability)
        reasons = explain_signals(features)
        values = (
            student_id, name, programme, int(request.form.get("year_of_study", 1)),
            int(features["Age at enrollment"]), features["Admission grade"],
            features["Previous qualification (grade)"],
            int(features["Curricular units 1st sem (enrolled)"]),
            int(features["Curricular units 1st sem (evaluations)"]),
            int(features["Curricular units 1st sem (approved)"]),
            features["Curricular units 1st sem (grade)"],
            int(features["Curricular units 1st sem (without evaluations)"]),
            int(features["Tuition fees up to date"]), int(features["Debtor"]),
            int(features["Scholarship holder"]), probability, level,
            json.dumps(reasons), datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        with get_db() as db:
            db.execute(
                """INSERT INTO students (
                    student_id, name, programme, year_of_study, age, admission_grade,
                    previous_qualification_grade, first_sem_enrolled, first_sem_evaluations,
                    first_sem_approved, first_sem_grade, first_sem_without_evaluations,
                    tuition_fees_up_to_date, debtor, scholarship_holder, risk_probability,
                    risk_level, reasons, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
        flash(f"Assessment saved for {name}. Treat this result as a prompt for human review, not a decision.", "success")
    except sqlite3.IntegrityError:
        flash("That student ID already exists. Use a unique ID or update the existing record.", "error")
    except (ValueError, TypeError):
        flash("Please check that all numeric fields contain valid numbers.", "error")
    return redirect(url_for("dashboard"))


@app.route("/student/<int:record_id>")
def student_detail(record_id: int):
    with get_db() as db:
        student = db.execute("SELECT * FROM students WHERE id = ?", (record_id,)).fetchone()
    if student is None:
        flash("Student record not found.", "error")
        return redirect(url_for("dashboard"))
    return render_template("student.html", student=student, reasons=json.loads(student["reasons"]))


@app.route("/export.csv")
def export_csv():
    with get_db() as db:
        rows = db.execute("SELECT * FROM students ORDER BY risk_probability DESC").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Programme", "Year", "Age", "Admission grade", "Previous qualification grade", "First-sem enrolled", "First-sem evaluations", "First-sem approved", "First-sem grade", "Without evaluations", "Fees up to date", "Debtor", "Scholarship holder", "Risk probability", "Risk level", "Review signals", "Assessed at"])
    for row in rows:
        writer.writerow([row["student_id"], row["name"], row["programme"], row["year_of_study"], row["age"], row["admission_grade"], row["previous_qualification_grade"], row["first_sem_enrolled"], row["first_sem_evaluations"], row["first_sem_approved"], row["first_sem_grade"], row["first_sem_without_evaluations"], row["tuition_fees_up_to_date"], row["debtor"], row["scholarship_holder"], round(row["risk_probability"] * 100, 1), row["risk_level"], row["reasons"], row["created_at"]])
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=student-risk-report.csv"
    return response


@app.route("/health")
def health():
    return {"status": "ok", "mode": "trained_ml" if MODEL_PATH.exists() else "demo"}


init_db()

if __name__ == "__main__":
    app.run(debug=True)
