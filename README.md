# RetainIQ — Early Detection of Student Dropout Risk & Retention Intelligence System

A Flask web application prototype for student retention teams. It stores student assessments, displays a risk dashboard, provides review signals, and exports records to CSV. It includes a Random Forest training script for the UCI Student Dropout dataset.

> **Responsible-use note:** A risk score is a screening signal, not a statement about a student's future. A trained model is only activated after you train it with the dataset. Until then, the website uses a clearly labelled demonstration scoring formula. Do not use the sample records or demo scores for real student decisions.

## Features

- Responsive dashboard with High / Medium / Low risk counts.
- Student assessment form using first-semester academic and institutional indicators.
- Individual student risk profile and the signals that should be reviewed.
- SQLite storage for local development.
- CSV report export.
- Optional Random Forest model training with a held-out test set.
- Model evaluation metrics: precision, recall, F1, ROC-AUC and confusion matrix.
- Search and risk-level filters for the student watchlist.

## 1. Run the website on your computer

Install Python 3.10 or newer. Open a terminal in this project folder and run:

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser. The first run creates a local SQLite database and inserts fictional sample records.

## 2. Train the machine-learning model

The project is prepared for the public **UCI Predict Students' Dropout and Academic Success** dataset:

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

1. Download the dataset from the UCI page.
2. Extract the CSV file (usually named `data.csv`).
3. Copy it into this project's `data` folder.
4. Rename it to **`student_dropout.csv`**.
5. Confirm the CSV contains the `Target` column and the feature columns used in `train_model.py`.
6. With your virtual environment activated, run:

```bash
python train_model.py
```

The script creates `model.joblib` and `model_metrics.json`. Restart the Flask app. The dashboard should then display **Trained ML model** and the holdout metrics.

### Important modelling limitation

The UCI target contains three eventual outcomes: `Dropout`, `Graduate`, and `Enrolled`. This starter script treats `Dropout` as the positive class and groups `Graduate` and `Enrolled` as the other class. The supplied features include first-semester information, so this is a baseline retrospective prediction setup—not proof that the system can identify risk at any earlier point in time. For a stronger research project, define the prediction date, use only information available by that date, prevent data leakage, evaluate on a later cohort or time-based split, and separately discuss false positives and false negatives.

The risk bands in the website are illustrative thresholds. Select thresholds with your advisor or faculty guide using the relative costs of false positives and false negatives. Review recall, precision, confusion matrix, calibration and subgroup performance before drawing conclusions.

## 3. Upload this project to GitHub

### Option A — Upload using the GitHub website (easiest)

1. Sign in at https://github.com.
2. Click **New repository** (or the `+` menu → **New repository**).
3. Name it `student-dropout-retention-intelligence`.
4. Add a short description, such as `Early detection of student dropout risk using machine learning and an advisor dashboard`.
5. Choose **Public** for a portfolio project or **Private** while you are still developing it.
6. Do not initialize another README if you plan to upload this complete folder.
7. Create the repository.
8. On the empty repository page, choose **Add file → Upload files**.
9. Open the downloaded project ZIP on your computer, extract it, and drag the *contents* of the `dropout-retention-intelligence` folder into the upload area. Upload folders such as `templates`, `static`, and `data` along with `app.py` and `train_model.py`.
10. Check the file list, enter a commit message such as `Add RetainIQ Flask project`, and click **Commit changes**.

Do **not** upload `.venv`, `instance/students.db`, `model.joblib`, dataset CSVs, private student records, passwords, API keys or any identifiable student data. The included `.gitignore` excludes local databases, model files and CSV datasets by default. Keep dataset files on your local machine unless your instructor has explicitly approved sharing them.

### Option B — Upload using Git commands

Create an empty GitHub repository first, then open a terminal inside this project folder and run:

```bash
git init
git add .
git commit -m "Add student dropout retention intelligence system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-dropout-retention-intelligence.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Run `git remote -v` to check the repository address if needed.

## 4. Deploy the website (optional)

A simple deployment option is Render:

1. Push your repository to GitHub.
2. Sign in to https://render.com and create a **New Web Service**.
3. Connect the GitHub repository.
4. Set **Build Command** to `pip install -r requirements.txt`.
5. Set **Start Command** to `gunicorn app:app`.
6. Deploy the service.

Before deploying publicly:

- Change `SECRET_KEY` in `app.py` to a long random secret (preferably load it from an environment variable).
- Do not use real student personal data in a public demo.
- Configure persistent storage or a managed database if records must survive redeployments. The included SQLite database is intended for local/demo use.
- Add authentication and role-based access before exposing student records to users.
- Confirm the hosting setup can retain any trained model file; otherwise use a secure model artifact store or deployment build step.

## Project structure

```text
dropout-retention-intelligence/
├── app.py                 # Flask routes, SQLite, dashboard and assessment logic
├── train_model.py         # Train and evaluate the Random Forest model
├── requirements.txt       # Python dependencies
├── README.md              # Setup and GitHub instructions
├── .gitignore             # Excludes local data and generated model files
├── data/
│   └── README.txt         # Dataset placement instructions
├── instance/              # Local SQLite database is created here
├── static/
│   ├── app.js             # Watchlist search and filters
│   └── style.css          # Responsive dashboard styling
└── templates/
    ├── index.html         # Main dashboard and assessment form
    └── student.html       # Individual student risk profile
```

## Suggested future improvements

- Add login, advisor roles and audit logs.
- Add consent, data minimisation and retention controls.
- Use SHAP or another validated explanation approach for model-level explanations.
- Add threshold tuning and subgroup fairness checks.
- Compare Logistic Regression, Random Forest and Gradient Boosting.
- Evaluate false-positive and false-negative costs with an advisor-reviewed intervention policy.
- Add a time-based validation split to test early-warning performance on a later student cohort.
