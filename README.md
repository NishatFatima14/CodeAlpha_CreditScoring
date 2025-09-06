# CodeAlpha — Credit Scoring Model (Ready-to-Run)

**Task 1: Credit Scoring**.  
It trains multiple models to predict **Good vs Bad credit** and saves the best model for inference.

## ✅ Features
- Clean project structure
- Robust preprocessing (imputation, scaling, one-hot encoding)
- Models tried: Logistic Regression, Random Forest, (optional) XGBoost
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Saves: preprocessing pipeline + best model + metrics report
- Batch inference script for new applicants CSV

## 🧰 Tech Stack
Python, scikit-learn, pandas, numpy, xgboost (optional), joblib

## 📂 Structure
```
CodeAlpha_CreditScoring/
├── README.md
├── requirements.txt
├── data/
│   └── credit.csv
├── src/
│   ├── train.py
│   ├── infer.py
│   ├── report_utils.py
│   └── schema.json
└── artifacts/  (created by train.py; contains model + pipeline + report)
```

## 🚀 Quickstart
1) **Create and activate a venv** (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

> If `xgboost` fails to install on your system, don't worry—training will still run with Logistic Regression and Random Forest.

3) **Train the model**
```bash
python src/train.py --data data/credit.csv --target default --test-size 0.2 --random-state 42
```
Artifacts will appear in `artifacts/`:
- `preprocessor.joblib`
- `best_model.joblib`
- `metrics.json`
- `metrics_report.md`
- `label_encoder.joblib` (encodes target)

4) **Run batch inference on new applicants**
```bash
# Example using the same structure as credit.csv but without the target column
python src/infer.py --input data/credit.csv --output artifacts/predictions.csv
```
This will create `artifacts/predictions.csv` with `prob_bad_credit` and `pred_label` columns.

## 📝 Dataset
The provided `data/credit.csv` is **synthetic** but realistic and balanced.  
Columns include numeric, boolean, and categorical features commonly used in credit scoring.

## 🧪 Reproducibility
- Set `--random-state` in `train.py` for repeatable splits and training.
- All key versions are recorded in `metrics_report.md`.

## 📣 Submission Tips
- Push this repo to GitHub as `CodeAlpha_CreditScoring`.
- Record a short LinkedIn video demo: show dataset → train → metrics → inference.
- Mention: preprocessing, algorithms tried, metrics, and what you’d improve next.

Good luck—ship it! 🚀
