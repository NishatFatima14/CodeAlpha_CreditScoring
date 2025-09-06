import argparse, json, os, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from report_utils import make_md_report

warnings.filterwarnings("ignore", category=UserWarning)

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_preprocessor(schema):
    numeric = schema["numeric"]
    categorical = schema["categorical"]
    binary = schema["binary"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
            ("bin", binary_transformer, binary)
        ]
    )
    return preprocessor

def try_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception:
        return None

def main(args):
    # Paths
    data_path = args.data
    target_col = args.target
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset. Found: {list(df.columns)}")
    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Load schema & build preprocessing
    schema = load_schema(schema_path)
    preprocessor = build_preprocessor(schema)

    # Encode labels (0=good, 1=bad or vice versa; normalize)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Candidate models
    models = []

    log_reg = LogisticRegression(max_iter=200, n_jobs=None)
    pipe_lr = Pipeline(steps=[("pre", preprocessor), ("clf", log_reg)])
    params_lr = {
        "clf__C": [0.1, 1.0, 2.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    models.append(("LogisticRegression", pipe_lr, params_lr))

    rf = RandomForestClassifier(random_state=args.random_state)
    pipe_rf = Pipeline(steps=[("pre", preprocessor), ("clf", rf)])
    params_rf = {
        "clf__n_estimators": [150, 300],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_split": [2, 5]
    }
    models.append(("RandomForest", pipe_rf, params_rf))

    XGBClassifier = try_import_xgb()
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            eval_metric="logloss",
            random_state=args.random_state,
            tree_method="hist"
        )
        pipe_xgb = Pipeline(steps=[("pre", preprocessor), ("clf", xgb)])
        params_xgb = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0]
        }
        models.append(("XGBoost", pipe_xgb, params_xgb))

    best = None
    best_name = None
    best_metrics = None

    for name, pipe, params in models:
        print(f"\\n>>> Tuning {name} ...")
        gs = GridSearchCV(pipe, params, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        y_prob = gs.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "model": name,
            "best_params": gs.best_params_,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob))
        }
        print(metrics)

        if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
            best = gs.best_estimator_
            best_name = name
            best_metrics = metrics

    # Save artifacts
    joblib.dump(best, os.path.join(out_dir, "best_model.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))

    # Save preprocessor separately for visibility (it's already inside the pipeline though)
    # Extract and save the preprocessor from the pipeline
    pre = best.named_steps.get("pre", None)
    if pre is not None:
        joblib.dump(pre, os.path.join(out_dir, "preprocessor.joblib"))

    # Save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    # Save report
    extras = {"Best Model": best_name, "Data": os.path.basename(data_path)}
    report_md = make_md_report(best_metrics, extras)
    with open(os.path.join(out_dir, "metrics_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\\nSaved artifacts to: {out_dir}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV")
    parser.add_argument("--target", type=str, default="default")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)
