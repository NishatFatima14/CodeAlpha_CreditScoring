import argparse, os, json, joblib, pandas as pd
import numpy as np

def main(args):
    # Load model & label encoder
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
    best_model_path = os.path.join(artifacts_dir, "best_model.joblib")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("best_model.joblib not found. Please run train.py first.")
    model = joblib.load(best_model_path)

    # Load input
    df = pd.read_csv(args.input)

    # Predict probabilities of class 1 (bad credit)
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    out = df.copy()
    out["prob_bad_credit"] = probs
    out["pred_label"] = np.where(preds == 1, "Bad", "Good")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to applicants CSV (same columns as training minus target)")
    parser.add_argument("--output", type=str, required=True, help="Where to write predictions CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()
    main(args)
