"""
Training CLI: reads data/census.csv, trains model, saves artifacts, writes slice metrics.
"""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import (
    CATEGORICAL_FEATURES, TARGET, Artifacts, train_model,
    performance_on_categorical_slice, compute_model_metrics, inference
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/census.csv")
    ap.add_argument("--outdir", type=str, default="model")
    ap.add_argument("--slice-feature", type=str, default="education")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column '{TARGET}' in {args.data}")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(X_tr, y_tr, categorical_features=CATEGORICAL_FEATURES)
    Artifacts(model=model, categorical_features=CATEGORICAL_FEATURES).save(args.outdir)
    preds = inference(model, X_te)
    p, r, f1 = compute_model_metrics(y_te, preds)
    print(f"Test metrics — precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    df_te = X_te.copy(); df_te[TARGET] = y_te
    slices = performance_on_categorical_slice(model, df_te, feature=args.slice_feature)
    out_path = "slice_output.txt"
    with open(out_path, "w") as f:
        for val, (sp, sr, sf1) in slices.items():
            f.write(f"{args.slice_feature}={val} | precision={sp:.3f} recall={sr:.3f} f1={sf1:.3f}\n")
    print(f"Wrote slice metrics to {out_path}")

if __name__ == "__main__":
    main()
