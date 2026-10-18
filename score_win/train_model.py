"""
Train the Agent X win-probability model and save it to model/win_model.joblib.

By default this generates a synthetic dataset of 6,000 RFQs (the same size as the
project's sample data). To train on real history, export decided bids from your
CRM/ERP to a CSV with the columns in FEATURES plus a `won` column (1 = won, 0 = lost)
and run:

    python train_model.py --csv path/to/bids.csv

Run this in the SAME virtual environment you deploy with, so the scikit-learn
version that saves the model matches the one that loads it.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "markup",            # quoted price / total cost (e.g. 1.20 = 20% above cost)
    "lead_time_weeks",   # promised delivery lead time
    "ld_percent",        # liquidated damages cap, % of contract value
    "warranty_months",   # warranty period offered
    "clause_count",      # numbered requirement clauses in the RFQ
    "deviations",        # number of deviations / clarifications we take
    "repeat_customer",   # 1 if we have won work from this customer before
    "competitors",       # known or estimated number of competing bidders
]

# Win probability can only fall as price, lead time or deviations rise,
# and only rise for repeat customers. Monotonic constraints keep the price
# curve sensible, which matters because we search over price.
MONOTONIC = {
    "markup": -1,
    "lead_time_weeks": -1,
    "ld_percent": 0,
    "warranty_months": 1,
    "clause_count": 0,
    "deviations": -1,
    "repeat_customer": 1,
    "competitors": -1,
}

MODEL_DIR = Path(__file__).parent / "model"


def make_synthetic(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Synthetic ETO bid history: roughly 3 in 10 decided bids won."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "markup": rng.uniform(1.05, 1.45, n).round(3),
            "lead_time_weeks": rng.integers(8, 41, n),
            "ld_percent": rng.choice([5, 10, 15, 20], n),
            "warranty_months": rng.choice([12, 18, 24, 36], n),
            "clause_count": rng.integers(10, 28, n),
            "deviations": rng.poisson(2.0, n),
            "repeat_customer": rng.binomial(1, 0.35, n),
            "competitors": rng.integers(2, 7, n),
        }
    )
    logit = (
        1.3
        - 9.0 * (df.markup - 1.15)
        - 0.05 * (df.lead_time_weeks - 20)
        + 0.02 * (df.warranty_months - 18)
        - 0.25 * df.deviations
        + 0.9 * df.repeat_customer
        - 0.35 * df.competitors
    )
    p = 1 / (1 + np.exp(-logit))
    df["won"] = rng.binomial(1, p)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV of decided bids (FEATURES + won)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv) if args.csv else make_synthetic()
    missing = [c for c in FEATURES + ["won"] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")

    X, y = df[FEATURES], df["won"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        monotonic_cst=[MONOTONIC[f] for f in FEATURES],
        random_state=0,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    # Training range per feature: the function refuses to recommend outside it.
    ranges = {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES}

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "win_model.joblib")
    meta = {
        "features": FEATURES,
        "ranges": ranges,
        "rows": int(len(df)),
        "win_rate": round(float(y.mean()), 3),
        "test_auc": round(float(auc), 3),
        "test_brier": round(float(brier), 3),
        "source": args.csv or "synthetic",
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
