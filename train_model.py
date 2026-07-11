"""Train the ML signal model offline, then save it for strategy.py / main.py to load.

This does NOT place any orders — it only downloads public Coinbase candles, builds
features + a forward-looking label, trains a classifier, and writes:
  - model.pkl        (the trained sklearn model)
  - model_meta.json  (feature list, horizon, label threshold, symbol, metrics)

Run:
    python train_model.py                  # defaults: BTC, 720 days of hourly bars
    python train_model.py --coin ETH --days 365

Re-run this periodically (e.g. weekly) to keep the model fresh with recent regime data.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import feed
import features

HERE = Path(__file__).parent

# Bars ahead to look when labeling ("did price rise meaningfully over the next N hours?").
HORIZON_BARS = 4
# Minimum forward return to count as a positive ("up") label; filters out pure noise moves.
LABEL_THRESHOLD = 0.0025

MODEL_PATH_TEMPLATE = str(HERE / "model_{coin}.pkl")
META_PATH_TEMPLATE = str(HERE / "model_meta_{coin}.json")


def build_dataset(coin, days):
    """Fetch history, compute features, and attach the forward-return label."""
    df = feed.recent_bars(coin=coin, days=days)
    if df.empty or len(df) < 300:
        raise SystemExit(f"Not enough history returned for {coin} ({len(df)} bars) — try more --days.")

    feat = features.compute_features(df)
    forward_return = feat["close"].shift(-HORIZON_BARS) / feat["close"] - 1
    feat["label"] = (forward_return > LABEL_THRESHOLD).astype(int)

    data = feat.dropna(subset=features.FEATURE_COLUMNS + ["label"]).reset_index(drop=True)
    return data


def train(coin="BTC", days=720):
    data = build_dataset(coin, days)
    x = data[features.FEATURE_COLUMNS].values
    y = data["label"].values

    # Chronological split — never shuffle time series data, or the test score is meaningless.
    split = int(len(data) * 0.85)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"{coin}: {len(data)} labeled bars  ->  train {len(x_train)} / test {len(x_test)}")
    print(f"positive label rate: train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    proba_test = model.predict_proba(x_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    report = classification_report(y_test, pred_test, digits=3)
    auc = roc_auc_score(y_test, proba_test) if len(set(y_test)) > 1 else float("nan")
    print(report)
    print(f"test AUC: {auc:.3f}")

    importances = sorted(zip(features.FEATURE_COLUMNS, model.feature_importances_),
                          key=lambda pair: pair[1], reverse=True)
    print("feature importances:")
    for name, score in importances:
        print(f"  {name:16s} {score:.3f}")

    model_path = Path(MODEL_PATH_TEMPLATE.format(coin=coin))
    meta_path = Path(META_PATH_TEMPLATE.format(coin=coin))

    joblib.dump(model, model_path)
    meta = {
        "coin": coin,
        "feature_columns": features.FEATURE_COLUMNS,
        "horizon_bars": HORIZON_BARS,
        "label_threshold": LABEL_THRESHOLD,
        "trained_at": int(time.time()),
        "test_auc": None if np.isnan(auc) else round(float(auc), 4),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {model_path.name} and {meta_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--days", type=int, default=720)
    args = parser.parse_args()
    train(args.coin, args.days)
