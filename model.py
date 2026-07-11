"""Loads the model trained by train_model.py and exposes a single scoring call
for live inference."""

import json
from pathlib import Path

import joblib

import features

HERE = Path(__file__).parent


class ModelNotTrained(Exception):
    """Raised when model_<coin>.pkl / model_meta_<coin>.json aren't present yet."""


def load(coin="BTC"):
    """Load the trained model + its metadata for `coin`. Raises ModelNotTrained if missing."""
    model_path = HERE / f"model_{coin}.pkl"
    meta_path = HERE / f"model_meta_{coin}.json"
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotTrained(
            f"{model_path.name} / {meta_path.name} not found — "
            f"run `python train_model.py --coin {coin}` first."
        )
    clf = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return clf, meta


def predict_up_probability(clf, df):
    """Return (probability_of_up_move, is_valid) for the latest bar in df.
    is_valid is False if there isn't enough warmup history for the features yet."""
    row, valid = features.latest_feature_row(df)
    if not valid:
        return 0.5, False
    proba = float(clf.predict_proba(row)[0, 1])
    return proba, True
