"""
Agent X - score_win Azure Function (Python v2 programming model).

POST /api/score_win
    Scores a range of candidate prices for one RFQ and recommends the price with
    the highest expected margin = P(win) x (price - total_cost), searching only
    inside the markup range the model was trained on.

GET /api/health
    Returns model metadata so you can check the deployment is alive.
"""

import json
import logging
from pathlib import Path

import azure.functions as func
import joblib
import numpy as np
import pandas as pd

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

MODEL_DIR = Path(__file__).parent / "model"
MODEL = joblib.load(MODEL_DIR / "win_model.joblib")
META = json.loads((MODEL_DIR / "model_meta.json").read_text())
FEATURES = META["features"]
RANGES = META["ranges"]

# Features the caller supplies (everything except markup, which we search over).
RFQ_FIELDS = [f for f in FEATURES if f != "markup"]
DEFAULT_STEP = 0.01
MAX_POINTS = 500


def _error(message: str, status: int = 400) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"error": message}), status_code=status, mimetype="application/json"
    )


def _parse(body: dict) -> tuple[float, dict, list[float]]:
    """Validate the request and return (total_cost, rfq_features, markups)."""
    try:
        total_cost = float(body["total_cost"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("total_cost (number, from the estimator) is required")
    if total_cost <= 0:
        raise ValueError("total_cost must be greater than 0")

    rfq = {}
    for field in RFQ_FIELDS:
        if field not in body:
            raise ValueError(f"missing field: {field}")
        try:
            rfq[field] = float(body[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a number")

    lo, hi = RANGES["markup"]
    if "candidate_prices" in body:
        prices = body["candidate_prices"]
        if not isinstance(prices, list) or not prices:
            raise ValueError("candidate_prices must be a non-empty list of numbers")
        if len(prices) > MAX_POINTS:
            raise ValueError(f"at most {MAX_POINTS} candidate_prices")
        markups = [float(p) / total_cost for p in prices]
    else:
        m_min = float(body.get("markup_min", lo))
        m_max = float(body.get("markup_max", hi))
        step = float(body.get("step", DEFAULT_STEP))
        if step <= 0 or m_max < m_min:
            raise ValueError("need markup_min <= markup_max and step > 0")
        markups = [float(m) for m in np.round(np.arange(m_min, m_max + step / 2, step), 4)]
        if len(markups) > MAX_POINTS:
            raise ValueError(f"range gives more than {MAX_POINTS} points; raise step")
    return total_cost, rfq, markups


def score(total_cost: float, rfq: dict, markups: list[float]) -> dict:
    lo, hi = RANGES["markup"]
    rows = pd.DataFrame([{**rfq, "markup": m} for m in markups])[FEATURES]
    probs = MODEL.predict_proba(rows)[:, 1]

    curve = []
    for m, p in zip(markups, probs):
        price = total_cost * m
        in_range = bool(lo <= m <= hi)
        curve.append(
            {
                "price": round(price, 2),
                "markup": round(m, 4),
                "win_probability": round(float(p), 4),
                "expected_margin": round(float(p) * (price - total_cost), 2),
                "in_training_range": in_range,
            }
        )

    eligible = [c for c in curve if c["in_training_range"] and c["markup"] > 1]
    recommended = max(eligible, key=lambda c: c["expected_margin"]) if eligible else None

    warnings = []
    for field, value in rfq.items():
        f_lo, f_hi = RANGES[field]
        if not f_lo <= value <= f_hi:
            warnings.append(
                f"{field}={value:g} is outside the training range [{f_lo:g}, {f_hi:g}]; "
                "treat the scores with caution"
            )
    if recommended is None:
        warnings.append(
            f"no candidate price is inside the trained markup range "
            f"[{lo:g}, {hi:g}] and above cost; no recommendation made"
        )

    return {
        "recommended": recommended,
        "curve": curve,
        "warnings": warnings,
        "model": {k: META[k] for k in ("source", "rows", "test_auc", "win_rate")},
        "note": "Recommendation only. Engineering and sales approve the final price.",
    }


@app.route(route="score_win", methods=["POST"])
def score_win(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return _error("request body must be JSON")
    if not isinstance(body, dict):
        return _error("request body must be a JSON object")

    try:
        total_cost, rfq, markups = _parse(body)
    except ValueError as exc:
        return _error(str(exc))

    try:
        result = score(total_cost, rfq, markups)
    except Exception:
        logging.exception("scoring failed")
        return _error("scoring failed", status=500)

    logging.info(
        "score_win: %d prices scored, recommended=%s",
        len(markups),
        result["recommended"] and result["recommended"]["price"],
    )
    return func.HttpResponse(json.dumps(result), mimetype="application/json")


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"status": "ok", "model": META}), mimetype="application/json"
    )
