import os
import argparse
import threading
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

from models.hearing_high_frequency_impairment import (
    HearingHighFrequencyImpairmentClassifier,
)
from models.hearing_low_frequency_impairment import (
    HearingLowFrequencyImpairmentClassifier,
)

EXPECTED_FREQ_COUNT = 7  # 125, 250, 500, 1000, 2000, 4000, 8000 Hz

# ---------------------------
# Arg configuration
# ---------------------------
parser = argparse.ArgumentParser(
    description="Hearing impairment model server (no wrapper)"
)
parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))

args, _ = parser.parse_known_args()

# ---------------------------
# Flask app & model singletons
# ---------------------------
app = Flask(__name__)

_high_model = None
_low_model = None
_model_lock = threading.Lock()


def get_models():
    """
    Lazily instantiate (or load) both classifiers once.
    Returns (high_model, low_model).
    """
    global _high_model, _low_model
    if _high_model is None or _low_model is None:
        with _model_lock:
            if _high_model is None:
                _high_model = HearingHighFrequencyImpairmentClassifier()
            if _low_model is None:
                _low_model = HearingLowFrequencyImpairmentClassifier()
    return _high_model, _low_model


# ---------------------------
# Helpers
# ---------------------------
def _validate_ear_array(arr, name):
    if not isinstance(arr, list):
        raise ValueError(f"'{name}' must be a list.")
    if len(arr) != EXPECTED_FREQ_COUNT:
        raise ValueError(
            f"'{name}' must have {EXPECTED_FREQ_COUNT} values (125..8000 Hz). Got {len(arr)}."
        )
    try:
        floats = [float(x) for x in arr]
    except Exception as e:
        raise ValueError(f"'{name}' contains non-numeric values: {e}")
    return floats


def _boolean_response(flag: int | bool):
    return jsonify({"result": bool(flag)})


# ---------------------------
# Error handling
# ---------------------------
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify(error=str(e)), e.code
    return jsonify(error=str(e)), 500


# ---------------------------
# Routes
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200


@app.route("/hearing/high-frequency-loss", methods=["POST"])
def high_frequency_loss():
    """
    Body JSON:
      { "earResults": [7 floats] }

    Returns:
      { "result": true/false }
    """
    payload = request.get_json(force=True, silent=False)
    if payload is None or "earResults" not in payload:
        return jsonify(error="JSON must include 'earResults'"), 400

    ear = _validate_ear_array(payload["earResults"], "earResults")
    high_model, _ = get_models()
    result = high_model.predict(ear)
    return _boolean_response(result)


@app.route("/hearing/low-frequency-loss", methods=["POST"])
def low_frequency_loss():
    """
    Body JSON:
      { "earResults": [7 floats] }

    Returns:
      { "result": true/false }
    """
    payload = request.get_json(force=True, silent=False)
    if payload is None or "earResults" not in payload:
        return jsonify(error="JSON must include 'earResults'"), 400

    ear = _validate_ear_array(payload["earResults"], "earResults")
    _, low_model = get_models()
    result = low_model.predict(ear)
    return _boolean_response(result)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    get_models()
    app.run(host=args.host, port=args.port, debug=False)
