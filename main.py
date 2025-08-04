import os
import argparse
import joblib
from classifier import AudiogramClassifier
import json


def parse_ear_results(env_var_name):
    val = os.environ.get(env_var_name)
    if val is None:
        raise ValueError(f"Missing environment variable: {env_var_name}")
    # Remove brackets and split, then convert to floats
    val = val.strip("[]")
    return [float(x.strip()) for x in val.split(",") if x.strip()]


def main(args):
    left = parse_ear_results("LEFT_EAR_RESULTS")
    right = parse_ear_results("RIGHT_EAR_RESULTS")

    classifier = AudiogramClassifier(args)

    result = classifier.predict(left, right)

    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and run hearing loss model")
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Force retraining the model even if saved model exists",
    )

    parser.add_argument(
        "--search_for_params",
        action="store_true",
        help="Use grid search to get best parameters for the model",
    )

    args = parser.parse_args()

    main(args)
