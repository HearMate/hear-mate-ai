import os
import argparse
import joblib
from classifier import AudiogramClassifier


def main(args):
    classifier = AudiogramClassifier(args)
    sample = [0, 10, -5, 0, 10, 5, 10]
    sample1 = [20, 30, 15, 30, 25, 35, 30]

    classifier.predict(sample, sample1)


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
