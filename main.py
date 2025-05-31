import os
import argparse
import joblib
from classifier import AudiogramClassifier


def main(args):
    classifier = AudiogramClassifier(args)
    sample = [0, 10, -5, 0, 10, 5, 10]
    sample1 = [20, 30, 15, 30, 25, 35, 30]
    sample2 = [40, 60, 75, 80, 60, 75, 80]
    sample3 = [80, 85, 90, 95, 100, 85, 75]

    classifier.predict_for_ear(sample)
    classifier.predict_for_ear(sample1)
    classifier.predict_for_ear(sample2)
    classifier.predict_for_ear(sample3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and run hearing loss model")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force retraining the model even if saved model exists",
    )
    args = parser.parse_args()

    main(args)
