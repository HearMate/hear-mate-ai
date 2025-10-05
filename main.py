import os
import argparse
import json

from models.hearing_high_frequency_impairment import (
    HearingHighFrequencyImpairmentClassifier,
)
from models.hearing_low_frequency_impairment import (
    HearingLowFrequencyImpairmentClassifier,
)


def main(args):
    hearing_high_frequency_impairment_classifier = (
        HearingHighFrequencyImpairmentClassifier(args)
    )
    hearing_low_frequency_impairment_classifier = (
        HearingLowFrequencyImpairmentClassifier(args)
    )


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
