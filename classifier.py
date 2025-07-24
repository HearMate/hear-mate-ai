from models.hearing_loss import HearingLossClassifier
from models.hearing_symmetry import HearingSymmetryClassifier
from models.hearing_high_frequency_impairment import (
    HearingHighFrequencyImpairmentClassifier,
)
from models.hearing_low_frequency_impairment import (
    HearingLowFrequencyImpairmentClassifier,
)


class AudiogramClassifier:
    def __init__(self, args):
        self.hearing_loss_classifier = HearingLossClassifier(args)
        self.hearing_symmetry_classifier = HearingSymmetryClassifier(args)
        self.hearing_high_frequency_impairment_classifier = (
            HearingHighFrequencyImpairmentClassifier(args)
        )
        self.hearing_low_frequency_impairment_classifier = (
            HearingLowFrequencyImpairmentClassifier(args)
        )

    def predict(self, left_ear, right_ear):
        is_asymmetric = self.hearing_symmetry_classifier.predict(left_ear, right_ear)

        left_loss_class = self.hearing_loss_classifier.predict(left_ear)
        right_loss_class = self.hearing_loss_classifier.predict(right_ear)

        left_high_freq = self.hearing_high_frequency_impairment_classifier.predict(
            left_ear
        )
        right_high_freq = self.hearing_high_frequency_impairment_classifier.predict(
            right_ear
        )

        left_low_freq = self.hearing_low_frequency_impairment_classifier.predict(
            left_ear
        )
        right_low_freq = self.hearing_low_frequency_impairment_classifier.predict(
            right_ear
        )

        both_ears_impaired = left_loss_class != 0 and right_loss_class != 0

        print("=== Audiogram Classification Result ===")
        print(f"Symmetry: {is_asymmetric}")
        print(f"Both ears impaired: {both_ears_impaired}")
        print(
            f"Left Ear:  Class {left_loss_class} | High Freq Impairment: {left_high_freq} | Low Freq Impairment: {left_low_freq}"
        )
        print(
            f"Right Ear: Class {right_loss_class} | High Freq Impairment: {right_high_freq} | Low Freq Impairment: {right_low_freq}"
        )
        print("=======================================")
