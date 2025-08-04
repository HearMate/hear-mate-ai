from models.hearing_high_frequency_impairment import (
    HearingHighFrequencyImpairmentClassifier,
)
from models.hearing_low_frequency_impairment import (
    HearingLowFrequencyImpairmentClassifier,
)


class AudiogramClassifier:
    def __init__(self, args):
        self.hearing_high_frequency_impairment_classifier = (
            HearingHighFrequencyImpairmentClassifier(args)
        )
        self.hearing_low_frequency_impairment_classifier = (
            HearingLowFrequencyImpairmentClassifier(args)
        )

    def predict(self, left_ear, right_ear):
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

        return {
            "left_high": int(left_high_freq),
            "right_high": int(right_high_freq),
            "left_low": int(left_low_freq),
            "right_low": int(right_low_freq),
        }
