from models.hearing_loss import HearingLossClassifier


class AudiogramClassifier:
    def __init__(self, args):
        self.hearing_loss_classifier = HearingLossClassifier(args)

    def predict_for_ear(self, data):
        print(
            f"Hearing Loss Clasifier Output: {self.hearing_loss_classifier.predict(data)}"
        )
