from input_data import load_data, extract_features
from model_train import train_model
from predict_and_plot import predict_sample, plot_audiogram
import os
import argparse
import joblib


def main(args):
    model_path = "hearing_model.pkl"

    X_raw, y = load_data("data.csv")
    X = [extract_features(row) for row in X_raw]

    if args.force_train or not os.path.exists(model_path):
        print("Training model...")
        model = train_model(X, y, model_path=model_path)
    else:
        print("Loading model from file...")

    # TODO: Remove this, after making it available on server.
    sample = [45, 50, 55, 60, 65, 70, 75, 15, 20, 25, 20, 25, 30, 25]
    predict_sample(sample, model_path=model_path)
    plot_audiogram(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and run hearing loss model")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force retraining the model even if saved model exists",
    )
    args = parser.parse_args()

    main(args)
