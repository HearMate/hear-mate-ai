import matplotlib.pyplot as plt
import joblib
from input_data import extract_features

hearing_loss_desc = {
    0: "0 (norma słyszenia)",
    1: "1 (lekki ubytek słuchu)",
    2: "2 (umiarkowany ubytek słuchu)",
    3: "3 (znaczny ubytek słuchu)",
    4: "4 (głęboki ubytek słuchu)",
}

categories = [
    "obustronny",
    "symetryczny",
    "stopień L",
    "stopień P",
    "wysokoczęstotliwościowy L",
    "wysokoczęstotliwościowy P",
    "niskoczęstotliwościowy L",
    "niskoczęstotliwościowy P",
]


def predict_sample(sample, model_path="model.pkl"):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    """
    try:
        features = extract_features(sample)
    except Exception as e:
        raise ValueError(f"Error extracting features from sample: {e}")
    """
    try:
        prediction = model.predict([sample])[0]
    except Exception as e:
        raise RuntimeError(f"Error making prediction: {e}")

    #rounded = [round(x) for x in prediction]

    print("Prediction:", prediction)
    """
    for label, val in zip(categories, rounded):
        if "stopień" in label:
            print(f"- {label}: {hearing_loss_desc.get(val, val)}")
        else:
            print(f"- {label}: {val}")
    """

def plot_audiogram(sample):
    frequencies = [125, 250, 500, 1000, 2000, 4000, 8000]
    x_ticks = list(range(len(frequencies)))
    left_ear = sample[:7]
    right_ear = sample[7:]

    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, left_ear, "ro-", label="Left Ear")
    plt.plot(x_ticks, right_ear, "bo-", label="Right Ear")
    plt.gca().invert_yaxis()

    plt.xticks(ticks=x_ticks, labels=[str(f) for f in frequencies])
    plt.yticks(range(0, 121, 10))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Hearing Threshold (dB HL)")
    plt.title("Audiogram")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
