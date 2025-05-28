import pandas as pd
import numpy as np


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] < 14:
            raise ValueError(f"CSV must have at least 14 columns, found {df.shape[1]}")
        X_raw = df.iloc[:, 0:14].values
        y = df.iloc[:, 14:].values
        return X_raw, y
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")


"""
Decision trees benefit from interpretable, numerical, and diverse features. A tree splits data by testing one feature at a time. So the more relevant the features are to the task (e.g., detecting patterns in hearing loss), the better the model performs.
"""


def extract_features(audiogram):
    if len(audiogram) != 14:
        raise ValueError(
            f"Audiogram must have exactly 14 values (7 per ear), got {len(audiogram)}"
        )

    left = np.array(audiogram[:7])
    right = np.array(audiogram[7:])
    diff = np.abs(left - right)
    return [
        np.mean(left),
        np.mean(right),
        np.max(left),
        np.max(right),
        np.mean(diff),
        np.max(diff),
        np.std(left),
        np.std(right),
        np.min(left),
        np.min(right),
        np.percentile(left, 75),
        np.percentile(right, 75),
    ]
