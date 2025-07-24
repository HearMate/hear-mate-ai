import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import os
import numpy as np
from data.consts import *

MODEL_PATH = MODEL_DIRECTORY + "hearing_symmetry.pkl"


class HearingSymmetryClassifier:
    def __init__(self, args):
        self.args = args
        X, y = self._load_data()
        if os.path.exists(MODEL_PATH) and not args.force_train:
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = self._train_model(X, y)

    def _load_data(self):
        try:
            x_df = pd.read_csv(X_FILEPATH, sep=";")
            y_df = pd.read_csv(Y_FILEPATH, sep=";")

            if x_df.empty or y_df.empty:
                raise ValueError("One of the CSV files is empty")

            if len(x_df) != len(y_df):
                raise ValueError(
                    f"Mismatch in number of samples: X has {len(x_df)} rows, y has {len(y_df)} rows"
                )

            X_df_r = x_df.iloc[:, 4:11]
            y_df_r = y_df.iloc[:, 3]

            X_df_l = x_df.iloc[:, 12:19]
            y_df_l = y_df.iloc[:, 4]

            y_r = y_df_r.dropna().to_numpy()
            X_r = X_df_r.loc[y_df_r.dropna().index].to_numpy()

            y_l = y_df_l.dropna().to_numpy()
            X_l = X_df_l.loc[y_df_l.dropna().index].to_numpy()

            # Now stack them vertically
            X = np.vstack([X_r, X_l])
            y = np.concatenate([y_r, y_l])

            return X, y

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e.filename}")
        except pd.errors.EmptyDataError:
            raise ValueError("One of the CSV files is empty")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

    def _train_model(self, X, y):
        pass

    def predict(self, left_ear, right_ear):
        if len(left_ear) != 7 or len(right_ear) != 7:
            raise ValueError(
                "Input data must contain 7 values for each ear corresponding to hearing loss at frequencies 125, 250, 500, 1000, 2000, 4000, 8000 Hz."
            )

        left_ear = np.array(left_ear)
        right_ear = np.array(right_ear)
        diff = np.abs(left_ear - right_ear)

        # 1. Check for ≥20 dB HL at two contiguous frequencies
        count_20dB_contiguous = sum(
            diff[i] >= 20 and diff[i + 1] >= 20 for i in range(len(diff) - 1)
        )

        if count_20dB_contiguous > 0:
            return True

        # 2. Check for ≥15 dB HL at any two frequencies between 2000 Hz and 8000 Hz (indices 4,5,6)
        high_freq_indices = [4, 5, 6]
        high_freq_diffs = diff[high_freq_indices]
        count_15dB_high = np.sum(high_freq_diffs >= 15)

        if count_15dB_high >= 2:
            return True

        return False
