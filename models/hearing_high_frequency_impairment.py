import joblib
from matplotlib import cm
import seaborn as sns
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

MODEL_PATH = MODEL_DIRECTORY + "hearing_high_frequency_impairment.pkl"


class HearingHighFrequencyImpairmentClassifier:
    def __init__(self, args=None):
        self.args = args
        if os.path.exists(MODEL_PATH) and (args is None or not args.force_train):
            self.model = joblib.load(MODEL_PATH)
        else:
            X, y = self._load_data()
            self.model = self._train_model(X, y)
            joblib.dump(self.model, MODEL_PATH)

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
            y_df_r = y_df.iloc[:, 5]

            X_df_l = x_df.iloc[:, 12:19]
            y_df_l = y_df.iloc[:, 6]

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
        if X is None or y is None:
            raise ValueError("Input features X and target y cannot be None")
        if len(X) != len(y):
            raise ValueError(
                "Features X and targets y must have the same number of samples"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        from collections import Counter

        counter = Counter(y_train)
        num_neg = counter[0]
        num_pos = counter[1]

        scale_pos_weight = num_neg / num_pos
        print("Scale_pos_weight =", scale_pos_weight)

        scale_pos_weight = scale_pos_weight * 0.6

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.3],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.1, 1, 10],
        }

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0.01,
            reg_lambda=0.1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )

        if self.args.search_for_params:
            grid_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=50,  # Number of parameter settings sampled
                cv=5,
                scoring="balanced_accuracy",  # Better for imbalanced classes
                n_jobs=-1,
                verbose=2,
                random_state=42,
            )

            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_

            model = best_model

            # Display results
            print("\n" + "=" * 50)
            print("GRID SEARCH RESULTS")
            print("=" * 50)
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy_alt = model.score(X_test, y_test) * 100
        print(f"{MODEL_PATH} Accuracy (using score method): {accuracy_alt:.2f}%")

        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Macierz błędów")
        plt.xlabel("Przewidziane")
        plt.ylabel("Prawdziwe")
        plt.show()

        try:
            joblib.dump(model, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model to {MODEL_PATH}: {e}")
            raise
        return model

    def predict(self, data):
        try:
            arr = np.asarray(data, dtype=float)

            # We need to handle training model & server post requests.

            if arr.ndim == 1:
                # Single flat sample
                if arr.shape[0] != 7:
                    raise ValueError(f"Expected 7 features, got {arr.shape[0]}")
                arr = arr.reshape(1, 7)
            elif arr.ndim == 2:
                # Batch or already single row
                if arr.shape == (7, 1):
                    # Column vector -> single row
                    arr = arr.reshape(1, 7)
                elif arr.shape[1] != 7:
                    raise ValueError(
                        f"Each sample must have 7 features, got shape {arr.shape}"
                    )
                # else: arr is (n,7) or (1,7) already; no reshape
            else:
                raise ValueError(f"Unsupported input dimensionality: {arr.ndim}")

            preds = self.model.predict(arr)
            if arr.shape[0] == 1:
                return int(preds[0])
            return [int(p) for p in preds]
        except Exception as e:
            raise RuntimeError(f"Error making prediction: {e}")
