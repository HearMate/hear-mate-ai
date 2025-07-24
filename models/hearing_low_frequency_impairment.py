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

MODEL_PATH = MODEL_DIRECTORY + "hearing_low_frequency_impairment.pkl"


class HearingLowFrequencyImpairmentClassifier:
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
            y_df_r = y_df.iloc[:, 7]

            X_df_l = x_df.iloc[:, 12:19]
            y_df_l = y_df.iloc[:, 8]

            y_r = y_df_r.dropna().to_numpy()
            X_r = X_df_r.loc[y_df_r.dropna().index].to_numpy()

            y_l = y_df_l.dropna().to_numpy()
            X_l = X_df_l.loc[y_df_l.dropna().index].to_numpy()

            # Additional feature engineering
            # 1. Add frequency averages
            X_r_avg = np.mean(X_r, axis=1).reshape(-1, 1)
            X_l_avg = np.mean(X_l, axis=1).reshape(-1, 1)

            # 2. Add variance of frequencies
            X_r_var = np.var(X_r, axis=1).reshape(-1, 1)
            X_l_var = np.var(X_l, axis=1).reshape(-1, 1)

            # 3. Add differences between adjacent frequencies
            X_r_diff = np.diff(X_r, axis=1)
            X_l_diff = np.diff(X_l, axis=1)

            X_r_ratios = X_r[:, 1:] / (
                X_r[:, :-1] + 1e-10
            )  # Add small epsilon to avoid division by zero
            X_l_ratios = X_l[:, 1:] / (X_l[:, :-1] + 1e-10)

            # 5. Add percentile-based features
            X_r_p25 = np.percentile(X_r, 25, axis=1).reshape(-1, 1)
            X_r_p75 = np.percentile(X_r, 75, axis=1).reshape(-1, 1)
            X_l_p25 = np.percentile(X_l, 25, axis=1).reshape(-1, 1)
            X_l_p75 = np.percentile(X_l, 75, axis=1).reshape(-1, 1)
            X_r_iqr = X_r_p75 - X_r_p25  # Interquartile range
            X_l_iqr = X_l_p75 - X_l_p25

            # 6. Add features capturing skewness
            X_r_skew = (
                (np.mean(X_r, axis=1) - np.median(X_r, axis=1))
                / (np.std(X_r, axis=1) + 1e-10)
            ).reshape(-1, 1)
            X_l_skew = (
                (np.mean(X_l, axis=1) - np.median(X_l, axis=1))
                / (np.std(X_l, axis=1) + 1e-10)
            ).reshape(-1, 1)

            # Update the feature stacking:
            X_r = np.hstack(
                [
                    X_r,
                    X_r_avg,
                    X_r_var,
                    X_r_diff,
                    X_r_ratios,
                    X_r_p25,
                    X_r_p75,
                    X_r_iqr,
                    X_r_skew,
                ]
            )
            X_l = np.hstack(
                [
                    X_l,
                    X_l_avg,
                    X_l_var,
                    X_l_diff,
                    X_l_ratios,
                    X_l_p25,
                    X_l_p75,
                    X_l_iqr,
                    X_l_skew,
                ]
            )

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
        print(f"[{MODEL_PATH} Accuracy (using score method): {accuracy_alt:.2f}%")

        # # Print confusion matrix
        # print("\nConfusion Matrix:")
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)

        # # Print classification report
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))

        try:
            joblib.dump(model, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model to {MODEL_PATH}: {e}")
            raise
        return model

    def predict(self, data):
        try:
            # Check if input data has the correct number of features before engineering
            if len(data) != 7:
                raise ValueError(
                    f"Input data should have 7 features, but got {len(data)}"
                )

            # Apply the same feature engineering as during training
            data_array = np.array(data).reshape(1, -1)

            # Add all the same engineered features as in training
            # 1. Add frequency average
            data_avg = np.mean(data_array, axis=1).reshape(-1, 1)
            # 2. Add frequency variance
            data_var = np.var(data_array, axis=1).reshape(-1, 1)
            # 3. Add differences between adjacent frequencies
            data_diff = np.diff(data_array, axis=1)
            # 4. Add ratios between adjacent bands
            data_ratios = data_array[:, 1:] / (data_array[:, :-1] + 1e-10)
            # 5. Add percentile features
            data_p25 = np.percentile(data_array, 25, axis=1).reshape(-1, 1)
            data_p75 = np.percentile(data_array, 75, axis=1).reshape(-1, 1)
            data_iqr = data_p75 - data_p25
            # 6. Add skewness
            data_skew = (
                (np.mean(data_array, axis=1) - np.median(data_array, axis=1))
                / (np.std(data_array, axis=1) + 1e-10)
            ).reshape(-1, 1)

            # Combine all features
            data_engineered = np.hstack(
                [
                    data_array,
                    data_avg,
                    data_var,
                    data_diff,
                    data_ratios,
                    data_p25,
                    data_p75,
                    data_iqr,
                    data_skew,
                ]
            )

            # Apply feature selection if used in training
            if hasattr(self, "feature_selector"):
                data_engineered = self.feature_selector.transform(data_engineered)

            # Make prediction with the engineered features
            return int(self.model.predict(data_engineered)[0])
        except Exception as e:
            raise RuntimeError(f"Error making prediction: {e}")
