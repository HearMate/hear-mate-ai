import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

MODEL_PATH = "models/hearing_loss.pkl"


class HearingLossClassifier:
    def __init__(self, args):
        X, y = self._load_data()
        if os.path.exists(MODEL_PATH) and not args.force_train:
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = self._train_model(X, y)

    def _load_data(self):
        x_filepath = "data/xdata.csv"
        y_filepath = "data/ydata.csv"
        try:
            x_df = pd.read_csv(x_filepath, sep=";")
            y_df = pd.read_csv(y_filepath, sep=";")

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

            # Feature engineering
            # 1. Add frequency averages
            X_r_avg = np.mean(X_r, axis=1).reshape(-1, 1)
            X_l_avg = np.mean(X_l, axis=1).reshape(-1, 1)

            # 2. Add variance of frequencies
            X_r_var = np.var(X_r, axis=1).reshape(-1, 1)
            X_l_var = np.var(X_l, axis=1).reshape(-1, 1)

            # 3. Add differences between adjacent frequencies
            X_r_diff = np.diff(X_r, axis=1)
            X_l_diff = np.diff(X_l, axis=1)

            # Combine with original features
            X_r = np.hstack([X_r, X_r_avg, X_r_var, X_r_diff])
            X_l = np.hstack([X_l, X_l_avg, X_l_var, X_l_diff])

            # Now stack them vertically
            X = np.vstack([X_r, X_l])
            y = np.concatenate([y_r, y_l])

            unique_values, counts = np.unique(y, return_counts=True)
            print("Target values distribution:")
            for value, count in zip(unique_values, counts):
                print(f"Class {value}: {count} samples ({count/len(y)*100:.2f}%)")

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
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Use a more flexible parameter grid
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 8, 10, 15, 20, None],
            "min_samples_split": [2, 5, 8, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
            "class_weight": ["balanced", "balanced_subsample", None],
        }

        # Create a more sensitive model
        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            bootstrap=True,
            criterion="gini",  # Try "entropy" as an alternative
            oob_score=True,  # Enable out-of-bag evaluation
        )

        # Use a more appropriate scoring metric
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="balanced_accuracy",  # Better for imbalanced classes
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Make predictions with the best model
        y_pred = best_model.predict(X_test)

        # Display results
        print("\n" + "=" * 50)
        print("GRID SEARCH RESULTS")
        print("=" * 50)
        print(f"Best parameters: {grid_search.best_params_}")
        try:
            joblib.dump(best_model, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model to {MODEL_PATH}: {e}")
            raise
        return best_model

    def predict(self, data):
        try:
            # Check if input data has the correct number of features before engineering
            if len(data) != 7:
                raise ValueError(
                    f"Input data should have 7 features, but got {len(data)}"
                )

            # Apply the same feature engineering as during training
            data_array = np.array(data).reshape(1, -1)

            # 1. Add frequency average
            data_avg = np.mean(data_array, axis=1).reshape(-1, 1)

            # 2. Add frequency variance
            data_var = np.var(data_array, axis=1).reshape(-1, 1)

            # 3. Add differences between adjacent frequencies
            data_diff = np.diff(data_array, axis=1)

            # Combine with original features
            data_engineered = np.hstack([data_array, data_avg, data_var, data_diff])

            # Now make prediction with the engineered features
            return int(self.model.predict(data_engineered)[0])
        except Exception as e:
            raise RuntimeError(f"Error making prediction: {e}")
