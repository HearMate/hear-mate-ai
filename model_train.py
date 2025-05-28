import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def train_model(X, y, model_path="model.pkl"):
    if X is None or y is None:
        raise ValueError("Input features X and target y cannot be None")
    if len(X) != len(y):
        raise ValueError(
            "Features X and targets y must have the same number of samples"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # TODO: Test other depth levels.
    model = MultiOutputRegressor(DecisionTreeRegressor(max_depth=3))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.3f}")

    try:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")
        raise
    return model
