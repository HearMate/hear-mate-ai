import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def train_model(X, y, model_path="model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TODO: Test other depth levels.
    model = MultiOutputRegressor(DecisionTreeRegressor(max_depth=3))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.3f}")

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model
