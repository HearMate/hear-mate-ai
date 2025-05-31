import pandas as pd
import numpy as np


def load_data(x_filepath, y_filepath, column):
    try:
        x_df = pd.read_csv(x_filepath, sep=';')
        y_df = pd.read_csv(y_filepath, sep=';')
        x_df = x_df.iloc[:, 4:19]
        x_df = x_df.drop(columns = ['HTL RE'])

        y = y_df.dropna().to_numpy()
        x = x_df.loc[y_df.dropna().index].to_numpy()

        return x, y
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {x_filepath, y_filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")


def load_data2(x_filepath, y_filepath):
    try:
        x_df = pd.read_csv(x_filepath, sep=';')
        y_df = pd.read_csv(y_filepath, sep=';')

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
