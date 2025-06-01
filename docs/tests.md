# Model Accuracy Improvements History

## Better Feature Engineering

Added new statistical features:
- Frequency ratios between adjacent bands
- Percentile-based features (25th, 75th, IQR)
- Variance and skewness metrics

## RandomForestClassifier Optimization

We conducted systematic hyperparameter tuning to improve our hearing loss classification model's accuracy from its baseline performance of 83%.

### Parameters Tested via Grid Search:

- **n_estimators**: [100, 200, 300, 500]
- **max_depth**: [5, 8, 10, 15, 20, None]
- **min_samples_split**: [2, 5, 8, 10]
- **min_samples_leaf**: [1, 2, 4]
- **max_features**: ["sqrt", "log2", None]
- **bootstrap**: [True, False]
- **class_weight**: ["balanced", "balanced_subsample", {0: 1, 1: 5, 2: 1, 3: 1}]

### Best Parameter Combination:
```python
{
    'bootstrap': True,
    'class_weight': 'balanced_subsample',
    'max_depth': 8,
    'max_features': None,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 100
}
```

## XGBoost Implementation

We further improved model performance with XGBoost, achieving 87% accuracy (↑4% from RandomForest baseline).

### Parameters Tested via RandomizedSearchCV:

- **n_estimators**: [100, 200, 300]
- **max_depth**: [3, 5, 7, 9]
- **learning_rate**: [0.01, 0.05, 0.1, 0.2]
- **subsample**: [0.6, 0.8, 1.0]
- **colsample_bytree**: [0.6, 0.8, 1.0]
- **min_child_weight**: [1, 3, 5]
- **gamma**: [0, 0.1, 0.3]
- **reg_alpha**: [0, 0.01, 0.1, 1]
- **reg_lambda**: [0.1, 1, 10]

### Best Parameter Combination:
```python
{
    'subsample': 0.6,
    'reg_lambda': 0.1,
    'reg_alpha': 0.01,
    'n_estimators': 300,
    'min_child_weight': 3,
    'max_depth': 3,
    'learning_rate': 0.01,
    'gamma': 0,
    'colsample_bytree': 0.6
}
```

## LightGBM Implementation

We transitioned to LightGBM but experienced a performance regression, achieving 78% accuracy (↓9% from XGBoost baseline). Despite improved infrastructure and feature engineering, the model underperformed compared to previous implementations.

### Parameters Tested via RandomizedSearchCV:

- **n_estimators**: [200, 300, 400]
- **learning_rate**: [0.05, 0.1, 0.15]
- **max_depth**: [5, 6, 7, 8]
- **num_leaves**: [31, 50, 70]
- **feature_fraction**: [0.7, 0.8, 0.9]
- **bagging_fraction**: [0.7, 0.8, 0.9]
- **min_child_samples**: [10, 20, 30]
- **reg_alpha**: [0, 0.1, 0.5]
- **reg_lambda**: [0, 0.1, 0.5]

### Best Parameter Combination:
```python
{
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_samples': 10,
    'n_estimators': 400,
    'num_leaves': 31,
    'reg_alpha': 0.1,
    'reg_lambda': 0
}
```

## Neural Network Implementation

We implemented a Multi-Layer Perceptron (MLP) classifier but encountered further performance degradation, achieving 71% accuracy (↓7% from LightGBM, ↓16% from XGBoost baseline). The neural network approach showed promise in terms of architecture but struggled with the specific characteristics of our hearing loss dataset.

### Default Configuration:
```python
{
    'hidden_layer_sizes': (128, 64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 1000,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'random_state': 42
}
```

### Parameters Tested via RandomizedSearchCV:

- **hidden_layer_sizes**: [(64,), (128,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]
- **activation**: ["relu", "tanh"]
- **solver**: ["adam", "lbfgs"]
- **alpha**: [0.0001, 0.001, 0.01, 0.1]
- **learning_rate**: ["constant", "adaptive"]
- **learning_rate_init**: [0.001, 0.01, 0.1]
- **max_iter**: [500, 1000, 2000]
- **early_stopping**: [True]
- **validation_fraction**: [0.1, 0.15, 0.2]

### Best Parameter Combination:
```python
{
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.01,
    'max_iter': 1000,
    'early_stopping': True,
    'validation_fraction': 0.1
}
```
