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