**What is `BaseEstimator`?**

`BaseEstimator` is a base class in scikit-learn (sklearn) that provides a common interface for all estimators. An estimator is an object that learns from data and makes predictions or estimates. Examples of estimators include classifiers, regressors, clustering algorithms, and transformers.

**Why do we need `BaseEstimator`?**

By inheriting from `BaseEstimator`, you can create your own custom estimators that are compatible with sklearn's API. This allows you to:

1. Use your custom estimator with sklearn's utilities, such as `GridSearchCV` for hyperparameter tuning.
2. Take advantage of sklearn's pipeline functionality, which enables you to chain multiple estimators together.
3. Leverage sklearn's built-in support for common tasks, such as data preprocessing and feature selection.

**Creating a custom estimator**

To create a custom estimator, you need to:

1. Import `BaseEstimator` from sklearn.
2. Define a class that inherits from `BaseEstimator`.
3. Implement the required methods: `__init__`, `fit`, and `predict` (or `transform` for transformers).

Here's an example:
```python
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class MyLinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, X, y):
        # Learn from data
        self.coef_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

    def predict(self, X):
        # Make predictions
        return X.dot(self.coef_)
```
In this example, we define a simple linear regressor that learns from data using ordinary least squares (OLS). The `__init__` method initializes the estimator with a regularization parameter `alpha`. The `fit` method learns the coefficients from the data, and the `predict` method makes predictions using these coefficients.

**Using the custom estimator**

Now, you can use your custom estimator with sklearn's utilities:
```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of our custom estimator
my_regressor = MyLinearRegressor(alpha=0.1)

# Train the estimator
my_regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = my_regressor.predict(X_test)

# Evaluate the performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
```
In this example, we load the Boston housing dataset, split it into training and testing sets, and train our custom estimator on the training set. We then make predictions on the testing set and evaluate the performance using the mean squared error metric.

That's it! You've now created a custom estimator that's compatible with sklearn's API.