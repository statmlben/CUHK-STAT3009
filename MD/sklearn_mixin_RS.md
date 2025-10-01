Making RS codes as sklearn estimators:

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class UserMeanRS(BaseEstimator, RegressorMixin):
    def __init__(self, n_users, min_data=3):
        self.n_users = n_users
        self.global_mean_ = 0
        self.min_data = min_data
        self.user_means_ = np.zeros(n_users)

    def fit(self, X, y):
        self.global_mean_ = np.mean(y)
        for user in range(self.n_users):
            user_indices = np.where(X[:, 0] == user)[0]
            if len(user_indices) <= self.min_data:
                self.user_means_[user] = self.global_mean_
            else:
                self.user_means_[user] = np.mean(y[user_indices])
        return self

    def predict(self, X):
        user_indices = X[:, 0]
        return self.user_means_[user_indices]

class ItemMeanRS(BaseEstimator, RegressorMixin):
    def __init__(self, n_items, min_data=3):
        self.n_items = n_items
        self.global_mean_ = 0
        self.min_data = min_data
        self.item_means_ = np.zeros(n_items)

    def fit(self, X, y):
        self.global_mean_ = np.mean(y)
        for item in range(self.n_items):
            item_indices = np.where(X[:, 1] == item)[0]
            if len(item_indices) <= self.min_data:
                self.item_means_[item] = self.global_mean_
            else:
                self.item_means_[item] = np.mean(y[item_indices])
        return self

    def predict(self, X):
        item_indices = X[:, 1]
        return self.item_means_[item_indices]
```



