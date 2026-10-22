import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


class GlobalMeanRS(RegressorMixin, BaseEstimator):
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items

    def fit(self, X, y):
        self.global_mean_ = np.asarray(y).mean()
        return self

    def predict(self, X):
        check_is_fitted(self, 'global_mean_')
        return np.full(len(X), self.global_mean_)


class UserMeanRS(RegressorMixin, BaseEstimator):
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        users = X[:, 0].astype(int)

        self.global_mean_ = y.mean()
        self.user_means_ = np.full(
            self.n_users, self.global_mean_
        )

        for user in set(users):
            ratings = y[users == user]
            self.user_means_[user] = ratings.mean()

        return self

    def predict(self, X):
        check_is_fitted(
            self, ['global_mean_', 'user_means_']
        )
        users = np.asarray(X)[:, 0].astype(int)

        predictions = np.full(len(users), self.global_mean_)
        valid = (users >= 0) & (users < len(self.user_means_))
        predictions[valid] = self.user_means_[users[valid]]
        return predictions

class ItemMeanRS(RegressorMixin, BaseEstimator):
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        items = X[:, 1].astype(int)

        self.global_mean_ = y.mean()
        self.item_means_ = np.full(
            self.n_items, self.global_mean_
        )

        for item in set(items):
            ratings = y[items == item]
            self.item_means_[item] = ratings.mean()

        return self

    def predict(self, X):
        check_is_fitted(
            self, ['global_mean_', 'item_means_']
        )
        items = np.asarray(X)[:, 1].astype(int)

        predictions = np.full(len(items), self.global_mean_)
        valid = (items >= 0) & (items < len(self.item_means_))
        predictions[valid] = self.item_means_[items[valid]]
        return predictions

class BiasedMF(RegressorMixin, BaseEstimator):
    def __init__(self, n_users, n_items, K=2, lambda_=1.0, n_iters=3, random_state=42, verbose=False):
        # Dataset dimensions and settings are available to GridSearchCV.
        self.n_users = n_users
        self.n_items = n_items
        self.K = K
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError('X must have two ID columns and match a nonempty y.')
        if np.any(X < 0) or np.any(X[:, 0] >= self.n_users) or np.any(X[:, 1] >= self.n_items):
            raise ValueError('User and item IDs must lie within the supplied dimensions.')
        if self.K < 1 or self.lambda_ < 0 or self.n_iters < 1:
            raise ValueError('Use K >= 1, lambda_ >= 0, and n_iters >= 1.')

        users, items = X[:, 0], X[:, 1]
        n_users, n_items = self.n_users, self.n_items
        self.mu_ = y.mean()  # recomputed from this fit's rows, including in CV
        self.a_ = np.zeros(n_users)
        self.b_ = np.zeros(n_items)
        rng = np.random.default_rng(self.random_state)
        self.P_ = rng.normal(0, 0.1, size=(n_users, self.K))
        self.Q_ = rng.normal(0, 0.1, size=(n_items, self.K))

        rows_by_user = [[] for _ in range(n_users)]
        rows_by_item = [[] for _ in range(n_items)]
        for row, (u, i) in enumerate(X):
            rows_by_user[u].append(row)
            rows_by_item[i].append(row)

        self.seen_users_ = np.array([bool(rows) for rows in rows_by_user])
        self.seen_items_ = np.array([bool(rows) for rows in rows_by_item])
        # Unseen IDs keep zero bias and factor, so the one prediction formula still works.
        self.P_[~self.seen_users_] = 0.0
        self.Q_[~self.seen_items_] = 0.0
        self.objective_history_ = [self.objective(X, y)]
        if self.verbose:
            print(f'ALS iteration 0: objective = {self.objective_history_[-1]:.4f}')

        for iteration in range(1, self.n_iters + 1):
            # Fix users, update each observed item: intercept = b_i, coefficients = q_i.
            for i, rows in enumerate(rows_by_item):
                if not rows:
                    continue
                user_ids = users[rows]
                features = self.P_[user_ids]
                targets = y[rows] - self.mu_ - self.a_[user_ids]
                ridge = Ridge(alpha=self.lambda_, fit_intercept=True)
                ridge.fit(features, targets)
                self.b_[i] = ridge.intercept_
                self.Q_[i] = ridge.coef_

            # Fix items, update each observed user: intercept = a_u, coefficients = p_u.
            for u, rows in enumerate(rows_by_user):
                if not rows:
                    continue
                item_ids = items[rows]
                features = self.Q_[item_ids]
                targets = y[rows] - self.mu_ - self.b_[item_ids]
                ridge = Ridge(alpha=self.lambda_, fit_intercept=True)
                ridge.fit(features, targets)
                self.a_[u] = ridge.intercept_
                self.P_[u] = ridge.coef_

            objective = self.objective(X, y)
            self.objective_history_.append(objective)
            if self.verbose:
                print(f'ALS iteration {iteration}: objective = {objective:.4f}')
        return self

    def predict(self, X):
        check_is_fitted(self, 'mu_')
        X = np.asarray(X, dtype=int)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError('X must have two columns: [user ID, item ID].')
        if np.any(X < 0) or np.any(X[:, 0] >= self.n_users) or np.any(X[:, 1] >= self.n_items):
            raise ValueError('User and item IDs must lie within the supplied dimensions.')
        users, items = X[:, 0], X[:, 1]
        return (
            self.mu_ + self.a_[users] + self.b_[items]
            + np.sum(self.P_[users] * self.Q_[items], axis=1)
        )

    def objective(self, X, y):
        # Same sum-loss convention as the slides; only fitted rating rows enter.
        residuals = np.asarray(y, dtype=float) - self.predict(X)
        penalty = (
            np.sum(self.P_ ** 2) + np.sum(self.Q_ ** 2)
        )
        return np.sum(residuals ** 2) + self.lambda_ * penalty
