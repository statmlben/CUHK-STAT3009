import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

class GlobalMeanRS(BaseEstimator):
    def __init__(self):
        self.global_mean_ = 0

    def fit(self, X, y):
        self.global_mean_ = np.mean(y)
        return self

    def predict(self, X):
        return self.global_mean_*np.ones(len(X))

class UserMeanRS(BaseEstimator):
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

class ItemMeanRS(BaseEstimator):
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

class SVD(BaseEstimator):
    """
    Matrix Factorization (MF) class for collaborative filtering.

    Parameters
    ----------
    n_users : int
        The number of users in the system.
    n_items : int
        The number of items in the system.
    lam : float, optional (default=0.001)
        The regularization parameter.
    K : int, optional (default=10)
        The number of latent factors.
    iterNum : int, optional (default=10)
        The number of iterations for the ALS algorithm.
    tol : float, optional (default=1e-4)
        The tolerance for convergence.
    verbose : int, optional (default=1)
        A flag to control the verbosity of the algorithm.

    Attributes
    ----------
    mu: float
        The global effect of the ratings.
    a : array, shape (n_users)
        The user bias term.
    b : array, shape (n_items)
        The item bias term.
    P : array, shape (n_users, K)
        The user latent factors.
    Q : array, shape (n_items, K)
        The item latent factors.

    Methods
    -------
    fit(X, y)
        Fits the matrix factorization model to the given data.
    predict(X)
        Predicts the ratings for a given set of user-item pairs.
    mse(X, y)
        Computes the mean squared error (MSE) between the predicted ratings and the actual ratings.
    obj(X, y)
        Computes the objective function value, which is the sum of the MSE and the regularization term.

    Notes
    -----
    This implementation uses the Alternating Least Squares (ALS) algorithm to learn the latent factors.
    """

    def __init__(self, n_users, n_items, lam=.001, K=10, iterNum=10, tol=1e-4, verbose=1):
        self.mu = 0.0
        self.a = np.zeros(n_users)
        self.b = np.zeros(n_items)
        self.P = np.random.randn(n_users, K)
        self.Q = np.random.randn(n_items, K)
        self.n_users = n_users
        self.n_items = n_items
        self.K = K
        self.lam = lam
        self.iterNum = iterNum
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the matrix factorization model to the given data.

        Parameters
        ----------
        X : array, shape (n_samples, 2)
            A 2D array of shape `(n_samples, 2)`, where each row represents a user-item pair.
        y : array, shape (n_samples,)
            A 1D array of shape `(n_samples,)`, where each element represents the actual rating.

        Returns
        -------
        self
            The fitted model.
        """
        diff = 1.0
        n_users, n_items, n_obs = self.n_users, self.n_items, len(X)
        K, lam = self.K, self.lam

        if self.verbose:
            print('Fitting Reg-SVD: K: %d, lam: %.5f' %(self.K, self.lam))

        self.index_item = [np.where(X[:,1] == i)[0] for i in range(n_items)]
        self.index_user = [np.where(X[:,0] == u)[0] for u in range(n_users)]

        for l in range(self.iterNum):
            obj_old = self.obj(X, y)

            ## update global bias term
            self.mu = np.mean(y - self.predict(X) + self.mu)
            
            ## update item params
            for item_id in range(n_items):
                index_item_tmp = self.index_item[item_id]
                if len(index_item_tmp) == 0:
                    self.Q[item_id,:] = 0.0
                    continue
                ## update item bias term
                y_tmp = y[index_item_tmp]
                X_tmp = X[index_item_tmp]
                U_tmp = X_tmp[:,0]
                self.b[item_id] = np.mean(y_tmp - self.predict(X_tmp) + self.b[item_id])
                
                ## update item latent factors
                res_tmp = y_tmp - self.mu - self.b[item_id] - self.a[U_tmp]
                P_tmp = self.P[U_tmp]
                clf = Ridge(alpha=lam*n_obs, fit_intercept=False)
                clf.fit(X=P_tmp, y=res_tmp)
                self.Q[item_id,:] = clf.coef_

            ## update user params
            for user_id in range(n_users):
                index_user_tmp = self.index_user[user_id]
                if len(index_user_tmp) == 0:
                    self.P[user_id,:] = 0.0
                    continue
                ## update item bias term
                y_tmp = y[index_user_tmp]
                X_tmp = X[index_user_tmp]
                I_tmp = X_tmp[:,1]
                self.a[user_id] = np.mean(y_tmp - self.predict(X_tmp) + self.a[user_id])
                
                ## update user latent factors
                res_tmp = y_tmp - self.mu - self.b[I_tmp] - self.a[user_id]
                Q_tmp = self.Q[I_tmp]
                clf = Ridge(alpha=lam*n_obs, fit_intercept=False)
                clf.fit(X=Q_tmp, y=res_tmp)
                self.P[user_id,:] = clf.coef_
				
            obj_new = self.obj(X, y)
            diff = abs(obj_old - obj_new)

            rmse_tmp = self.mse(X, y)
            if self.verbose:
                print("RegSVD-ALS: %d; obj: %.3f; rmse:%.3f, diff: %.3f"
                      %(l, obj_new, rmse_tmp, diff))
            
            if diff < self.tol:
                break

        return self

    def predict(self, X):
        """
        Predicts the ratings for a given set of user-item pairs.

        Parameters
        ----------
        X : array, shape (n_samples, 2)
            A 2D array of shape `(n_samples, 2)`, where each row represents a user-item pair.

        Returns
        -------
        pred_y : array, shape (n_samples,)
            A 1D array of shape `(n_samples,)`, where each element represents the predicted rating.
        """
        pred_y = [self.mu + self.a[X_tmp[0]] + self.b[X_tmp[1]] + np.dot(self.P[X_tmp[0]], self.Q[X_tmp[1]]) for X_tmp in X]
        # pred_y = self.mu + self.a[X[:,0]] + self.b[X[:,1]] + self.P[X[:,0]].dot(self.Q[X[:,1]].T)
        return np.array(pred_y)

    def mse(self, X, y):
        """
        Computes the mean squared error (MSE) between the predicted ratings and the actual ratings.

        Parameters
        ----------
        X : array, shape (n_samples, 2)
            A 2D array of shape `(n_samples, 2)`, where each row represents a user-item pair.
        y : array, shape (n_samples,)
            A 1D array of shape `(n_samples,)`, where each element represents the actual rating.

        Returns
        -------
        mse : float
            The mean squared error.
        """
        pred_y = self.predict(X)
        return np.mean( (pred_y - y)**2 )

    def obj(self, X, y):
        """
        Computes the objective function value, which is the sum of the MSE and the regularization term.

        Parameters
        ----------
        X : array, shape (n_samples, 2)
            A 2D array of shape `(n_samples, 2)`, where each row represents a user-item pair.
        y : array, shape (n_samples,)
            A 1D array of shape `(n_samples,)`, where each element represents the actual rating.

        Returns
        -------
        obj : float
            The objective function value.
        """
        mse_tmp = self.mse(X, y)
        pen_tmp = np.sum( (self.P)**2 ) + np.sum( (self.Q)**2 )
        return mse_tmp + self.lam*pen_tmp
