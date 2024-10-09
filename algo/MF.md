```python
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

class MF(BaseEstimator):
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
        self.P = np.random.randn(n_user, K)
        self.Q = np.random.randn(n_item, K)
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
        diff, tol = 1.0, self.tol
        n_users, n_items, n_obs = self.n_users, self.n_items, len(X)
        K, lam = self.K, self.lam

        if self.verbose:
            print('Fitting Reg-MF: K: %d, lam: %.5f' %(self.K, self.lam))

        self.index_item = [np.where(X[:,1] == i)[0] for i in range(n_items)]
        self.index_user = [np.where(X[:,0] == u)[0] for u in range(n_users)]

        for l in range(self.iterNum):
            obj_old = self.obj(X, y)
            
            for item_id in range(n_items):
                index_item_tmp = self.index_item[item_id]
                if len(index_item_tmp) == 0:
                  continue
                y_tmp = y[index_item_tmp]
                X_tmp = self.P[X[index_item_tmp,0]]
                clf = Ridge(alpha=lam*n_obs, fit_intercept=False)
                clf.fit(X=X_tmp, y=y_tmp)
                self.Q[item_id,:] = clf.coef_

            for user_id in range(n_users):
                index_user_tmp = self.index_user[user_id]
                if len(index_user_tmp) == 0:
                  continue
                y_tmp = y[index_user_tmp]
                X_tmp = self.Q[X[index_user_tmp,1]]
                clf = Ridge(alpha=lam*n_obs, fit_intercept=False)
                clf.fit(X=X_tmp, y=y_tmp)
                self.P[user_id,:] = clf.coef_
				
            obj_new = self.obj(X, y)
            diff = abs(obj_old - obj_new)

            rmse_tmp = self.mse(X, y)
            if self.verbose:
                print("RegMF-ALS: %d; obj: %.3f; rmse:%.3f, diff: %.3f"
                      %(l, obj_new, rmse_tmp, diff))

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
        pred_ratings : array, shape (n_samples,)
            A 1D array of shape `(n_samples,)`, where each element represents the predicted rating.
        """
        pred_rating = [self.P[pair_tmp[0],:].dot(self.Q[pair_tmp[1],:]) for pair_tmp in X]
        return np.array(pred_rating)

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
        pred_rating = self.predict(X)
        return np.mean( (pred_rating - y)**2 )

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

```



Here's an explanation of each method in the `MF` class:

**`__init__`**: This is the constructor method that initializes the object. It takes in several parameters:

* `n_users`: The number of users in the system.
* `n_items`: The number of items in the system.
* `lam`: The regularization parameter (default is 0.001).
* `K`: The number of latent factors (default is 10).
* `iterNum`: The number of iterations for the ALS algorithm (default is 10).
* `tol`: The tolerance for convergence (default is 1e-4).
* `verbose`: A flag to control the verbosity of the algorithm (default is 1).

The constructor initializes the `P` and `Q` matrices, which represent the user and item latent factors, respectively.

**`fit`**: This method fits the matrix factorization model to the given data. It takes in two parameters:

* `X`: A 2D array of shape `(n_samples, 2)`, where each row represents a user-item pair.
* `y`: A 1D array of shape `(n_samples,)`, where each element represents the rating associated with the corresponding user-item pair.

The method performs the following steps:

1. Initializes the `index_item` and `index_user` arrays, which store the indices of the user-item pairs for each item and user, respectively.
2. Performs the ALS algorithm for the specified number of iterations.
3. In each iteration, it updates the item and user latent factors using the current estimates of the other factor.
4. Computes the objective function value and checks for convergence.
5. Returns the fitted model.

**`predict`**: This method predicts the ratings for a given set of user-item pairs. It takes in a 2D array `X` of shape `(n_samples, 2)`, where each row represents a user-item pair. It returns a 1D array of shape `(n_samples,)`, where each element represents the predicted rating.

**`mse`**: This method computes the mean squared error (MSE) between the predicted ratings and the actual ratings. It takes in two parameters:

* `X`: A 2D array of shape `(n_samples, 2)`, where each row represents a user-item pair.
* `y`: A 1D array of shape `(n_samples,)`, where each element represents the actual rating.

It returns the MSE value.

**`obj`**: This method computes the objective function value, which is the sum of the MSE and the regularization term. It takes in two parameters:

* `X`: A 2D array of shape `(n_samples, 2)`, where each row represents a user-item pair.
* `y`: A 1D array of shape `(n_samples,)`, where each element represents the actual rating.

It returns the objective function value.

Note that the `fit` method is the main method that performs the matrix factorization, and the other methods are used to compute the predicted ratings, MSE, and objective function value.