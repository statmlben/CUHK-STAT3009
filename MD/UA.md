## User Mean Implementation Idea

### User Average

The user mean implementation idea is based on the assumption that a user's rating behavior is consistent across different items. The idea is to predict a user's rating for an item based on the user's average rating across all items.

Mathematically, the user mean can be represented as:

$
    \bar{r}_{u} = \frac{1}{|\mathcal{I}_u|} \sum_{i \in \mathcal{I}_u} r_{ui}, \text{ for } u=1, \cdots, n; \quad \hat{r}_{ui} = \bar{r}_u
$

where:

* $\bar{r}_{u}$ is the average rating for user $u$
* $\mathcal{I}_u$ is the set of items rated by user $u$
* $r_{ui}$ is the rating given by user $u$ to item $i$
* $\hat{r}_{ui}$ is the predicted rating for user $u$ and item $i$

The implementation steps for the user mean idea are:

### Algorithm

1. **Loop for all users**:
  * Find all records for this user in both training and testing sets.
  * Compute the average ratings for this user in the training set.
  * Predict the ratings for this user in the testing set.

### Python Implementation
```python
def user_mean(train_pair, train_rating, test_pair):
    """
    Predict ratings based on the mean rating for each user.

    Parameters:
    train_pair (array): Array of user-item pairs for training
    train_rating (array): Array of training ratings
    test_pair (array): Array of user-item pairs for testing

    Returns:
    pred (array): Predicted ratings
    """
    n_user = max(train_pair[:,0].max(), test_pair[:,0].max())+1
    pred = np.zeros(len(test_pair))
    glb_mean_value = train_rating.mean()
    
    # Loop through each user
    for u in range(n_user):
        # Find the indices for both train and test for user_id = u
        ind_test = np.where(test_pair[:,0] == u)[0]
        ind_train = np.where(train_pair[:,0] == u)[0]
        
        if len(ind_test) == 0:
            continue
        if len(ind_train) < 3:
            # If user has less than 3 ratings, use global mean
            pred[ind_test] = glb_mean_value
        else:
            # Predict as user average
            pred[ind_test] = train_rating[ind_train].mean()
    
    return pred
```