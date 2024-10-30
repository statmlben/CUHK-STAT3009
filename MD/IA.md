## Item Mean Implementation Idea

### Item Average

The item mean implementation idea is based on the assumption that an item's rating behavior is consistent across different users. The idea is to predict a user's rating for an item based on the item's average rating across all users.

Mathematically, the item mean can be represented as:

$
		\bar{r}_{i} = \frac{1}{|\mathcal{U}_i|} \sum_{u \in \mathcal{U}_i} r_{ui}, \text{ for } i=1, \cdots, m; \quad \hat{r}_{ui} = \bar{r}_i,
$

where:

* $\bar{r}_{i}$ is the average rating for item $i$
* $\mathcal{U}_i$ is the set of users who rated item $i$
* $r_{ui}$ is the rating given by user $u$ to item $i$
* $\hat{r}_{ui}$ is the predicted rating for user $u$ and item $i$

The implementation steps for the item mean idea are:

### Algorithm

1. **Loop for all items**:
	* Find all records for this item in both training and testing sets.
	* Compute the average ratings for this item in the training set.
	* Predict the ratings for this item in the testing set.

### Python Implementation
```python
def item_mean(train_pair, train_rating, test_pair):
    """
    Predict ratings based on the mean rating for each item.

    Parameters:
    train_pair (array): Array of user-item pairs for training
    train_rating (array): Array of training ratings
    test_pair (array): Array of user-item pairs for testing

    Returns:
    pred (array): Predicted ratings
    """
    n_item = max(train_pair[:,1].max(), test_pair[:,1].max())+1
    pred = np.zeros(len(test_pair))
    glb_mean_value = train_rating.mean()
    
    # Loop through each item
    for i in range(n_item):
        # Find the indices for both train and test for item_id = i
        ind_test = np.where(test_pair[:,1] == i)[0]
        ind_train = np.where(train_pair[:,1] == i)[0]
        
        if len(ind_test) == 0:
            continue
        if len(ind_train) < 3:
            # If item has less than 3 ratings, use global mean
            pred[ind_test] = glb_mean_value
        else:
            # Predict as item average
            pred[ind_test] = train_rating[ind_train].mean()
    
    return pred