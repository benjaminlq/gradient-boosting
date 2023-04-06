# Gradient Boosting
Explanation and performance benchmark on GB algorithms (GBDT, XGBoost, LGBM, CatBoost)

# Gradient Boosting Decision Tree
## Resources:
- Original Paper (Friedman, 1999): https://jerryfriedman.su.domains/ftp/trebst.pdf 
- Youtube Explanation (StatQuest): https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6
- Scikit-Learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html & https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

## Algoirithms:
Extracted from Original Paper: https://jerryfriedman.su.domains/ftp/trebst.pdf 
### 1. General Algorithm:
- Step 1: Find Initial Predicted y s.t. residual is minimized (m = 0)
- Step 2: Iterate through M trees:
    - Step 2.1: For each data point, compute the residual between predicted value based on last tree (m - 1) and ground truth target value.
    - Step 2.2: Fit a new tree that minimizes the difference between the residuals and predicted residuals
    - Step 2.3: For each leave of the new tree, calculate the output value such that Loss function is minimized.
    - Step 2.4: Update new prediction by adding the new tree predicted residual to previous tree prediction
- Step 3: Inference by running the input data through the fitted trees.
<img align="center" width=800 src="https://user-images.githubusercontent.com/99384454/230105918-2a90ec6b-97c4-46a4-a754-cf4ba9190cf0.png">

### 2. Gradient-Boosting Regression Trees (GBRT)
There are several types of loss functions which can be used: least-square (LS), least-absolute-deviation (LSD), Huber (M). The most common is least-square, which is also the default implementation inside scikit-learn library. For LS, the initial prediction which minimizes the residual is simply the mean of all target values inside the training set.

<img align="center" width=800 src="https://user-images.githubusercontent.com/99384454/230270553-b3955362-d6fb-4c99-8ad2-185e64ad5517.png">
<img align="center" width=800 src="https://user-images.githubusercontent.com/99384454/230275697-121d4c95-1650-4c81-a364-8ae9258fe7a2.png">
<img align="center" width=800 src="https://user-images.githubusercontent.com/99384454/230275794-82c5fda2-9e7b-4ccf-80f1-c08088d6ff6c.png">

### 3. Gradient-Boosting Classification Trees (GBCT)
#### 3.1 Binary Classification
Loss function is **negative log-likelihood** .

$$ L(y,F(x)) = log(1+exp(-2yF(x))) $$

Prediction values for GBCT is **log(odds)**.

$$ F(x) = 1/2\*log(odds) = 1/2\*log(\frac{P(y=1|x)}{1-P(y=1|x)}) $$

Inference converts predicted **log(odds)** into predicted probability and make classification based on classification threshold (default = 0.5)

$$ Probability = \frac{e^{logodds}}{1+e^{logodds}} $$

<img align="center" width=800 src="https://user-images.githubusercontent.com/99384454/230282462-18d3a7e4-f7ac-40f9-9b95-a14483a39e4c.png">

#### 3.2 Multiclass Classification
Loss function is **Cross Entropy Loss** .

$$ Loss = -\sum_{k=1}^K{y_k\*logp_k(x)} $$

Tree prediction for each individual class

$$ F_k(x) = logp_k(x) - \frac{1}{K}\sum_{l=1}^{K}logp_l(x) $$

Probability for each individual class = **Softmax Probability**

$$ p_k(x) = \frac{exp(F_k(x))}{\sum{exp(F_l(x))}} $$

<img align="center" width=800 src="https://user-images.githubusercontent.com/99384454/230290602-0430560a-6eb4-48c0-8a3d-50a87ff255b5.png">

### 4. Implementation
#### 4.1 Fit, train and inference using **scikit-learn** library

```
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

### Classfication ###
gbc_model = GradientBoostingClassifier()
gbc_model.fit(x_train, y_train_c)
y_preds_c = gb_model.predict(x_test)
classification_score = accuracy_score(y_test_c, y_preds_c)

### Regression ###
gbr_model = GradientBoostingRegressor()
gbr_model.fit(x_train, y_train_r)
y_preds_r = gbr_model.predict(x_test)
regression_score = mean_squared_error(y_test_r, y_preds_r)
```

#### 4.2 Hyperparameters for tuning
- `loss`: For regression: ‘squared_error’, ‘absolute_error’, ‘huber’, ‘quantile’. Huber loss function (M-regression) is used for cases with long-tailed error distributions and outliers while maintaining high efficiency for normally distributed errors. 'quantile' is used for quantile regression
  For classification, 'log-loss' or 'exponential'.
- `learning rate` and `n-estimators`: `learning-rate` shrinks contribution of each tree by a fixed amount (Regularization), reducing the overfitting impact of a single tree to data. This also means that with lower `learning-rate`, more `n-estimators` (trees) are needed for GBDT to fit to the training data (reducing bias). A good strategy is to use **low** `learning-rate` and **high** `n-estimators`. <br>

<img align="center" width=1000 src="https://user-images.githubusercontent.com/99384454/230295836-82eec017-7e76-4964-afd7-b88a59bfdea2.png">

<img align="center" width=800 src="https://user-images.githubusercontent.com/99384454/230298501-bffb6ff9-8f06-4cc8-b87b-1d2b43308c34.png">

<img align="center" width=1000 src="https://user-images.githubusercontent.com/99384454/230301686-d26074dc-ed00-43e4-ba93-81ea4c04b092.png">

  However, large number of `n-estimators` can also cause overfitting. Early Stopping can be used to prevent GBDT to overfit to training data by stopping to add new trees once score on validation set no longer improves. `n_iter_no_change` stops adding new trees once validation score no longer improves after n iterations (= early stopping patience parameters). `validation_fraction` is the proportion of training set set aside as validation set for early stopping.
- **Regularization**
  - `min_samples_split`, `min_samples_leaf` and `min_impurity_decrease` control the tree size through split quality.
  - `max_depth` control tree size by limiting tree depth.
  - `max_leaf_nodes` control tree size by limiting no of leaf nodes. 
  - `ccp_alpha` add regularization for each nodes of the tree (Minimal Cost-Complexity Pruning)
- **Training Speed**
  - `max_features`: Number of features to consider when looking for best split. Choosing `max_features < n_features` reduces training time but leads to a reduction in variance and an increase in bias.
  - `subsample`: Determines the amount of training data used for fitting. Using a portion of training data results in faster training time but leads to a reduction in variance and an increase in bias.

#### 4.3 Key model attributes and methods
- `feature_importance_`: feature importance
- `staged_predict(X)` and `staged_predict_proba(X)`: Predict classification and regression target at each stage for X.
- `predict_proba(X)` and `predict_log_proba(X)`: For **classification** only. Return predicted values in probability. Useful if a different threshold needs to be set. 
 
# eXtreme Gradient-Boost (XGBoost)

# Dropout Multiple Additive Regression Trees (DART)

# Light Gradient-Boosting Machine (LGBM)

# CatBoost
