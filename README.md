# Gradient Boosting
Explanation and performance benchmark on GB algorithms (GBDT, XGBoost, LGBM, CatBoost)

# Gradient Boosting Decision Tree
## I. Resources:
- Original Paper (Friedman, 1999): https://jerryfriedman.su.domains/ftp/trebst.pdf 
- Youtube Explanation (StatQuest): https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6
- Scikit-Learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html & https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

## II. Algorithms:
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

Loss function is **mean-squared-error** .

$$ L(y,F(x)) = 1/2\*(y-F(x))^2 $$

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

## III. Implementation
### 1. Fit, train and inference using **scikit-learn** library

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

### 2. Hyperparameters for tuning
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
  - `max_features`: Number of features to consider when looking for best split. Choosing `max_features < n_features` leads to a reduction in variance and an increase in bias.
  - `subsample`: Determines the amount of training data used for fitting. Using a portion of training data leads to a reduction in variance and an increase in bias.

### 3 Key model attributes and methods
- `feature_importance_`: feature importance
- `staged_predict(X)` and `staged_predict_proba(X)`: Predict classification and regression target at each stage for X.
- `predict_proba(X)` and `predict_log_proba(X)`: For **classification** only. Return predicted values in probability. Useful if a different threshold needs to be set. 
 
# eXtreme Gradient-Boost (XGBoost)
## I. Resources
- Original Paper (Tianqi Chen, 2016): https://arxiv.org/pdf/1603.02754.pdf
- Youtube Explanation (StatQuest): https://www.youtube.com/watch?v=OtD8wVaFm6E&t=4s
- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/

## II. Algorithms
### 1. Regularized Loss Function with Gradient Tree Boosting

Compared to GBDT, xGBoost loss function add extra regularization term, which penalizes complex tree. The regularization terms penalizes the trees for having additional leaf nodes and sum of scores of the leave nodes (which increases with the number of leaf nodes).

<img align="center" width=1000 src="https://user-images.githubusercontent.com/99384454/230407389-8b2efc40-eefc-473f-b151-69748e5aae5e.png">

The difference between the observed values $y_i$ and predicted values $y\_hat_i^{t-1} + f_t(x_i)$ is approximated using Second Degree Taylor Series Expansion to yield an expression easily differentiable w.r.t f(x). $x = f_t(x_i)$ and $a = \frac{d}{d(y_hat_i^{t-1})}l(y_i, y_hat_i^{t-1})$

**Taylor's Theorem** (https://en.wikipedia.org/wiki/Taylor%27s_theorem)

<img align="center" width=1000 src="https://user-images.githubusercontent.com/99384454/230409510-e3327e99-6cb3-4272-8339-6f1bb57dc8e5.png">

<img align="center" width=650 src="https://user-images.githubusercontent.com/99384454/230411978-739fd6ba-7758-4644-85b4-6c840d008595.png">

Regularization term lambda reduces the impact of data to the objective function (Loss = Tree Impurity), hence reduces overfitting. Higher regularization term will lead to bigger score (impurity) and smaller split score (impurity reduction) on nodes.

<img align="center" width=650 src="https://user-images.githubusercontent.com/99384454/230417813-1f2ba1e3-a34f-48cd-b99b-523d5b31aee5.png">

### 2. Regularization
Several regularization methods were included inside XGBoost implementation
- Regularization Loss Function with `lambda` and `gamma` (II.1)
- `learning-rate`: Shrinkage reduce impact of current tree to the prediction and allows future tree to improve the model.
- **Sub-sampling** of data and features both help to reduce variance at the expense of increase bias.

### 3. Split Algorithm
Split Algorithm is done by greedily creating tree nodes at each time step with reduces most loss. Loss reduction is sum of losses of child leaves minus parent leaf.

![image](https://user-images.githubusercontent.com/99384454/230453838-7d809364-3526-410e-825a-583d9f5da760.png)

#### 3.1 Exact Greedy Algorithm
Iterate through all possible splits. Guarantee to find the best possible split at that node. For continuous features, sorting is required making this computationally expensive.

![image](https://user-images.githubusercontent.com/99384454/230454604-2cccea8c-1f15-42aa-ae6e-06f746571ea9.png)

#### 3.2 Approximate Algorithm
Allocate continuous values into buckets (histogram) and aggregate statistics within buckets. Best split is determined based on the buckets aggregated statistics. Candidate split proposals can be computed just once at the beginning (**global**) or every iterations (**local**). **Global** method needs few proposal but requires more splits while **Local** needs more proposal (every iteration), but need fewer split.

## III. Implementations

# Dropout Multiple Additive Regression Trees (DART)
## I. Resources
- Original Paper (Rashmi & Gilad-Bachrach, 2015): https://arxiv.org/abs/1505.01866

## II. Algorithms

## III. Implementations


# Light Gradient-Boosting Machine (LGBM)
## I. Resources
- Original Paper (Guolin Ke et al, 2017): https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
- LGBM documentation: https://lightgbm.readthedocs.io/en/v3.3.2/index.html

## II. Algorithms
### 1. 
### 2. 

## III. Implementations


# CatBoost
## I. Resources
- Original Paper (Tianqi Chen, 2016): https://arxiv.org/pdf/1603.02754.pdf
- Youtube Explanation (StatQuest): https://www.youtube.com/watch?v=OtD8wVaFm6E&t=4s
- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/

## II. Algorithms

## III. Implementations
