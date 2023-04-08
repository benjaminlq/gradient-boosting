# eXtreme Gradient-Boost (XGBoost)

## I. Resources
- Original Paper (Tianqi Chen, 2016): https://arxiv.org/pdf/1603.02754.pdf
- Youtube Explanation (StatQuest): https://www.youtube.com/watch?v=OtD8wVaFm6E&t=4s
- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/
- LightGBM documentation: https://lightgbm.readthedocs.io/en/v3.3.2/index.html

## II. Algorithms
#### [Histogram-Based Algorithm](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

![image](https://user-images.githubusercontent.com/99384454/230736463-14ecad2f-a50d-4b75-96cd-8d6a8ea4c020.png)

#### [Depth-Wise Tree Growth - Breadth First](https://lightgbm.readthedocs.io/en/v3.3.2/index.html)
![image](https://user-images.githubusercontent.com/99384454/230736536-5cda458f-c2bc-4c4f-a80f-79e873514e92.png)

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
- Regularization Loss Function with `lambda` and `gamma`
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

![image](https://user-images.githubusercontent.com/99384454/230474208-63e51f61-4df2-4a88-8e87-e499faecc330.png)

Splits are computed using **weighted quantile sketch** algorithm, which uses a data structure that supports *merge* and *prune* operations to handle weighted datasets.

#### 3.3. Sparsity-Aware Split Finding
To handle (1) Missing Values, (2) High sparsity in data (frequent zeroes, one-hot encoding, etc). Sparsity-Aware Split Finding learns the default direction from the data for each feature. Analogical to imputation at each node. For missing values, add gain of going left and right separately and set default direction to child node giving higher gain. 

![image](https://user-images.githubusercontent.com/99384454/230477228-11ea8d2c-8a25-42c3-889d-8b9ba4ae4b8a.png)

### 4. Performance Improvement
#### 4.1. Column Block

![image](https://user-images.githubusercontent.com/99384454/230479088-193afdff-806e-4b1c-8cad-85ddbd5d5cd5.png)

- Sorted columns stored in-memory **blocks** in compressed-column (CSC) format at the beginning.
- For exact greedy algorithm, only 1 block
- For approximate algorithm, many blocks (subset of rows) can be created and distributed across multiple machines or store on disk in out-of-core settings. Useful to speed-up split proposals, collection of bucket statistics as well as subsample of features.

#### 4.2. Cache-Aware and Out-Of-Core computation optimization
- Cache Aware Access: fetch the gradient statistics into cache memory to increase CPU speed during split determination step.
- Out-Of-Core computation: store compressed blocks on disks and pre-fetch them using separate threads

## III. Implementations

### 1. Basic Implementation

### 2. Hyper Parameter Tuning

### 3. Save/Load Model

### 4. Distributed Computing
#### 4.1 XGBoost with Dask

#### 4.2 XGBoost with PySpark
