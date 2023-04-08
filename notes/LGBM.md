# Light Gradient-Boosting Machine (LGBM)

## I. Resources
- Original Paper (Guolin Ke et al, 2017): https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
- LGBM documentation: https://lightgbm.readthedocs.io/en/v3.3.2/index.html

## II. Algorithms

LGBM uses a leaf-wise tree growth algorithm, which may result in tree with higher depth (compared to depth-wise strategy).
This may lead to model being more susceptible to overfitting and high variance. Consider to limit tree depth as a potential solution if test performance is very difference from train performance.

![image](https://user-images.githubusercontent.com/99384454/230742896-c2f165b3-9959-46f5-b9a5-80aff5654130.png)

### 1. Gradient-based One-Side Sampling (GOSS)

![image](https://user-images.githubusercontent.com/99384454/230741615-c70618b8-62eb-40fd-b71c-91fbe21d2b9a.png)

Training set at a iteration consists of:
- Sample top a% data with large gradient
- Sample bottom b% data from remaining (1-a)% data with low gradients. Multiply sampled low-gradient data points with $\frac{1-a}{b}$ to maintain similar data distribution.
- 
Note that with a = 0, GOSS is equivalent to random sampling at b% rate.

![image](https://user-images.githubusercontent.com/99384454/230741760-6f02616d-3d50-45d9-a44c-18370aff4bae.png)

### 2. Exclusive Feature Bundling (EFB) for handling data sparsity

![image](https://user-images.githubusercontent.com/99384454/230742388-6c9285b9-b576-4d70-a167-13df163fd36f.png)

- Group exclusive features into a single feature (exclusive feature bundle). Feature bundles are features which rarely take nonzero values simultaneously (e.g. One-Hot-Encoded features). Bundle Grouping is approximated using Graph Coloring Algorithm.
- After grouping, create histogram on the bundle to represent it as binarized feature (each feature = 1 bin in feature bundle)

## III. Implementations

