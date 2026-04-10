# Machine Learning Model Cards (RAG-Ready)

> Scope: model-only cards for retrieval and LLM grounding.  
> Excludes workflow topics (e.g., cross-validation, train/test split).

---

## 1) Kernel Ridge Regression

- **id**: `kernel_ridge_regression`
- **library_name**: `sklearn.kernel_ridge`
- **labels**: `supervised_learning` `regression` `kernel_method` `nonlinear_mapping` `medium_scale`
- **annotations**:  
  Kernel Ridge Regression (KRR) combines ridge regression with the kernel trick to model nonlinear relationships.  
  It applies L2 regularization in an implicit high-dimensional feature space, often yielding smooth function fits.  
  KRR is typically effective on small-to-medium datasets where kernel matrix computation is still tractable.
- **When to use**:
  - Nonlinear regression signal is clear and you want regularized fitting.
  - Dataset size is small to medium.
  - You need a strong classical nonlinear baseline before deep models.
- **Key hyperparameters**: `alpha` `kernel` `gamma` `degree` `coef0`
- **Tuning playbook**:
  1. Start with `kernel=rbf`; run log-scale search on `alpha` and `gamma`.
  2. If underfitting, decrease `alpha` or increase `gamma`; if overfitting, do the opposite.
  3. Try `polynomial` kernel and tune `degree`/`coef0` when domain structure suggests polynomial interactions.

---

## 2) Support Vector Machines (SVC/SVR)

- **id**: `support_vector_machines`
- **library_name**: `sklearn.svm`
- **labels**: `supervised_learning` `classification_or_regression` `kernel_method` `max_margin` `medium_scale`
- **annotations**:  
  Support Vector Machines optimize a maximum-margin objective and support both linear and nonlinear decision boundaries.  
  SVC is used for classification and SVR for regression.  
  They are strong on high-dimensional data and moderate dataset sizes, but can be expensive on very large datasets.
- **When to use**:
  - You need robust generalization on medium-scale tabular or text features.
  - Feature dimension is high and linear models are insufficient.
  - You want a well-established classical baseline with strong theory.
- **Key hyperparameters**: `C` `kernel` `gamma` `degree` `epsilon` `class_weight`
- **Tuning playbook**:
  1. Start with `kernel=rbf`; tune `C` and `gamma` on a log grid.
  2. Use `class_weight='balanced'` for class imbalance.
  3. For SVR, tune `epsilon` to control tolerance band and robustness.

---

## 3) Nearest Neighbors (KNN)

- **id**: `nearest_neighbors`
- **library_name**: `sklearn.neighbors`
- **labels**: `supervised_learning` `classification_or_regression` `instance_based` `distance_based` `small_to_medium_scale`
- **annotations**:  
  K-Nearest Neighbors predicts by local neighborhood voting or averaging.  
  It has near-zero training cost but higher inference-time cost due to distance computation.  
  Performance is highly sensitive to feature scaling and distance metric choice.
- **When to use**:
  - You want a fast, simple baseline without heavy training.
  - Local similarity is a valid assumption in your data.
  - Dataset size is manageable for nearest-neighbor lookup.
- **Key hyperparameters**: `n_neighbors` `weights` `metric` `p` `algorithm` `leaf_size`
- **Tuning playbook**:
  1. Always scale features before tuning.
  2. Sweep `n_neighbors` to balance bias and variance.
  3. Compare `weights='uniform'` vs `weights='distance'` and test multiple metrics.

---

## 4) Random Forest

- **id**: `random_forest`
- **library_name**: `sklearn.ensemble`
- **labels**: `supervised_learning` `classification_or_regression` `tree_based` `bagging_ensemble` `scalable_cpu`
- **annotations**:  
  Random Forest is an ensemble of decision trees built with bootstrap sampling and feature subsampling.  
  It captures nonlinear patterns and feature interactions with strong out-of-the-box performance on tabular data.  
  It is robust and parallelizable on CPU, making it a common production baseline.
- **When to use**:
  - You need a reliable tabular baseline with minimal preprocessing.
  - Nonlinear effects and interactions are expected.
  - You want stable performance with good robustness to noisy features.
- **Key hyperparameters**: `n_estimators` `max_depth` `min_samples_split` `min_samples_leaf` `max_features` `bootstrap`
- **Tuning playbook**:
  1. Increase `n_estimators` until validation performance plateaus.
  2. Control overfitting via `max_depth` and `min_samples_leaf`.
  3. Tune `max_features` to improve generalization; use class weighting if needed.

---

## 5) Neural Network (MLP, sklearn)

- **id**: `mlp_neural_network`
- **library_name**: `sklearn.neural_network`
- **labels**: `supervised_learning` `classification_or_regression` `neural_network` `nonlinear_function_approximator` `medium_scale`
- **annotations**:  
  sklearn MLP provides feedforward neural networks for classification and regression on structured data.  
  It is more expressive than linear models but usually more sensitive to scaling and hyperparameters.  
  It is best suited for small-to-medium datasets and quick neural baselines in the sklearn ecosystem.
- **When to use**:
  - Linear models underfit and you need more expressive power.
  - You want a neural baseline without leaving sklearn.
  - Dataset is medium scale and model customization needs are limited.
- **Key hyperparameters**: `hidden_layer_sizes` `activation` `alpha` `learning_rate_init` `solver` `batch_size` `max_iter` `early_stopping`
- **Tuning playbook**:
  1. Standardize inputs and start with a small network.
  2. Increase width/depth for underfitting; increase regularization for overfitting.
  3. Stabilize optimization by lowering `learning_rate_init` when needed.

---

## 6) Neural Network (PyTorch Feedforward DNN)

- **id**: `pytorch_feedforward_nn`
- **library_name**: `torch.nn`
- **labels**: `supervised_learning` `classification_or_regression` `deep_neural_network` `gpu_accelerated` `distributed_training_ready`
- **annotations**:  
  PyTorch feedforward networks support highly customizable architectures and training loops.  
  They integrate naturally with GPU acceleration, mixed precision, and distributed training.  
  This makes them suitable for larger datasets and production-grade deep learning pipelines.
- **When to use**:
  - You need custom losses, layers, or training logic.
  - You want GPU or multi-GPU acceleration.
  - You need flexibility beyond sklearn-level abstractions.
- **Key hyperparameters**: `num_layers` `hidden_dim` `dropout` `learning_rate` `weight_decay` `batch_size` `optimizer` `scheduler`
- **Tuning playbook**:
  1. Validate pipeline with a small model first.
  2. Tune `learning_rate` and regularization before scaling model size.
  3. Use AMP and DDP for throughput on large workloads.

---

## 7) Neural Network (PyTorch CNN)

- **id**: `pytorch_cnn`
- **library_name**: `torch.nn`
- **labels**: `supervised_learning` `classification` `convolutional_neural_network` `vision_model` `gpu_accelerated`
- **annotations**:  
  CNNs are designed for grid-structured data such as images, using convolutional filters and parameter sharing.  
  They efficiently capture local patterns and spatial hierarchies.  
  CNNs are a standard baseline for computer vision tasks and benefit strongly from transfer learning.
- **When to use**:
  - Inputs are images or spatially structured tensors.
  - You need strong vision performance with established architectures.
  - You can leverage pretrained backbones for faster convergence.
- **Key hyperparameters**: `num_filters` `kernel_size` `stride` `padding` `learning_rate` `batch_size` `weight_decay`
- **Tuning playbook**:
  1. Start from pretrained backbones and fine-tune head layers first.
  2. Add augmentation and regularization to handle overfitting.
  3. Unfreeze deeper layers progressively with learning-rate scheduling.

---

## 8) Gaussian Mixture Models (GMM)

- **id**: `gaussian_mixture_models`
- **library_name**: `sklearn.mixture`
- **labels**: `unsupervised_learning` `clustering` `probabilistic_model` `density_based_modeling` `medium_scale`
- **annotations**:  
  Gaussian Mixture Models represent data as a weighted sum of Gaussian components learned via EM.  
  Unlike K-Means, GMM provides soft assignments (cluster membership probabilities).  
  It is useful when clusters overlap and uncertainty estimates are valuable.
- **When to use**:
  - You need probabilistic cluster assignments.
  - Cluster overlap is expected.
  - You want density modeling in addition to partitioning.
- **Key hyperparameters**: `n_components` `covariance_type` `reg_covar` `init_params` `max_iter`
- **Tuning playbook**:
  1. Select `n_components` with BIC/AIC guidance.
  2. Compare `covariance_type` options based on cluster geometry.
  3. Improve stability with stronger regularization and multiple initializations.

---

## 9) K-Means

- **id**: `kmeans`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised_learning` `clustering` `centroid_based` `distance_based` `scalable_cpu`
- **annotations**:  
  K-Means partitions samples into K clusters by minimizing within-cluster squared distances.  
  It is fast and widely used for baseline clustering on large datasets.  
  It assumes roughly convex/spherical clusters and is sensitive to initialization and outliers.
- **When to use**:
  - You need a fast clustering baseline.
  - Cluster count can be estimated reasonably.
  - Data geometry is approximately centroid-separable.
- **Key hyperparameters**: `n_clusters` `init` `n_init` `max_iter` `tol` `algorithm`
- **Tuning playbook**:
  1. Estimate `n_clusters` using elbow/silhouette signals.
  2. Increase `n_init` for robust solutions.
  3. Normalize features and handle outliers before fitting.

---

## 10) DBSCAN

- **id**: `dbscan`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised_learning` `clustering` `density_based` `noise_robust` `arbitrary_shape_clusters`
- **annotations**:  
  DBSCAN groups points by density connectivity and explicitly marks noise points.  
  It handles arbitrary cluster shapes better than centroid-based methods.  
  It does not require a predefined number of clusters but is sensitive to neighborhood parameters.
- **When to use**:
  - You expect irregular cluster shapes.
  - Noise/outlier detection is important.
  - Predefining K is difficult.
- **Key hyperparameters**: `eps` `min_samples` `metric` `algorithm` `leaf_size`
- **Tuning playbook**:
  1. Use k-distance plots to initialize `eps`.
  2. Tune `min_samples` based on density and noise tolerance.
  3. Scale features; consider dimensionality reduction in high-dimensional spaces.

---

## 11) Agglomerative Clustering

- **id**: `agglomerative_clustering`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised_learning` `clustering` `hierarchical_method` `distance_linkage` `small_to_medium_scale`
- **annotations**:  
  Agglomerative clustering builds clusters bottom-up by repeatedly merging nearest groups.  
  It provides hierarchical structure that can be cut at different granularity levels.  
  Different linkage choices induce different cluster shape preferences.
- **When to use**:
  - You need hierarchical cluster structure for analysis.
  - Dataset size is manageable.
  - You want flexible post-hoc selection of cluster granularity.
- **Key hyperparameters**: `n_clusters` `metric` `linkage` `distance_threshold` `connectivity`
- **Tuning playbook**:
  1. Start with a baseline linkage (e.g., `ward` when applicable).
  2. Explore `distance_threshold` to choose meaningful hierarchy cuts.
  3. Compare linkage strategies using internal clustering metrics.

---

## 12) Spectral Clustering

- **id**: `spectral_clustering`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised_learning` `clustering` `graph_based_method` `nonconvex_cluster_friendly` `medium_scale`
- **annotations**:  
  Spectral clustering uses eigendecomposition of a graph Laplacian derived from sample similarities.  
  It can separate complex non-convex structures that challenge K-Means.  
  Performance depends strongly on affinity graph construction and scale parameters.
- **When to use**:
  - Cluster structure is non-convex or manifold-like.
  - You can define a meaningful similarity graph.
  - Dataset size supports eigendecomposition costs.
- **Key hyperparameters**: `n_clusters` `affinity` `gamma` `n_neighbors` `assign_labels`
- **Tuning playbook**:
  1. Choose affinity type (`rbf` or `nearest_neighbors`) based on data geometry.
  2. Tune `gamma` or `n_neighbors` to control graph connectivity.
  3. Validate cluster stability across multiple settings.

---

## 13) Birch

- **id**: `birch`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised_learning` `clustering` `hierarchical_compression_tree` `incremental_learning` `large_scale`
- **annotations**:  
  Birch incrementally builds a clustering feature tree to compress large datasets.  
  It is memory-efficient and supports scalable preprocessing before global clustering.  
  It is useful when full pairwise operations are too expensive.
- **When to use**:
  - You need scalable clustering on large datasets.
  - Incremental or memory-aware processing is required.
  - You want micro-cluster summarization before final grouping.
- **Key hyperparameters**: `threshold` `branching_factor` `n_clusters`
- **Tuning playbook**:
  1. Tune `threshold` to control micro-cluster granularity.
  2. Adjust `branching_factor` for tree capacity and efficiency.
  3. Set `n_clusters` based on downstream partition requirements.

---

## 14) OPTICS

- **id**: `optics`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised_learning` `clustering` `density_based` `variable_density_friendly` `noise_robust`
- **annotations**:  
  OPTICS extends density-based clustering to better handle variable-density structures.  
  It produces reachability ordering that supports multi-scale cluster extraction.  
  It is often more flexible than DBSCAN when a single global density threshold is inadequate.
- **When to use**:
  - Cluster densities vary across regions.
  - You need multi-scale density analysis.
  - Noise handling remains an important requirement.
- **Key hyperparameters**: `min_samples` `max_eps` `metric` `cluster_method` `xi` `min_cluster_size`
- **Tuning playbook**:
  1. Use a large `max_eps` for broad structure discovery.
  2. Tune `xi` and `min_cluster_size` for extraction granularity.
  3. Apply feature scaling and optional dimensionality reduction for stability.

---

## 15) MiniBatchKMeans

- **id**: `mini_batch_kmeans`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised_learning` `clustering` `centroid_based` `mini_batch_optimization` `large_scale`
- **annotations**:  
  MiniBatchKMeans is a scalable approximation of K-Means using mini-batch updates.  
  It significantly reduces training time on large datasets with modest quality trade-offs.  
  It is effective in streaming or high-throughput clustering workflows.
- **When to use**:
  - Standard K-Means is too slow at your data scale.
  - You need faster iterative updates.
  - Approximate clustering quality is acceptable.
- **Key hyperparameters**: `n_clusters` `batch_size` `max_iter` `n_init` `reassignment_ratio`
- **Tuning playbook**:
  1. Start with a reasonably large `batch_size` for stable centroid updates.
  2. Increase `n_init` to reduce sensitivity to initialization.
  3. Monitor inertia trend and adjust iteration budget accordingly.

---

## 16) Linear / Logistic Models (Baseline Family)

- **id**: `linear_and_logistic_models`
- **library_name**: `sklearn.linear_model`
- **labels**: `supervised_learning` `classification_or_regression` `linear_model` `regularized_optimization` `highly_scalable`
- **annotations**:  
  Linear and logistic models are efficient, interpretable, and strong first-line baselines.  
  With regularization (L1/L2/Elastic Net), they often generalize well and can support feature selection.  
  They are especially effective on high-dimensional sparse features.
- **When to use**:
  - You need interpretable and fast models.
  - You want a robust baseline before nonlinear models.
  - Feature-target relation is approximately linear or linearly separable after engineering.
- **Key hyperparameters**: `alpha` `C` `penalty` `solver` `l1_ratio`
- **Tuning playbook**:
  1. Standardize features and start with L2 regularization.
  2. Tune regularization strength (`alpha`/`C`) for bias-variance balance.
  3. Use L1 or Elastic Net when sparsity is beneficial.