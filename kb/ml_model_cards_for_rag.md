# Machine Learning Model Cards (RAG-Ready, Data-Centric Labels)

> Scope: model cards only.  
> Excludes workflow/process topics (e.g., train/test split, cross-validation).

---

## 1) Kernel Ridge Regression

- **id**: `kernel_ridge_regression`
- **library_name**: `sklearn.kernel_ridge`
- **labels**: `supervised` `continuous_target` `medium_sample_size` `nonlinear_signal` `smooth_function_prior` `moderate_compute_budget`
- **annotations**:  
  Kernel Ridge Regression combines L2-regularized regression with kernelized nonlinear mapping.  
  It is effective when the target is continuous and the relationship is smooth but nonlinear.  
  It often performs well on small-to-medium datasets where kernel matrix operations remain feasible.
- **When to use**:
  - Continuous prediction with clear nonlinear trend and moderate dataset size.
  - Need smoother function estimates than tree-based piecewise fits.
  - Compute budget can handle kernel-based training/inference.
- **Key hyperparameters**: `alpha` `kernel` `gamma` `degree` `coef0`
- **Tuning playbook**:
  1. Start with `kernel=rbf`; run log-scale search for `alpha` and `gamma`.
  2. If underfitting, reduce `alpha` or increase `gamma`; reverse for overfitting.
  3. Try `polynomial` kernel if domain suggests polynomial interactions.

---

## 2) Support Vector Machines (SVC/SVR)

- **id**: `support_vector_machines`
- **library_name**: `sklearn.svm`
- **labels**: `supervised` `binary_or_multiclass_target` `medium_sample_size` `high_dimensional_features` `margin_separable_or_near_separable` `moderate_to_high_compute_budget`
- **annotations**:  
  Support Vector Machines optimize a margin-based objective for strong generalization.  
  They are often strong on high-dimensional feature spaces and medium-scale datasets.  
  Kernel variants capture nonlinear boundaries but can become expensive at large scale.
- **When to use**:
  - Classification or regression with medium sample count and high-dimensional features.
  - Decision boundary is not purely linear but still structured.
  - You can afford moderate-to-high optimization cost for better margin behavior.
- **Key hyperparameters**: `C` `kernel` `gamma` `degree` `epsilon` `class_weight`
- **Tuning playbook**:
  1. Use `rbf` kernel as baseline; tune `C` and `gamma` on log grid.
  2. For imbalance, set `class_weight='balanced'`.
  3. For SVR, tune `epsilon` to trade off fit tightness vs robustness.

---

## 3) Nearest Neighbors (KNN)

- **id**: `nearest_neighbors`
- **library_name**: `sklearn.neighbors`
- **labels**: `supervised` `local_similarity_structure` `small_to_medium_sample_size` `distance_metric_reliable` `scaled_numeric_features` `low_training_high_inference_cost`
- **annotations**:  
  KNN predicts from nearby examples in feature space without learning a global parametric function.  
  It is useful when local neighborhood structure is meaningful and features are well-scaled.  
  Training is cheap, while inference can be costly for larger datasets.
- **When to use**:
  - Small/medium data where local similarity is a good predictor of target.
  - Numeric feature space where distance metrics are trustworthy after scaling.
  - You need a simple nonparametric baseline quickly.
- **Key hyperparameters**: `n_neighbors` `weights` `metric` `p` `algorithm` `leaf_size`
- **Tuning playbook**:
  1. Standardize features first.
  2. Sweep `n_neighbors` to balance variance and bias.
  3. Compare distance metrics and `weights='uniform'` vs `weights='distance'`.

---

## 4) Random Forest

- **id**: `random_forest`
- **library_name**: `sklearn.ensemble`
- **labels**: `supervised` `tabular_mixed_features` `nonlinear_interactions` `missing_scaling_tolerance` `medium_to_large_sample_size` `outlier_robust_requirement`
- **annotations**:  
  Random Forest aggregates many decorrelated trees for stable nonlinear prediction.  
  It handles mixed tabular features and interaction effects with minimal preprocessing.  
  It is a reliable baseline when robustness and practical performance matter.
- **When to use**:
  - Tabular data with possible nonlinear feature interactions.
  - Limited preprocessing pipeline and uncertain feature scaling quality.
  - Need robust baseline for medium-to-large datasets.
- **Key hyperparameters**: `n_estimators` `max_depth` `min_samples_split` `min_samples_leaf` `max_features` `bootstrap`
- **Tuning playbook**:
  1. Increase `n_estimators` until validation performance stabilizes.
  2. Control overfitting using `max_depth` and `min_samples_leaf`.
  3. Tune `max_features` for bias/variance tradeoff; add class weights if needed.

---

## 5) Neural Network (MLP, sklearn)

- **id**: `mlp_neural_network`
- **library_name**: `sklearn.neural_network`
- **labels**: `supervised` `continuous_or_categorical_target` `medium_sample_size` `nonlinear_signal` `feature_scaling_required` `moderate_compute_budget`
- **annotations**:  
  sklearn MLP is a feedforward neural approach for structured data tasks.  
  It captures nonlinear patterns beyond linear baselines but is sensitive to scaling and optimization settings.  
  It is suitable for medium-scale tasks when lightweight neural modeling is needed.
- **When to use**:
  - Structured data with nonlinear signal and moderate sample size.
  - You can standardize features and tune optimization settings.
  - Need a neural baseline without moving to full deep learning stacks.
- **Key hyperparameters**: `hidden_layer_sizes` `activation` `alpha` `learning_rate_init` `solver` `batch_size` `max_iter` `early_stopping`
- **Tuning playbook**:
  1. Standardize inputs and begin with compact architecture.
  2. Increase width/depth for underfitting; increase `alpha` for overfitting.
  3. Lower learning rate if loss is unstable; enable `early_stopping` when needed.

---

## 6) Neural Network (PyTorch Feedforward DNN)

- **id**: `pytorch_feedforward_nn`
- **library_name**: `torch.nn`
- **labels**: `supervised` `large_sample_size` `complex_nonlinear_patterns` `representation_learning_needed` `gpu_available` `distributed_training_possible`
- **annotations**:  
  PyTorch feedforward networks support custom architectures and full control of training logic.  
  They are appropriate for larger-scale nonlinear tasks requiring GPU acceleration.  
  They integrate naturally with mixed precision and distributed training workflows.
- **When to use**:
  - Large datasets and complex nonlinear decision surfaces.
  - Need custom loss functions, regularizers, or training loops.
  - GPU/distributed infrastructure is available.
- **Key hyperparameters**: `num_layers` `hidden_dim` `dropout` `learning_rate` `weight_decay` `batch_size` `optimizer` `scheduler`
- **Tuning playbook**:
  1. Establish baseline with small architecture and verified data pipeline.
  2. Tune learning rate and regularization before scaling model capacity.
  3. Use AMP and DDP to improve throughput on large training jobs.

---

## 7) Neural Network (PyTorch CNN)

- **id**: `pytorch_cnn`
- **library_name**: `torch.nn`
- **labels**: `supervised` `grid_like_inputs` `spatial_local_correlation` `large_sample_size_or_pretraining_available` `translation_tolerance_needed` `gpu_available`
- **annotations**:  
  CNNs are specialized for spatially structured inputs such as images.  
  They exploit local patterns and weight sharing for parameter-efficient representation learning.  
  They are standard for vision tasks and benefit strongly from pretrained backbones.
- **When to use**:
  - Input has spatial locality (images, maps, frames).
  - Need translation-tolerant pattern extraction.
  - GPU resources or transfer learning checkpoints are available.
- **Key hyperparameters**: `num_filters` `kernel_size` `stride` `padding` `learning_rate` `batch_size` `weight_decay`
- **Tuning playbook**:
  1. Start with pretrained backbone and tune head first.
  2. Add augmentation and regularization to reduce overfitting.
  3. Unfreeze deeper layers progressively with learning-rate scheduling.

---

## 8) Gaussian Mixture Models (GMM)

- **id**: `gaussian_mixture_models`
- **library_name**: `sklearn.mixture`
- **labels**: `unsupervised` `overlapping_groups` `soft_membership_needed` `elliptical_cluster_tendency` `density_estimation_use_case` `medium_sample_size`
- **annotations**:  
  GMM models data as a mixture of Gaussian components estimated by EM.  
  It provides probabilistic cluster memberships instead of hard assignments.  
  It is useful when groups overlap and uncertainty-aware assignment is required.
- **When to use**:
  - Clusters are expected to overlap in feature space.
  - Need probability outputs for cluster membership.
  - Interested in both clustering and density modeling.
- **Key hyperparameters**: `n_components` `covariance_type` `reg_covar` `init_params` `max_iter`
- **Tuning playbook**:
  1. Select component count using BIC/AIC.
  2. Compare covariance structures (`full`, `diag`, `tied`, `spherical`).
  3. Improve stability with multiple restarts and regularization.

---

## 9) K-Means

- **id**: `kmeans`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised` `compact_cluster_tendency` `low_to_moderate_noise` `roughly_equal_cluster_scale` `large_sample_size` `fast_iteration_requirement`
- **annotations**:  
  K-Means partitions points around centroids to minimize within-cluster variance.  
  It is fast and scalable, but assumes compact clusters and can be sensitive to outliers.  
  It is often the first clustering baseline for large tabular embeddings.
- **When to use**:
  - Need fast unsupervised grouping at scale.
  - Cluster geometry is roughly compact and centroid-like.
  - Approximate cluster count can be estimated.
- **Key hyperparameters**: `n_clusters` `init` `n_init` `max_iter` `tol` `algorithm`
- **Tuning playbook**:
  1. Estimate `n_clusters` with elbow/silhouette diagnostics.
  2. Increase `n_init` to reduce initialization variance.
  3. Normalize features and mitigate outliers before fitting.

---

## 10) DBSCAN

- **id**: `dbscan`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised` `unknown_cluster_count` `arbitrary_shape_groups` `noise_points_expected` `density_separation_present` `small_to_medium_sample_size`
- **annotations**:  
  DBSCAN discovers clusters as dense connected regions and labels sparse points as noise.  
  It works well for irregular cluster shapes without requiring predefined cluster count.  
  Performance depends heavily on local density parameter settings.
- **When to use**:
  - Number of clusters is unknown.
  - Expect non-convex groups and meaningful outliers.
  - Data has density-separated structure at a useful scale.
- **Key hyperparameters**: `eps` `min_samples` `metric` `algorithm` `leaf_size`
- **Tuning playbook**:
  1. Use k-distance plot to initialize `eps`.
  2. Adjust `min_samples` according to expected local density.
  3. Scale features and reduce dimensionality if distance concentration appears.

---

## 11) Agglomerative Clustering

- **id**: `agglomerative_clustering`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised` `hierarchical_group_structure` `small_to_medium_sample_size` `distance_matrix_meaningful` `multi_granularity_analysis_needed` `interpretability_priority`
- **annotations**:  
  Agglomerative clustering merges samples/clusters bottom-up to form a hierarchy.  
  It is useful when analysts need multi-level grouping rather than one fixed partition.  
  Linkage choice controls how cluster proximity is defined.
- **When to use**:
  - Need hierarchical interpretation of group structure.
  - Pairwise distance notion is meaningful for your domain.
  - Dataset size allows hierarchical merging costs.
- **Key hyperparameters**: `n_clusters` `metric` `linkage` `distance_threshold` `connectivity`
- **Tuning playbook**:
  1. Choose linkage based on geometry assumptions.
  2. Explore hierarchy cuts via `distance_threshold`.
  3. Compare cluster validity metrics across cut levels.

---

## 12) Spectral Clustering

- **id**: `spectral_clustering`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised` `nonconvex_group_structure` `graph_similarity_available` `manifold_like_structure` `medium_sample_size` `higher_compute_budget`
- **annotations**:  
  Spectral clustering uses graph Laplacian eigenspace to separate complex cluster geometry.  
  It is strong for manifold-like or non-convex structures where centroid methods fail.  
  Success depends on good similarity graph construction and moderate compute budget.
- **When to use**:
  - Data manifold is curved/non-convex.
  - A meaningful affinity graph can be defined.
  - Dataset size is moderate enough for eigendecomposition.
- **Key hyperparameters**: `n_clusters` `affinity` `gamma` `n_neighbors` `assign_labels`
- **Tuning playbook**:
  1. Pick affinity (`rbf` or `nearest_neighbors`) from domain structure.
  2. Tune graph scale parameters (`gamma` or `n_neighbors`).
  3. Validate stability across multiple random seeds and graph settings.

---

## 13) Birch

- **id**: `birch`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised` `very_large_sample_size` `memory_constrained_setting` `incremental_ingestion` `coarse_to_fine_grouping_needed` `high_throughput_requirement`
- **annotations**:  
  Birch incrementally compresses data into clustering features using a CF tree.  
  It is designed for large-scale or memory-constrained clustering scenarios.  
  It supports coarse-to-fine grouping workflows efficiently.
- **When to use**:
  - Dataset is very large or arrives in chunks.
  - Memory efficiency is a hard requirement.
  - Need fast pre-clustering before downstream refinement.
- **Key hyperparameters**: `threshold` `branching_factor` `n_clusters`
- **Tuning playbook**:
  1. Tune `threshold` to set subcluster granularity.
  2. Adjust `branching_factor` for tree growth/efficiency balance.
  3. Configure final global clustering (`n_clusters`) per downstream use.

---

## 14) OPTICS

- **id**: `optics`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised` `variable_density_groups` `noise_points_expected` `unknown_cluster_count` `multi_scale_density_structure` `small_to_medium_sample_size`
- **annotations**:  
  OPTICS is a density-based method that handles variable-density structure better than fixed-threshold alternatives.  
  It produces reachability ordering for multi-scale cluster extraction.  
  It is useful when one global density setting is too restrictive.
- **When to use**:
  - Cluster density varies across regions.
  - Need noise-aware clustering without fixed cluster count.
  - Multi-scale exploratory clustering is required.
- **Key hyperparameters**: `min_samples` `max_eps` `metric` `cluster_method` `xi` `min_cluster_size`
- **Tuning playbook**:
  1. Set broad `max_eps` for structure discovery.
  2. Tune `xi` and `min_cluster_size` for extraction detail level.
  3. Standardize features; reduce dimension when density estimation is unstable.

---

## 15) MiniBatchKMeans

- **id**: `mini_batch_kmeans`
- **library_name**: `sklearn.cluster`
- **labels**: `unsupervised` `very_large_sample_size` `streaming_or_chunked_input` `compact_cluster_tendency` `fast_update_requirement` `approximation_acceptable`
- **annotations**:  
  MiniBatchKMeans scales centroid-based clustering through stochastic mini-batch updates.  
  It offers major speed gains with small quality trade-offs versus full K-Means.  
  It is practical for high-throughput or continuously updated data pipelines.
- **When to use**:
  - Data volume is too large for full-batch K-Means.
  - Need frequent centroid updates in production.
  - Slight approximation error is acceptable.
- **Key hyperparameters**: `n_clusters` `batch_size` `max_iter` `n_init` `reassignment_ratio`
- **Tuning playbook**:
  1. Increase `batch_size` to stabilize updates.
  2. Use larger `n_init` for robust starting centroids.
  3. Monitor inertia trajectory and adjust iteration budget.

---

## 16) Linear / Logistic Models (Baseline Family)

- **id**: `linear_and_logistic_models`
- **library_name**: `sklearn.linear_model`
- **labels**: `supervised` `linear_or_near_linear_signal` `high_dimensional_sparse_features` `large_sample_size` `strong_interpretability_requirement` `low_latency_requirement`
- **annotations**:  
  Linear and logistic models are efficient and interpretable baselines for regression/classification.  
  They perform especially well on high-dimensional sparse features with proper regularization.  
  They are strong first-choice models when latency and explainability are priorities.
- **When to use**:
  - Signal is approximately linear after feature engineering.
  - Sparse high-dimensional features are present (e.g., text vectors).
  - Need low-latency and interpretable production behavior.
- **Key hyperparameters**: `alpha` `C` `penalty` `solver` `l1_ratio`
- **Tuning playbook**:
  1. Standardize when appropriate and start with L2 regularization.
  2. Tune regularization strength (`alpha` or inverse `C`) on log scale.
  3. Use L1/Elastic Net when feature sparsity is desired.