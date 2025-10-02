## Beginner Math Guide for Credit Card Rewards Notebook

This guide explains the core math behind the notebook so you can understand what each step is doing. It focuses on simple, practical intuition first, with the exact formulas where useful.

### 1) Data Basics and Notation
- **Dataset**: a table with rows = customers and columns = features describing their behavior, plus a label `Reward_Tier`.
- We denote a single customer by a feature vector \(x = [x_1, x_2, \dots, x_d]\), where each \(x_j\) is a number (e.g., income, spend ratio).
- The label \(y\) is a class among {Basic, Gold, Platinum}.

### 2) Train/Test Split
We split data into training and test sets to estimate generalization.
- Training set: used to fit the model.
- Test set: used once at the end to measure performance.

No formula here; it’s a procedure to avoid overfitting. Typical split is 80%/20%.

### 3) Standardization (Scaler)
For numeric features, we often transform them to have mean 0 and standard deviation 1:
\[
z = \frac{x - \mu}{\sigma}
\]
where \(\mu\) is the feature mean on the training set and \(\sigma\) is the feature standard deviation. This keeps features on comparable scales, which helps many models and optimizers.

### 4) Classification Goal
We want a function \(f(x)\) that predicts the class label \(\hat{y}\) for a new customer \(x\). The classifier outputs either:
- a class label \(\hat{y} \in \{\text{Basic}, \text{Gold}, \text{Platinum}\}\), or
- class probabilities \(P(y = c \mid x)\) for each class \(c\).

### 5) Random Forest Classifier (Core Model)
A Random Forest is an ensemble of many decision trees. Each tree splits the feature space using simple rules like “Is `Annual_Income` > 900,000?”. The forest’s prediction is the majority vote across trees.

Key ideas and light math:
- Each decision tree partitions the feature space using binary splits to reduce impurity.
- Impurity (e.g., Gini) for a node with class proportions \(p_1, p_2, \dots, p_K\) is:
\[
Gini = 1 - \sum_{k=1}^{K} p_k^2
\]
- A split is chosen to maximize impurity reduction between parent and children nodes.
- The Random Forest trains many trees on bootstrapped (sampled with replacement) data and random feature subsets, which reduces variance and overfitting.

Prediction:
- For class probabilities, each tree votes a distribution; the forest averages them:
\[
\hat{P}(y=c\mid x) = \frac{1}{T} \sum_{t=1}^{T} P_t(y=c\mid x)
\]
where \(T\) is number of trees.

### 6) Evaluation Metrics

Given true labels \(y\) and predictions \(\hat{y}\), we compute:

- Accuracy:
\[
\text{Accuracy} = \frac{\text{# correct predictions}}{\text{# total predictions}}
\]

- Precision (per class c): out of all predicted as class c, how many are truly c?
\[
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}
\]

- Recall (per class c): out of all true class c, how many did we catch?
\[
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
\]

- F1-score (harmonic mean of precision and recall):
\[
\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
\]

- Confusion matrix: a table counting how predictions fall across true vs. predicted classes.

### 7) Feature Importance (Random Forest)
Random Forest feature importance estimates how much each feature reduces impurity, aggregated across all trees. A high importance means the feature frequently made helpful splits.

Simplified view:
\[
\text{Importance}(j) = \sum_{t=1}^{T} \sum_{\text{splits on } j} \Delta \text{Impurity}(\text{split})
\]
Values are typically normalized so they sum to 1 across features.

### 8) Cross-Validation (CV)
To get a more stable estimate of model performance, we do K-fold CV:
- Split training data into K folds.
- Train on K-1 folds and validate on the remaining fold.
- Repeat K times, each fold serving once as validation.
- Report the mean and standard deviation of the scores.

### 9) Optional: Text Features with TF‑IDF + Logistic Regression
If using the `Transaction_Summary` text column, we convert text into numeric features using TF‑IDF.

- Term Frequency (TF) of term t in document d:
\[
TF(t,d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}
\]

- Inverse Document Frequency (IDF): rarer terms get higher weight
\[
IDF(t) = \log \frac{N}{1 + DF(t)}
\]
where \(N\) is number of documents and \(DF(t)\) is number of documents containing term t.

- TF‑IDF weight:
\[
TF\mbox{-}IDF(t,d) = TF(t,d) \cdot IDF(t)
\]

Logistic Regression then models class probabilities via the logistic (softmax) function in multiclass:
\[
P(y=c\mid x) = \frac{\exp(w_c^\top x + b_c)}{\sum_{k} \exp(w_k^\top x + b_k)}
\]

### 10) Class Imbalance and Class Weights
If some classes (e.g., Platinum) are rare, the model can bias toward common classes. Class weights increase the penalty for misclassifying rare classes. Many models accept a weight \(w_c\) per class so that the loss emphasizes those examples more.

### 11) Practical Tips
- Keep a consistent preprocessing pipeline (scaler + model) to avoid data leakage.
- Validate inputs at inference time (ranges and types) before predicting.
- Prefer probabilistic outputs and thresholding if your business needs differ from accuracy (e.g., recall-oriented use cases).

### 12) Glossary (Quick Reference)
- **Feature**: an input variable describing a customer.
- **Label/Target**: the value to predict; here, the reward tier.
- **Overfitting**: model learns noise; performs well on train but poorly on unseen data.
- **Baseline**: simple method used for initial comparison.
- **Generalization**: performance on data not seen during training.


