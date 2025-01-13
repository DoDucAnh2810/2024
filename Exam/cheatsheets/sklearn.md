## 1. TF-IDF Feature Extraction

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one."
]

# 1. Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# 2. Learn vocabulary and create TF-IDF matrix
X_tfidf = vectorizer.fit_transform(corpus)

# 3. Print feature names
print("Feature Names:", vectorizer.get_feature_names_out())

# 4. Convert sparse matrix to array
print("TF-IDF Matrix:\n", X_tfidf.toarray())
```

### Common TfidfVectorizer Parameters

- **max_features**: Limit the maximum number of features (e.g., `max_features=1000`)
- **ngram_range**: Use unigrams, bigrams, etc. (e.g., `ngram_range=(1,2)`)
- **stop_words**: Remove stop words (e.g., `stop_words='english'`)
- **min_df, max_df**: Filter out rare or very frequent terms
- **tokenizer**: Supply a custom tokenizer for special preprocessing

---

## 2. Classification Models

### 2.1 Splitting the Dataset

```python
from sklearn.model_selection import train_test_split

# Suppose you have labels in y
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,   # feature matrix (TF-IDF)
    y,         # labels
    test_size=0.2,
    random_state=42
)
```

### 2.2 Logistic Regression Example

```python
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train (fit) the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)
```

### 2.3 Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
```

### 2.4 Support Vector Classifier (SVC)

```python
from sklearn.svm import SVC

svc_clf = SVC(kernel='linear')
svc_clf.fit(X_train, y_train)
y_pred = svc_clf.predict(X_test)
```

---

## 3. Model Evaluation

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
```

---

## 4. Cross-Validation and Hyperparameter Tuning

### 4.1 Cross-Validation

```python
from sklearn.model_selection import cross_val_score

log_reg = LogisticRegression()
scores = cross_val_score(log_reg, X_tfidf, y, cv=5, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))
```

### 4.2 GridSearchCV for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

svc_clf = SVC()

# Parameter grid for SVC
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10]
}

grid_search = GridSearchCV(svc_clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
```

---

## 5. Pipelines

Use pipelines to chain your feature extractor and classifier:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(corpus, y)  # Assuming y are the labels
```

### Parameter Tuning in a Pipeline

```python
param_grid = {
    'tfidf__max_df': [0.8, 1.0],
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(corpus, y)

print("Best Pipeline Params:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

## 6. Tips & Tricks

- **Check Shapes**: `X_train.shape`, `X_test.shape`, `y_train.shape`, `y_test.shape`
- **Check Distribution**: `np.unique(y, return_counts=True)`  
- **Set `random_state`** for reproducibility across experiments.
- **Various `scoring` options** in cross-validation: `'accuracy'`, `'f1_macro'`, `'precision_macro'`, etc.
- **Imbalanced Data**: 
  - Adjust class weights (`class_weight='balanced'`)  
  - Use oversampling/undersampling (e.g., SMOTE in `imblearn`)
 
---

## Metrics

Scikit-learn's `metrics` module offers a comprehensive suite of functions to evaluate the performance of machine learning models. Here's an overview of the key metrics available:

**Classification Metrics:**

- `accuracy_score`: Computes the accuracy classification score.
- `roc_auc_score`: Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
- `average_precision_score`: Computes the average precision (AP) from prediction scores.
- `balanced_accuracy_score`: Computes the balanced accuracy, which accounts for class imbalance.
- `brier_score_loss`: Measures the Brier score loss.
- `classification_report`: Generates a text report showing the main classification metrics, including precision, recall, and F1-score.
- `confusion_matrix`: Computes the confusion matrix to evaluate the accuracy of a classification.
- `f1_score`: Computes the F1 score, the harmonic mean of precision and recall.
- `fbeta_score`: Computes the F-beta score, which weights recall more than precision by a factor of beta.
- `hamming_loss`: Calculates the average Hamming loss, the fraction of labels that are incorrectly predicted.
- `jaccard_score`: Computes the Jaccard similarity coefficient score.
- `log_loss`: Computes the logarithmic loss between true labels and predicted probabilities.
- `matthews_corrcoef`: Calculates the Matthews correlation coefficient, a measure of the quality of binary classifications.
- `precision_recall_curve`: Computes precision-recall pairs for different probability thresholds.
- `precision_score`: Calculates the precision, the ratio of true positives to the sum of true positives and false positives.
- `recall_score`: Computes the recall, the ratio of true positives to the sum of true positives and false negatives.
- `roc_curve`: Computes the Receiver Operating Characteristic (ROC) curve.
- `zero_one_loss`: Calculates the zero-one classification loss.

**Regression Metrics:**

- `explained_variance_score`: Measures the explained variance regression score.
- `max_error`: Computes the maximum residual error.
- `mean_absolute_error`: Calculates the mean absolute error regression loss.
- `mean_squared_error`: Computes the mean squared error regression loss.
- `mean_squared_log_error`: Calculates the mean squared logarithmic error regression loss.
- `median_absolute_error`: Computes the median absolute error regression loss.
- `r2_score`: Calculates the R² (coefficient of determination) regression score function.
- `mean_absolute_percentage_error`: Computes the mean absolute percentage error (MAPE) regression loss.
- `mean_pinball_loss`: Calculates the pinball loss for quantile regression.
- `mean_poisson_deviance`: Computes the mean Poisson deviance regression loss.
- `mean_gamma_deviance`: Calculates the mean Gamma deviance regression loss.
- `mean_tweedie_deviance`: Computes the mean Tweedie deviance regression loss.
- `d2_absolute_error_score`: Measures the D² regression score function, representing the fraction of absolute error explained.
- `d2_pinball_score`: Computes the D² regression score function, representing the fraction of pinball loss explained.
- `d2_tweedie_score`: Calculates the D² regression score function, representing the fraction of Tweedie deviance explained.

**Clustering Metrics:**

- `adjusted_mutual_info_score`: Computes the Adjusted Mutual Information between two clusterings.
- `adjusted_rand_score`: Calculates the Rand index adjusted for chance.
- `calinski_harabasz_score`: Computes the Calinski and Harabasz score, also known as the Variance Ratio Criterion.
- `completeness_score`: Measures the completeness metric of a cluster labeling given a ground truth.
- `davies_bouldin_score`: Computes the Davies-Bouldin score, which evaluates the average similarity ratio of each cluster with its most similar cluster.
- `fowlkes_mallows_score`: Measures the similarity of two clusterings of a set of points.
- `homogeneity_score`: Computes the homogeneity metric of a cluster labeling given a ground truth.
- `mutual_info_score`: Calculates the Mutual Information between two clusterings.
- `normalized_mutual_info_score`: Computes the Normalized Mutual Information between two clusterings.
- `rand_score`: Calculates the Rand index, a measure of the similarity between two data clusterings.
- `silhouette_score`: Computes the mean Silhouette Coefficient of all samples, which measures how similar a sample is to its own cluster compared to other clusters.
- `v_measure_score`: Calculates the V-measure cluster labeling given a ground truth.

**Multilabel Ranking Metrics:**

- `coverage_error`: Measures the coverage error, which represents how far we need to go to cover all true labels.
- `label_ranking_average_precision_score`: Computes the ranking-based average precision.
- `label_ranking_loss`: Calculates the ranking loss measure.

**Pairwise Metrics:**

- `pairwise_distances`: Computes the distance matrix from a vector array.
- `cosine_similarity`: Calculates the cosine similarity between samples. 

