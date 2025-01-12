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
