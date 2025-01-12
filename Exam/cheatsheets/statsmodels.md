## 1. Creating Example Data

Here’s a simple dataset with two predictors (`X1`, `X2`) and a dependent variable (`Y`):

```python
np.random.seed(42)
n = 100

X1 = np.random.normal(10, 2, size=n)
X2 = np.random.normal(5, 1, size=n)
Y = 3 + 0.5 * X1 - 0.7 * X2 + np.random.normal(0, 1, size=n)

df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'Y': Y
})

df.head()
```

---

## 2. Ordinary Least Squares (OLS)

### 2.1 Fitting an OLS Model

```python
model_ols = smf.ols(formula="Y ~ X1 + X2", data=df)
results_ols = model_ols.fit()

# View model summary
print(results_ols.summary())
```

- **Formula Syntax**: 
  - `"Y ~ X1 + X2"` means `Y` is modeled as a function of `X1` and `X2`.
  - Use `+`, `-`, `:` for interaction terms, e.g. `"Y ~ X1 * X2"` for full interaction.

---

## 3. Generalized Linear Models (GLM)

### 3.1 Fitting a GLM (e.g., Poisson)

```python
model_glm = smf.glm(formula="Y ~ X1 + X2", 
                    data=df, 
                    family=sm.families.Poisson())
results_glm = model_glm.fit()

print(results_glm.summary())
```

- **Common Families**:
  - `sm.families.Poisson()`
  - `sm.families.Binomial()`
  - `sm.families.Gamma()`
  - `sm.families.Gaussian()`

---

## 4. Logistic Regression

To do logistic regression, specify **Binomial** family in `glm` or use `logit`:

### 4.1 Using `glm` with Binomial family

```python
# Assume df has a binary outcome 'Y_binary' with values {0, 1}
model_logistic_glm = smf.glm(formula="Y_binary ~ X1 + X2", 
                             data=df,
                             family=sm.families.Binomial())
results_logistic_glm = model_logistic_glm.fit()

print(results_logistic_glm.summary())
```

### 4.2 Using `logit`

```python
model_logistic = smf.logit(formula="Y_binary ~ X1 + X2", data=df)
results_logistic = model_logistic.fit()

print(results_logistic.summary())
```

---

## 5. Model Summaries & Diagnostics

### 5.1 Summaries

```python
# OLS example
print(results_ols.summary())
```
- Provides coefficient estimates, p-values, confidence intervals, R-squared, etc.

### 5.2 Residuals

```python
residuals = results_ols.resid
fitted = results_ols.fittedvalues

# Plot residuals
sns.residplot(x=fitted, y=df['Y'], lowess=True, line_kws={'color': 'red'})
plt.show()
```

### 5.3 Durbin-Watson Test (Autocorrelation)

```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)
print("Durbin-Watson statistic:", dw_stat)
```

### 5.4 Check Normality of Residuals

```python
fig = sm.qqplot(residuals, line='45', fit=True)
plt.show()
```

---

## 6. ANOVA and Other Post-hoc Analyses

- **ANOVA** in `statsmodels` can compare nested models or factors:

```python
# Example: One-way ANOVA using formula
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Suppose 'Group' is a categorical variable in df
model_aov = ols('Y ~ C(Group)', data=df).fit()
anova_table = sm.stats.anova_lm(model_aov, typ=2)
print(anova_table)
```

- **Post-hoc tests** often require external libraries or custom code (e.g., `statsmodels.stats.multicomp`).

---

## 7. Predictions

### 7.1 In-sample Predictions

```python
pred_in_sample = results_ols.predict(df)
```

### 7.2 Out-of-sample Predictions

```python
# New data
new_data = pd.DataFrame({
    'X1': [12, 8, 10],
    'X2': [4, 6, 5]
})

pred_out_sample = results_ols.predict(new_data)
print(pred_out_sample)
```

---

## 8. Tips & Tricks

1. **Interaction Terms**:  
   - `"Y ~ X1 * X2"` expands to `X1 + X2 + X1:X2` (the interaction).
2. **Categorical Variables**:  
   - `C(variable, Treatment('dummny'))` treats `variable` as categorical in formulas.
3. **Model Comparison**:  
   - Use `sm.stats.anova_lm(model1, model2, typ=1 or 2)` to compare nested models.
4. **Transformations**:  
   - Apply transformations in formula strings, e.g., `np.log(X1)` or `(X1)**2`.
5. **Robust Standard Errors**:  
   - `results_ols.get_robustcov_results(cov_type='HC3')` for robust SEs.
