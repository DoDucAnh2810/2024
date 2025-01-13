### 0. **`ttest_ind`**
   - **Paired t-test**: Used to compare means of two independent groups.
   - **Example**:
     ```python
    from scipy.stats import ttest_ind
    
    # Sample data
    data1 = [10, 12, 9, 11, 10]
    data2 = [8, 7, 6, 9, 10]
    
    # Perform a two-sample t-test (assuming equal variances)
    t_stat, p_value = ttest_ind(data1, data2)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    
    # Perform a two-sample t-test (Welch’s t-test, unequal variances)
    t_stat_welch, p_value_welch = ttest_ind(data1, data2, equal_var=False)
    print(f"T-statistic (Welch): {t_stat_welch}, P-value: {p_value_welch}")
     ```

---

### 1. **`ttest_rel`**
   - **Paired t-test**: Used when the two samples are related (e.g., measurements before and after treatment on the same subjects).
   - **Example**:
     ```python
     from scipy.stats import ttest_rel

     before = [80, 85, 88, 90, 78]
     after = [82, 87, 90, 92, 79]

     t_stat, p_value = ttest_rel(before, after)
     print(f"T-statistic: {t_stat}, P-value: {p_value}")
     ```

---

### 2. **`mannwhitneyu`**
   - **Mann-Whitney U test**: A non-parametric test to compare medians of two independent samples. Used when data is not normally distributed.
   - **Example**:
     ```python
     from scipy.stats import mannwhitneyu

     group1 = [7, 8, 9, 6, 10]
     group2 = [5, 6, 5, 7, 4]

     u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
     print(f"U-statistic: {u_stat}, P-value: {p_value}")
     ```

---

### 3. **`wilcoxon`**
   - **Wilcoxon signed-rank test**: A non-parametric test for paired data (e.g., before and after measurements). It's an alternative to the paired t-test.
   - **Example**:
     ```python
     from scipy.stats import wilcoxon

     before = [80, 85, 88, 90, 78]
     after = [82, 87, 90, 92, 79]

     stat, p_value = wilcoxon(before, after)
     print(f"Wilcoxon statistic: {stat}, P-value: {p_value}")
     ```

---

### 4. **`f_oneway`**
   - **One-way ANOVA**: Used to compare the means of three or more independent groups.
   - **Example**:
     ```python
     from scipy.stats import f_oneway

     group1 = [7, 8, 6, 9, 10]
     group2 = [5, 6, 7, 5, 8]
     group3 = [9, 10, 8, 7, 9]

     f_stat, p_value = f_oneway(group1, group2, group3)
     print(f"F-statistic: {f_stat}, P-value: {p_value}")
     ```

---

### 5. **`chi2_contingency`**
   - **Chi-squared test**: Used for testing independence in categorical data (e.g., contingency tables).
   - **Example**:
     ```python
     from scipy.stats import chi2_contingency

     contingency_table = [[10, 20], [20, 30]]
     chi2, p_value, dof, expected = chi2_contingency(contingency_table)
     print(f"Chi2: {chi2}, P-value: {p_value}, Degrees of freedom: {dof}")
     ```

---

### 6. **`ks_2samp`**
   - **Kolmogorov-Smirnov test**: A non-parametric test to compare two distributions.
   - **Example**:
     ```python
     from scipy.stats import ks_2samp

     data1 = [1, 2, 3, 4, 5]
     data2 = [3, 4, 5, 6, 7]

     stat, p_value = ks_2samp(data1, data2)
     print(f"KS Statistic: {stat}, P-value: {p_value}")
     ```

---

### 7. **`levene`**
   - **Levene’s test**: Used to test the equality of variances across groups.
   - **Example**:
     ```python
     from scipy.stats import levene

     group1 = [7, 8, 9, 6, 10]
     group2 = [5, 6, 7, 5, 8]

     stat, p_value = levene(group1, group2)
     print(f"Levene statistic: {stat}, P-value: {p_value}")
     ```

---

### 8. **`shapiro`**
   - **Shapiro-Wilk test**: Used to test whether a dataset is normally distributed.
   - **Example**:
     ```python
     from scipy.stats import shapiro

     data = [7, 8, 9, 10, 11]
     stat, p_value = shapiro(data)
     print(f"Shapiro-Wilk statistic: {stat}, P-value: {p_value}")
     ```

---

### 9. **`pearsonr`**
   - **Pearson correlation coefficient**: Measures the linear relationship between two continuous variables.
   - **Example**:
     ```python
     from scipy.stats import pearsonr

     x = [1, 2, 3, 4, 5]
     y = [2, 4, 6, 8, 10]

     corr, p_value = pearsonr(x, y)
     print(f"Pearson correlation: {corr}, P-value: {p_value}")
     ```

---

### 10. **`spearmanr`**
   - **Spearman rank correlation**: Measures the monotonic relationship between two variables (non-parametric).
   - **Example**:
     ```python
     from scipy.stats import spearmanr

     x = [1, 2, 3, 4, 5]
     y = [2, 4, 6, 8, 10]

     corr, p_value = spearmanr(x, y)
     print(f"Spearman correlation: {corr}, P-value: {p_value}")
     ```

---

### Summary Table:
| **Function**         | **Test Type**                           | **Use Case**                              |
|-----------------------|-----------------------------------------|-------------------------------------------|
| `ttest_ind`          | Two-sample t-test                      | Compare means of two independent groups   |
| `ttest_rel`          | Paired t-test                          | Compare means of paired data              |
| `mannwhitneyu`       | Mann-Whitney U test (non-parametric)   | Compare medians of two independent groups |
| `wilcoxon`           | Wilcoxon signed-rank test (non-param.) | Compare medians of paired data            |
| `f_oneway`           | One-way ANOVA                          | Compare means of 3+ groups                |
| `chi2_contingency`   | Chi-squared test                       | Test independence in categorical data     |
| `ks_2samp`           | Kolmogorov-Smirnov test                | Compare two distributions                 |
| `levene`             | Levene’s test                          | Test equality of variances                |
| `shapiro`            | Shapiro-Wilk test                      | Test normality                            |
| `pearsonr`           | Pearson correlation                    | Test linear relationship                  |
| `spearmanr`          | Spearman rank correlation              | Test monotonic relationship               |

By understanding these functions, you can perform a wide range of statistical analyses for various data scenarios.
