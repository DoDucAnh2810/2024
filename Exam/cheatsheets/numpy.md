# NumPy Statistical Functions Cheatsheet

## Descriptive Statistics

### 1. Mean
- **Function**: `np.mean(array)`
- **Description**: Calculates the arithmetic mean of the array.
- **Example**:
  ```python
  np.mean([10, 12, 9])  # Output: 10.33
  ```

### 2. Median
- **Function**: `np.median(array)`
- **Description**: Finds the middle value of the sorted array.
- **Example**:
  ```python
  np.median([10, 12, 9])  # Output: 10
  ```

### 3. Standard Deviation
- **Function**: `np.std(array, ddof=1)`
- **Description**: Calculates the standard deviation. Use `ddof=1` for sample standard deviation.
- **Example**:
  ```python
  np.std([10, 12, 9], ddof=1)  # Output: 1.5275
  ```

### 4. Variance
- **Function**: `np.var(array, ddof=1)`
- **Description**: Calculates the variance. Use `ddof=1` for sample variance.
- **Example**:
  ```python
  np.var([10, 12, 9], ddof=1)  # Output: 2.3333
  ```

### 5. Minimum Value
- **Function**: `np.min(array)`
- **Description**: Returns the smallest value in the array.
- **Example**:
  ```python
  np.min([10, 12, 9])  # Output: 9
  ```

### 6. Maximum Value
- **Function**: `np.max(array)`
- **Description**: Returns the largest value in the array.
- **Example**:
  ```python
  np.max([10, 12, 9])  # Output: 12
  ```

### 7. Range (Peak-to-Peak)
- **Function**: `np.ptp(array)`
- **Description**: Calculates the range (difference between the max and min values).
- **Example**:
  ```python
  np.ptp([10, 12, 9])  # Output: 3
  ```

### 8. Percentile
- **Function**: `np.percentile(array, q)`
- **Description**: Finds the value below which a given percentage of data falls.
  - `q`: Percentile value (e.g., 25 for the 25th percentile).
- **Example**:
  ```python
  np.percentile([10, 12, 9], 25)  # Output: 9.75
  np.percentile([10, 12, 9], 75)  # Output: 11.5
  ```
