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

### 9. Cumulative Sum
```python
numpy.cumsum(a, axis=None, dtype=None, out=None)
```

#### Parameters
1. **`a` (array-like):**  
   The input array for which the cumulative sum is computed.
   
2. **`axis` (int or None, optional):**  
   The axis along which the cumulative sum is performed.  
   - If `axis=0`, the cumulative sum is computed along columns (for 2D arrays).
   - If `axis=1`, it is computed along rows.
   - If `axis=None`, the array is flattened, and the cumulative sum is computed over all elements.

3. **`dtype` (dtype, optional):**  
   The type of the returned array. By default, it is the same as the type of the input array, but you can specify a different type.

4. **`out` (ndarray, optional):**  
   An alternative output array where the result will be stored. It must have the same shape as the expected output.

#### Return Value
An array of the same shape as the input, containing the cumulative sums.

#### Example Usage
##### 1. Cumulative sum over a 1D array
```python
import numpy as np

arr = np.array([1, 2, 3, 4])
cumsum = np.cumsum(arr)
print(cumsum)  # Output: [ 1  3  6 10 ]
```

##### 2. Cumulative sum over a 2D array
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Cumulative sum along rows (axis=0)
cumsum_axis0 = np.cumsum(arr_2d, axis=0)
print(cumsum_axis0)
# Output:
# [[ 1  2  3]
#  [ 5  7  9]]

# Cumulative sum along columns (axis=1)
cumsum_axis1 = np.cumsum(arr_2d, axis=1)
print(cumsum_axis1)
# Output:
# [[ 1  3  6]
#  [ 4  9 15]]
```

##### 3. Flattened cumulative sum (`axis=None`)
```python
flattened_cumsum = np.cumsum(arr_2d, axis=None)
print(flattened_cumsum)
# Output: [ 1  3  6 10 15 21 ]
```

##### 4. Specifying `dtype`
```python
arr = np.array([1, 2, 3, 4], dtype=np.int32)
cumsum_dtype = np.cumsum(arr, dtype=np.float64)
print(cumsum_dtype)  # Output: [ 1.  3.  6. 10.]
```

#### Key Points
- `np.cumsum` is useful for computing running totals or cumulative metrics.
- It works on arrays of any shape and can compute along any axis.
- The cumulative sum includes the current element in its computation.

