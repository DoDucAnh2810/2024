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


### 10. Digitize

```python
numpy.digitize(x, bins, right=False)
```

#### Parameters:

1. **x**:
   - Array-like input data to be binned.
   
2. **bins**:
   - Array of bin edges. Must be 1-dimensional and monotonically increasing or decreasing.

3. **right** (default: `False`):
   - If `False`, the intervals are left-inclusive (e.g., `[bins[i-1], bins[i])`).
   - If `True`, the intervals are right-inclusive (e.g., `(bins[i-1], bins[i]]`).

#### Returns:

- An array of integers indicating the index of the bin to which each value in `x` belongs.
  - Index starts at 1 for the first bin.
  - Values smaller than the first bin edge return `0`.
  - Values larger than the last bin edge return `len(bins)`.

---

#### Example Usage:

##### 1. Basic Example:
```python
import numpy as np

x = [0.2, 6.4, 3.0, 1.6, 10.0]  # Data to bin
bins = [0, 2, 4, 6, 8]          # Bin edges

result = np.digitize(x, bins)
print(result)  # Output: [1 4 2 1 5]
```

Explanation:
- `0.2` belongs to bin 1: `[0, 2)`
- `6.4` belongs to bin 4: `[6, 8)`
- `3.0` belongs to bin 2: `[2, 4)`
- `1.6` belongs to bin 1: `[0, 2)`
- `10.0` is greater than the last bin, so it’s assigned index `5`.

---

##### 2. Using `right=True`:
```python
result = np.digitize(x, bins, right=True)
print(result)  # Output: [0 4 2 1 5]
```

Explanation:
- With `right=True`, the intervals are `(bins[i-1], bins[i]]`.
- `0.2` is less than the first bin (`0`), so it’s `0`.

---

##### 3. Descending Bins:
If the bins are in descending order, you need to reverse the order to ensure the bin edges are monotonic.

```python
bins_desc = [8, 6, 4, 2, 0]
result = np.digitize(x, bins_desc)
print(result)  # Output is based on ascending interpretation of bins.
```

---

### Array Split

```python
import numpy as np

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parts = np.array_split(my_list, 5)

# Convert the NumPy arrays back to lists (optional)
parts = [list(part) for part in parts]
print(parts)
```

