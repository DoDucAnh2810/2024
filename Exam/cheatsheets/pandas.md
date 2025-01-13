## Creating DataFrames

### From a List of Lists

```python
import pandas as pd

data = [
    [1, 'Alice', 25],
    [2, 'Bob', 30],
    [3, 'Charlie', 35]
]
df = pd.DataFrame(data, columns=['ID', 'Name', 'Age'])
```

### From a Dictionary of Lists

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}
df = pd.DataFrame(data)
```

### From a CSV or Excel File

```python
import pandas as pd

# Load data from a CSV file
df = pd.read_csv('data.csv')

# Load data from an Excel file
df = pd.read_excel('data.xlsx')
```

### From JSON

```python
import pandas as pd

json_data = '''
[
    {"ID": 1, "Name": "Alice", "Age": 25},
    {"ID": 2, "Name": "Bob", "Age": 30}
]
'''
df = pd.read_json(json_data, lines=False)
```

---

## Viewing Data

```python
df.head()       # First 5 rows
df.tail()       # Last 5 rows
df.info()       # Summary of the DataFrame
df.describe()   # Statistical summary
```

---

## Selecting and Filtering Data

```python
df['Name']                # Select a column
df[['Name', 'Age']]       # Select multiple columns
df.iloc[0]                # Select the first row (by position)
df.loc[0]                 # Select the first row (by index)
df[df['Age'] > 30]        # Filter rows where Age > 30
```

---

## Modifying Data

```python
df['Age'] = df['Age'] + 1                      # Update a column
df['Gender'] = ['F', 'M', 'M']                 # Add a new column
df.rename(columns={'Age': 'Years'}, inplace=True)  # Rename a column
df.replace({'x': 'a', 2: 20}, inplace=True)
```

---

## Handling Missing Data

```python
df.dropna()               # Drop rows with missing values
df.fillna(0)              # Fill missing values with 0
df.isnull().sum()         # Check for missing values
```

---

## Sorting and Ranking

```python
df.sort_values(by='Age')                   # Sort by Age
df.sort_values(by='Age', ascending=False)  # Sort by Age in descending order
df.rank()                                  # Rank data
```

---

## Aggregation and Grouping

```python
df.groupby('Gender')['Age'].mean()                      # Group by Gender and calculate mean Age
df.groupby('Gender').agg({'Age': ['mean', 'sum']})      # Aggregate multiple functions
```

---

## Merging and Joining

```python
import pandas as pd

df1 = pd.DataFrame({
    'ID': [1, 2],
    'Name': ['Alice', 'Bob']
})
df2 = pd.DataFrame({
    'ID': [1, 2],
    'Score': [90, 80]
})
merged = pd.merge(df1, df2, on='ID')  # Merge on ID
```

---

## Advanced Functions

```python
df.apply(lambda x: x + 1)                             # Apply a function element-wise
df['Age'].map(lambda x: x * 2)                        # Apply a function to a column
df.pivot(index='Name', columns='Gender', values='Age')  # Pivot table
```

---

## Useful Attributes

```python
df.shape     # Dimensions of the DataFrame
df.columns   # Column names
df.index     # Row indices
```
