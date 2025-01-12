# NetworkX Graph Guide

## Graph Types

1. **`nx.Graph`**
2. **`nx.DiGraph`**
3. **`nx.MultiGraph`**
4. **`nx.MultiDiGraph`**

---

## Subgraph

*Content for Subgraph can be added here.*

---

## Adding Edges

### 1. Edgelist

```python
import networkx as nx

edges = [(1, 2), (2, 3), (3, 1)]
G = nx.Graph(edges)
```

### 2. Adjacency Dictionary

```python
import networkx as nx

adj_dict = {
    1: [2, 3],
    2: [3],
    3: [1]
}
G = nx.Graph(adj_dict)
```

### 3. DataFrame Edgelist

```python
import pandas as pd
import networkx as nx

df = pd.DataFrame({
    'source': [1, 2, 3],
    'target': [2, 3, 1],
    'weight': [4.5, 3.2, 1.8]
})
G = nx.from_pandas_edgelist(
    df,
    source='source',
    target='target',
    edge_attr='weight',
    create_using=nx.Graph()
)
```

### 4. NumPy Adjacency Matrix

```python
import numpy as np
import networkx as nx

matrix = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])
G = nx.from_numpy_matrix(matrix)
```

---

## Degree

### 1. Undirected Graph

```python
degree = dict(G.degree())
```

### 2. Directed Graph

```python
DG = nx.DiGraph(edges)

in_degree = dict(DG.in_degree())
out_degree = dict(DG.out_degree())
```

---

## Degree Centrality

### 1. Undirected Graph

```python
import networkx as nx

degree_centrality = nx.degree_centrality(G)  # Dictionary of degree centrality
```

### 2. Directed Graph

```python
import networkx as nx

in_degree_centrality = nx.in_degree_centrality(DG)
out_degree_centrality = nx.out_degree_centrality(DG)
```

---

## PageRank

```python
import networkx as nx

page_rank = nx.pagerank(DG)  # Dictionary of PageRank values
```

---

## Node Attributes

### 1. Assign Attributes to Nodes

```python
import networkx as nx

values = {1: 'A', 2: 'B', 3: 'C'}
nx.set_node_attributes(G, values, name='attribute_name')
```

### 2. Retrieve the Attribute Dictionary for All Nodes

```python
import networkx as nx

attributes = nx.get_node_attributes(G, 'attribute_name')  # Dictionary of node attributes
```

### 3. Access Attributes for a Specific Node

```python
# Access all attributes of node n
node_attributes = G.nodes[n]

# Access a specific attribute of node n
specific_attribute = G.nodes[n]['attr_name']
```

### 4. Iterate Through Nodes with Their Attribute Dictionaries

```python
import networkx as nx

for node, attr in G.nodes(data=True):
    print(node, attr)
```

---

## Edge Attributes

### 1. Assign Attributes to Edges

```python
import networkx as nx

values = {
    (1, 2): {'weight': 4.5},
    (2, 3): {'weight': 3.2},
    (3, 1): {'weight': 1.8}
}
nx.set_edge_attributes(G, values)
```

### 2. Retrieve the Attribute Dictionary for All Edges

```python
import networkx as nx

edge_attributes = nx.get_edge_attributes(G, 'weight')  # Dictionary of edge attributes
```

### 3. Access Attributes for a Specific Edge

```python
# Access all attributes between nodes u and v
edge_attrs = G[u][v]

# Access a specific attribute between nodes u and v
specific_attr = G[u][v]['attr_name']
```

### 4. Iterate Through Edges with Their Attribute Dictionaries

```python
import networkx as nx

for u, v, attr in G.edges(data=True):
    print(u, v, attr)
```
