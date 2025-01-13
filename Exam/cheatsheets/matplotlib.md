## Plot

```python
in_degrees = np.array(sorted([degree for node, degree in G.in_degree()], reverse=True))
out_degrees = np.array(sorted([degree for node, degree in G.out_degree()], reverse=True))

in_degrees_frac_cum = np.cumsum(in_degrees / len(G.edges()))
out_degrees_frac_cum = np.cumsum(out_degrees / len(G.edges()))

plt.figure(figsize=(10,6))

plt.plot(in_degrees_frac_cum, label='Fraction of researchers hired by top N hiring univeristies')
plt.plot(out_degrees_frac_cum, label='Fraction of researchers trained by top N hired univeristies')

plt.legend()
plt.xlabel('N')
plt.ylabel('Fraction')
plt.yscale('linear')
plt.title('Question 2 Graph')

plt.tight_layout()
plt.plot()
```

## Subplots

```python
import matplotlib.pyplot as plt
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Create a figure and a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# 1st subplot: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), label='sin(x)')
axes[0, 0].plot(x, np.cos(x), label='cos(x)', linestyle='--')
axes[0, 0].set_title("Line Plot")
axes[0, 0].set_xlabel("X-axis")
axes[0, 0].set_ylabel("Y-axis")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2nd subplot: Scatter plot
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)
scatter = axes[0, 1].scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='viridis')
axes[0, 1].set_title("Scatter Plot")
axes[0, 1].set_xlabel("X-axis")
axes[0, 1].set_ylabel("Y-axis")
fig.colorbar(scatter, ax=axes[0, 1], orientation='vertical')

# 3rd subplot: Bar plot
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 8, 5]
axes[1, 0].bar(categories, values, color=['blue', 'green', 'red', 'purple'])
axes[1, 0].set_title("Bar Plot")
axes[1, 0].set_xlabel("Categories")
axes[1, 0].set_ylabel("Values")

# 4th subplot: Heatmap
heatmap_data = np.random.rand(10, 10)
heatmap = axes[1, 1].imshow(heatmap_data, cmap='coolwarm', interpolation='nearest')
axes[1, 1].set_title("Heatmap")
fig.colorbar(heatmap, ax=axes[1, 1], orientation='horizontal')

# Set the overall title
fig.suptitle("2x2 Subplot Example with Matplotlib", fontsize=16)

# Show the plots
plt.show()
```
