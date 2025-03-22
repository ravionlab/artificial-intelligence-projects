# Hierarchical Clustering

## Overview & Introduction
Hierarchical clustering is an unsupervised learning algorithm that creates a hierarchy of clusters by either a bottom-up (agglomerative) or top-down (divisive) approach. It builds a tree-like structure called a dendrogram that represents the arrangement of clusters and their relationships, without requiring a pre-specified number of clusters.

**Role in Unsupervised Learning**:
Hierarchical clustering addresses problems where the underlying data has an inherent hierarchical structure or when the optimal number of clusters is unknown. It provides a complete clustering history and allows for the extraction of any number of clusters post-hoc.

### Historical Context
The concept of hierarchical clustering dates back to the 1950s and was formalized by various researchers including Joe Ward (1963, Ward's method) and Robert Sokal and Charles Michener (1958, UPGMA method). It remains a fundamental technique in data science, bioinformatics, and social network analysis.

---

## Theoretical Foundations

### Conceptual Explanation
Hierarchical clustering constructs a tree-like nested structure of clusters called a dendrogram. In agglomerative (bottom-up) clustering, each data point starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. In divisive (top-down) clustering, all data points start in one cluster, and splits are performed recursively as one moves down the hierarchy.

### Mathematical Formulation
**Distance Between Points**:
Euclidean distance is commonly used:
$$ d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$

**Linkage Methods** (for calculating distances between clusters):
1. **Single linkage** (minimum distance between elements):
$$ d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y) $$

2. **Complete linkage** (maximum distance between elements):
$$ d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y) $$

3. **Average linkage** (average distance between all pairs):
$$ d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y) $$

4. **Ward's linkage** (minimizes variance increase):
$$ d(C_i, C_j) = \sqrt{\frac{|C_i||C_j|}{|C_i|+|C_j|}} ||c_i - c_j||_2 $$
where $c_i$ and $c_j$ are the centroids of clusters $C_i$ and $C_j$.

### Assumptions
1. The appropriate distance metric for the data is known
2. The linkage method aligns with the cluster structure in the data
3. The hierarchical structure is meaningful for the application
4. Clusters can be represented well by recursive pairwise merging or splitting

---

## Algorithm Mechanics

### Step-by-Step Process (Agglomerative)
1. **Initialization**: Assign each data point to its own cluster.
2. **Distance Calculation**: Compute a distance matrix between all pairs of clusters.
3. **Merging**: Combine the two closest clusters according to the selected linkage criterion.
4. **Update Distances**: Recalculate distances between the new cluster and all other clusters.
5. **Iteration**: Repeat steps 3-4 until all points belong to a single cluster.
6. **Dendrogram Construction**: Represent the hierarchical structure as a tree.
7. **Cluster Extraction**: Cut the dendrogram at a certain level to obtain the desired number of clusters.

### Implementation Workflow
```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Create and fit the model
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = hierarchical.fit_predict(X)

# Create dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()
```

---

## Implementation Details

### Code Structure
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.dendrogram_ = []
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        # Initialize each point as a cluster
        self.labels_ = np.arange(n_samples)
        clusters = [{i} for i in range(n_samples)]
        
        # Compute initial distance matrix
        distances = squareform(pdist(X))
        
        # Main loop: merge clusters until we have n_clusters
        current_n_clusters = n_samples
        while current_n_clusters > self.n_clusters:
            # Find the closest pair of clusters
            i, j = np.unravel_index(np.argmin(distances + np.eye(len(distances)) * np.inf), 
                                    distances.shape)
            
            # Record merge in dendrogram
            self.dendrogram_.append({
                'clusters': (i, j),
                'distance': distances[i, j],
                'n_samples': len(clusters[i]) + len(clusters[j])
            })
            
            # Merge clusters
            clusters[i] = clusters[i].union(clusters[j])
            clusters.pop(j)
            
            # Update labels
            for idx in clusters[i]:
                self.labels_[idx] = i
                
            # Update distance matrix using specified linkage
            self._update_distances(distances, i, j, clusters, X)
            
            # Remove the row and column corresponding to cluster j
            distances = np.delete(np.delete(distances, j, axis=0), j, axis=1)
            
            current_n_clusters -= 1
            
        # Re-label clusters from 0 to n_clusters-1
        unique_labels = np.unique(self.labels_)
        for i, label in enumerate(unique_labels):
            self.labels_[self.labels_ == label] = i
            
        return self
        
    def _update_distances(self, distances, i, j, clusters, X):
        # Update distances based on linkage method
        if self.linkage == 'single':
            # Single linkage: minimum distance
            pass  # Implementation details omitted for brevity
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance
            pass  # Implementation details omitted for brevity
        elif self.linkage == 'average':
            # Average linkage: average distance
            pass  # Implementation details omitted for brevity
        elif self.linkage == 'ward':
            # Ward linkage: minimum variance increase
            pass  # Implementation details omitted for brevity
```

### Setup Instructions
```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Hyperparameters & Optimization

### Key Parameters
- **n_clusters**: Number of clusters to extract from the dendrogram.
- **affinity**: Distance metric ('euclidean', 'manhattan', 'cosine', etc.).
- **linkage**: Method for calculating distances between clusters ('ward', 'complete', 'average', 'single').
- **distance_threshold**: The threshold to apply when forming clusters (alternative to n_clusters).

### Tuning Strategies
- **Dendrogram Analysis**: Visually inspect the dendrogram to identify natural cluster boundaries.
- **Cophenetic Correlation**: Measure how well the hierarchical clustering preserves pairwise distances.
- **Silhouette Analysis**: Calculate silhouette scores for different cuts of the dendrogram.
- **Inconsistency Method**: Look for inconsistent links in the dendrogram.

---

## Evaluation Metrics
- **Cophenetic Correlation Coefficient**: Correlation between original distances and dendrogram distances.
- **Silhouette Score**: Measures how similar objects are to their own cluster compared to other clusters.
- **Davies-Bouldin Index**: Average similarity between clusters; lower values indicate better clustering.
- **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion.

---

## Practical Examples
**Dataset**: Genomic data, customer segmentation, document clustering.
**Use Case**: Taxonomical classification of species based on genetic markers.

```python
# Example: Hierarchical clustering on gene expression data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y = make_blobs(n_samples=50, centers=3, random_state=42)

# Perform hierarchical clustering
linked = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 7))
dendrogram(linked, 
           leaf_rotation=90,
           leaf_font_size=8,
           above_threshold_color='grey')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.axhline(y=20, color='r', linestyle='--')
plt.show()

# Visualize clusters
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title('Hierarchical Clustering Results')
plt.show()
```

---

## Advanced Theory

### Divisive Hierarchical Clustering
While agglomerative clustering builds the tree bottom-up, divisive clustering works top-down:
1. Start with all points in a single cluster
2. Recursively split the cluster with the largest diameter
3. Use a flat clustering algorithm (e.g., k-means) for the splitting
4. Continue until each cluster contains only one point or a stopping criterion is reached

### Lance-Williams Algorithm
A generalized recurrence formula for updating distances during hierarchical clustering:
$$ d(C_i \cup C_j, C_k) = \alpha_i d(C_i, C_k) + \alpha_j d(C_j, C_k) + \beta d(C_i, C_j) + \gamma |d(C_i, C_k) - d(C_j, C_k)| $$
Different values of $\alpha$, $\beta$, and $\gamma$ correspond to different linkage methods.

---

## Advantages & Limitations

**Pros**:
- No need to specify number of clusters in advance
- Produces a dendrogram for intuitive visualization
- Can uncover hierarchical relationships in data
- Provides multiple levels of granularity
- More deterministic than methods like k-means (no random initialization)

**Cons**:
- Computationally expensive (O(n²log n) for most efficient implementations)
- Sensitive to noise and outliers
- Cannot correct poor merging decisions once made
- May not scale well to very large datasets
- Different linkage criteria can lead to significantly different results

---

## Further Reading
1. Murtagh, F., & Contreras, P. (2012). "Algorithms for hierarchical clustering: an overview".
2. Ward Jr, J. H. (1963). "Hierarchical grouping to optimize an objective function".
3. Müllner, D. (2011). "Modern hierarchical, agglomerative clustering algorithms".
4. Rokach, L., & Maimon, O. (2005). "Clustering methods".
