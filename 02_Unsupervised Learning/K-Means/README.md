# K-means Clustering

## Overview & Introduction
K-means clustering is a fundamental unsupervised learning algorithm used to partition a dataset into K distinct, non-overlapping subgroups (clusters) where each data point belongs to the cluster with the nearest mean. It aims to minimize the variance within each cluster, organizing data into natural groupings without labeled training data.

**Role in Unsupervised Learning**:
K-means serves as a cornerstone clustering technique used for discovering hidden patterns in unlabeled data. It excels at identifying groups with similar characteristics when the number of groups is known in advance.

### Historical Context
Developed in 1957 by Stuart Lloyd as a technique for pulse-code modulation, though published outside of Bell Labs only in 1982. The term "k-means" was first used by James MacQueen in 1967. It remains widely used today due to its simplicity, efficiency, and empirical success across various domains.

---

## Theoretical Foundations

### Conceptual Explanation
K-means divides observations into k clusters where each observation belongs to the cluster with the nearest centroid, serving as a prototype of the cluster. The algorithm iteratively assigns points to the nearest cluster center and then recalculates those centers until convergence.

### Mathematical Formulation
**Objective Function**: The goal is to minimize the within-cluster sum of squares (WCSS):
$$ J = \sum_{j=1}^{k} \sum_{i=1}^{n} ||x_i^{(j)} - c_j||^2 $$
where:
- $x_i^{(j)}$ is the i-th data point belonging to the j-th cluster
- $c_j$ is the centroid of the j-th cluster
- $||x_i^{(j)} - c_j||^2$ is the squared Euclidean distance between $x_i^{(j)}$ and $c_j$

**Cluster Assignment**: Each data point is assigned to the nearest centroid:
$$ \text{cluster}(x_i) = \arg\min_j ||x_i - c_j||^2 $$

**Centroid Update**: Each centroid is updated to be the mean of all points assigned to its cluster:
$$ c_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i $$
where $|C_j|$ is the number of points in cluster $j$.

### Assumptions
1. Clusters are spherical (isotropic)
2. Clusters have similar size and density
3. The variance of the distribution of each attribute is similar
4. The prior probability of each cluster is equal

---

## Algorithm Mechanics

### Step-by-Step Process
1. **Initialization**: Randomly select K data points as initial centroids.
2. **Assignment Step**: Assign each data point to the nearest centroid, forming K clusters.
3. **Update Step**: Recalculate centroids of new clusters.
4. **Iteration**: Repeat steps 2-3 until convergence criteria are met (centroids no longer move significantly or maximum iterations reached).
5. **Final Clustering**: Return the final cluster assignments and centroids.

### Training & Implementation Workflow
```python
from sklearn.cluster import KMeans

# Create and train the model
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# Get cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```

---

## Implementation Details

### Code Structure
```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign clusters
            clusters = self._assign_clusters(X)
            
            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()
            
            # Update centroids
            for i in range(self.k):
                if np.sum(clusters == i) > 0:  # Avoid empty clusters
                    self.centroids[i] = np.mean(X[clusters == i], axis=0)
            
            # Check for convergence
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break
                
        return self
        
    def _assign_clusters(self, X):
        # Calculate distances between each point and all centroids
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        return self._assign_clusters(X)
```

### Setup Instructions
```bash
pip install numpy scikit-learn matplotlib
```

---

## Hyperparameters & Optimization

### Key Parameters
- **n_clusters (k)**: Number of clusters (default: 8).
- **init**: Method for initialization ('random', 'k-means++', or array of initial centroids).
- **max_iter**: Maximum number of iterations (default: 300).
- **tol**: Tolerance for declaring convergence (default: 1e-4).
- **n_init**: Number of times to run with different initializations (default: 10).

### Tuning Strategies
- **Elbow Method**: Plot WCSS against different k values and look for the "elbow" point.
- **Silhouette Analysis**: Calculate silhouette scores to evaluate cluster quality.
- **Gap Statistic**: Compare intra-cluster dispersion to expected dispersion under null reference.
- **Using k-means++**: An improved initialization method that spreads initial centroids.

---

## Evaluation Metrics
- **Inertia (WCSS)**: Sum of squared distances to closest centroid; lower values indicate better clustering.
- **Silhouette Score**: Measures how similar objects are to their cluster compared to other clusters (-1 to 1).
- **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion.
- **Davies-Bouldin Index**: Average similarity between clusters; lower values indicate better clustering.

---

## Practical Examples
**Dataset**: Customer segmentation, image compression, document clustering.
**Use Case**: Segmenting customers based on purchasing behavior for targeted marketing campaigns.

```python
# Customer segmentation example
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming X contains customer features (purchase frequency, spending amount, etc.)
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

# Visualize results (for 2D data)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='red')
plt.title('Customer Segments')
plt.show()
```

---

## Advanced Theory

### K-means++
An initialization strategy that selects initial centroids to be distant from each other, improving both speed and quality:
1. Choose first centroid randomly from data points
2. For each subsequent centroid, select a point with probability proportional to its squared distance from the nearest existing centroid

### Mini-batch K-means
A variant that uses mini-batches to reduce computation time while maintaining reasonable quality:
$$ J_{MB} = \sum_{x_i \in MB} \min_j ||x_i - c_j||^2 $$
where MB is a randomly selected subset of the training data.

---

## Advantages & Limitations

**Pros**:
- Computationally efficient (O(nkdi) where n = samples, k = clusters, d = dimensions, i = iterations)
- Easy to implement and interpret
- Scales well to large datasets
- Generalizes to clusters of different shapes and sizes with kernel variants

**Cons**:
- Requires specifying k beforehand
- Sensitive to initial centroid selection
- Converges to local optima
- Struggles with non-globular clusters
- Performance degrades in high dimensions
- Sensitive to outliers and skewed data

---

## Further Reading
1. MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations".
2. Arthur, D. and Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding".
3. Sculley, D. (2010). "Web-scale k-means clustering".
4. Hartigan, J. A., & Wong, M. A. (1979). "Algorithm AS 136: A k-means clustering algorithm".
