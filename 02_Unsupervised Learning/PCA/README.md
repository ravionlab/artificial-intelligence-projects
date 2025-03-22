# Principal Component Analysis (PCA)

## Overview & Introduction
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It identifies orthogonal axes (principal components) along which the data varies most, allowing for simplified analysis and visualization.

**Role in Unsupervised Learning**:
PCA is a cornerstone technique for unsupervised feature extraction and dimensionality reduction. It helps uncover hidden structures in data by identifying directions of maximum variance without requiring labeled examples.

### Historical Context
Developed by Karl Pearson in 1901 and later independently by Harold Hotelling in 1933. PCA has become a fundamental technique in data analysis, statistics, and machine learning, forming the foundation for many advanced dimension reduction methods.

---

## Theoretical Foundations

### Conceptual Explanation
PCA works by finding new coordinate axes (principal components) that capture the maximum amount of variance in the data. These components are linear combinations of the original features, with each successive component capturing the most variance possible while remaining orthogonal to previous components.

### Mathematical Formulation
**Objective**: Find a set of orthogonal vectors (principal components) that maximize the variance of projected data.

**Eigenvector Representation**:
The principal components are the eigenvectors of the data's covariance matrix:
$$ C = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^T $$
where $\mu$ is the mean of the data.

**Eigenvalue Decomposition**:
$$ C = V \Lambda V^T $$
where:
- $V$ is a matrix whose columns are eigenvectors of $C$
- $\Lambda$ is a diagonal matrix of corresponding eigenvalues

**Dimensionality Reduction**:
To reduce to $k$ dimensions, project data onto the top $k$ eigenvectors:
$$ X_{reduced} = X \cdot V_k $$
where $V_k$ contains the top $k$ eigenvectors (columns of $V$) corresponding to the $k$ largest eigenvalues.

### Assumptions
1. Linearity: Assumes relationships between variables are linear
2. Large variances represent important structures
3. Principal components are orthogonal
4. Data is centered (zero mean)

---

## Algorithm Mechanics

### Step-by-Step Process
1. **Standardization**: Center the data (subtract mean) and optionally scale to unit variance.
2. **Covariance Calculation**: Compute the covariance matrix of the standardized data.
3. **Eigendecomposition**: Calculate eigenvalues and eigenvectors of the covariance matrix.
4. **Sorting**: Sort eigenvectors by decreasing eigenvalues.
5. **Component Selection**: Choose top $k$ eigenvectors to form the new feature space.
6. **Projection**: Transform original data onto the new $k$-dimensional space.

### Implementation Workflow
```python
from sklearn.decomposition import PCA

# Create and fit the PCA model
pca = PCA(n_components=2)
pca.fit(X)

# Transform the data to the new space
X_reduced = pca.transform(X)
```

---

## Implementation Details

### Code Structure
```python
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store the first n_components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        return self
        
    def transform(self, X):
        # Center the data using the mean from fitting
        X_centered = X - self.mean
        
        # Project data onto principal components
        return np.dot(X_centered, self.components)
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### Setup Instructions
```bash
pip install numpy scikit-learn matplotlib
```

---

## Hyperparameters & Optimization

### Key Parameters
- **n_components**: Number of components to keep (default: min(n_samples, n_features)).
- **svd_solver**: Method for SVD calculation ('auto', 'full', 'arpack', 'randomized').
- **whiten**: Whether to whiten data (divide components by square root of eigenvalues).

### Tuning Strategies
- **Explained Variance Ratio**: Select components that explain a target percentage of variance (e.g., 95%).
- **Scree Plot**: Plot eigenvalues and look for the "elbow" point.
- **Cross-validation**: Select components that yield best performance on downstream tasks.

---

## Evaluation Metrics
- **Explained Variance Ratio**: Proportion of variance explained by each component.
- **Cumulative Explained Variance**: Sum of explained variance ratios.
- **Reconstruction Error**: Difference between original and reconstructed data when using k components.

---

## Practical Examples
**Dataset**: MNIST digits, facial recognition, gene expression data.
**Use Case**: Visualization of high-dimensional data, noise reduction, feature extraction for downstream models.

```python
# Visualization example
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter, label='Digit')
plt.title('PCA projection of the digits dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

---

## Advanced Theory

### Singular Value Decomposition (SVD) Implementation
PCA can be implemented using SVD, which is more numerically stable:
$$ X = U \Sigma V^T $$
where:
- $U$ contains the left singular vectors
- $\Sigma$ is a diagonal matrix of singular values
- $V$ contains the right singular vectors (principal components)

The principal components are the columns of $V$, and the projected data can be obtained as:
$$ X_{reduced} = U \Sigma $$

### Kernel PCA
An extension of PCA for nonlinear dimensionality reduction using the kernel trick:
$$ K_{ij} = k(x_i, x_j) = \phi(x_i) \cdot \phi(x_j) $$
This allows finding principal components in a high-dimensional feature space without explicitly computing the transformation $\phi$.

---

## Advantages & Limitations

**Pros**:
- Reduces dimensionality without supervised information
- Removes correlated features and reduces overfitting
- Improves computational efficiency for downstream tasks
- Effective for visualization of high-dimensional data
- Helps with noise reduction

**Cons**:
- Limited to linear transformations
- May lose important information if variance doesn't correlate with importance
- Scale-sensitive (requires standardization)
- Difficult to interpret transformed features
- Cannot handle categorical variables directly

---

## Further Reading
1. Jolliffe, I.T. (2002). "Principal Component Analysis", Springer Series in Statistics.
2. Shlens, J. (2014). "A Tutorial on Principal Component Analysis".
3. Abdi, H., & Williams, L.J. (2010). "Principal component analysis".
4. Wold, S., Esbensen, K., & Geladi, P. (1987). "Principal component analysis".
