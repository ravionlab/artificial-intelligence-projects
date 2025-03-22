# K-Nearest Neighbors (KNN) Algorithm

## Table of Contents
- [Overview & Introduction](#overview--introduction)
- [Theoretical Foundations](#theoretical-foundations)
- [Algorithm Mechanics](#algorithm-mechanics)
- [Distance Metrics](#distance-metrics)
- [Implementation Details](#implementation-details)
- [Hyperparameters & Optimization](#hyperparameters--optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Practical Examples](#practical-examples)
- [Advanced Theory & Extensions](#advanced-theory--extensions)
- [Computational Complexity](#computational-complexity)
- [Advantages & Limitations](#advantages--limitations)
- [Further Reading](#further-reading)

## Overview & Introduction

The K-Nearest Neighbors (KNN) algorithm is a versatile, intuitive, and powerful non-parametric method used for both classification and regression tasks in machine learning. Unlike many algorithms that build explicit models during training, KNN is an instance-based learning algorithm that stores all available examples and classifies new instances based on similarity measures.

**Role in Supervised Learning**:  
KNN belongs to the family of lazy learning algorithms, meaning it doesn't construct a generalized internal model. Instead, it memorizes the training instances and defers processing until classification time. This approach allows KNN to adapt to previously unseen patterns in the data without requiring retraining, making it particularly valuable for dynamic environments where new patterns emerge over time.

### Historical Context

The KNN algorithm was first introduced in the 1950s as a non-parametric method for pattern recognition. It was formalized by Cover and Hart in their 1967 paper "Nearest Neighbor Pattern Classification," which established many of the theoretical properties of the algorithm. KNN gained significant popularity in the 1960s and 1970s when computational resources became more accessible, allowing for the storage and processing of larger datasets. Today, KNN serves not only as a practical algorithm for various applications but also as a benchmark against which more complex algorithms are compared.

## Theoretical Foundations

### Conceptual Explanation

The fundamental premise of KNN is remarkably intuitive: similar instances exist in close proximity to each other. When classifying a new data point, KNN examines the k closest neighbors from the training set and determines the class assignment based on the majority vote (for classification) or average value (for regression) of these neighbors.

This approach is rooted in the concept of local approximation—the assumption that the function we're trying to learn is smooth and can be approximated locally by examining nearby points. This locality principle allows KNN to capture complex decision boundaries that many parametric models cannot represent effectively.

### Mathematical Formulation

For a given test point $x$, the KNN algorithm:

1. Calculates the distance between $x$ and all training samples
2. Selects the $k$ nearest neighbors 
3. Assigns a class or value based on these neighbors

**For Classification**:  
The predicted class $\hat{y}$ for a new instance $x$ is determined by majority voting:

$\hat{y} = \text{mode}(y_i \mid x_i \in N_k(x))$

where $N_k(x)$ is the set of $k$ nearest neighbors of $x$.

**For Regression**:  
The predicted value is typically the average of the values of its $k$ nearest neighbors:

$\hat{y} = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i$

Alternatively, a distance-weighted average can be used:

$\hat{y} = \frac{\sum_{x_i \in N_k(x)} w_i \cdot y_i}{\sum_{x_i \in N_k(x)} w_i}$

where $w_i = \frac{1}{d(x, x_i)^2}$ and $d(x, x_i)$ is the distance between $x$ and $x_i$.

### Assumptions

1. **Locality assumption**: Points that are close in feature space are likely to have similar target values or belong to the same class
2. **Feature relevance**: All features are equally important in determining similarity (though this can be adjusted through feature weighting)
3. **Sufficient training data**: The training set should be dense enough to capture the underlying structure of the data
4. **Feature space structure**: The distance metric used should create a meaningful structure in the feature space

## Distance Metrics

The choice of distance metric is crucial for KNN as it defines the notion of "nearness." Different distance metrics can lead to dramatically different results.

### Euclidean Distance

The most commonly used distance metric, Euclidean distance measures the straight-line distance between two points in Euclidean space:

$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$

This metric is most appropriate when the feature space is isotropic (uniform in all directions) and features are measured on comparable scales.

### Manhattan Distance (L1 norm)

Also known as city block distance, this metric calculates the sum of absolute differences between coordinates:

$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$

Manhattan distance is less sensitive to outliers compared to Euclidean distance and is often preferred when features represent fundamentally different quantities.

### Minkowski Distance

A generalization of both Euclidean and Manhattan distances:

$d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}$

where $p$ is a parameter. When $p=1$, it's Manhattan distance; when $p=2$, it's Euclidean distance.

### Hamming Distance

Used primarily for categorical variables, Hamming distance counts the number of positions at which the corresponding symbols are different:

$d(x, y) = \sum_{i=1}^{n} [x_i \neq y_i]$

where $[x_i \neq y_i]$ is 1 if $x_i \neq y_i$ and 0 otherwise.

### Cosine Similarity

Measures the cosine of the angle between two vectors, focusing on orientation rather than magnitude:

$\text{similarity} = \cos(\theta) = \frac{x \cdot y}{||x|| \cdot ||y||}$

Particularly useful for high-dimensional sparse data, such as in text analysis and recommendation systems.

## Algorithm Mechanics

### Step-by-Step Process

1. **Data Preparation**:
   - Normalize or standardize features to ensure all attributes contribute equally to the distance calculation
   - Handle missing values through imputation or removal
   - Encode categorical variables appropriately

2. **Distance Calculation**:
   - For a given test instance, compute its distance to all training instances using the chosen distance metric
   - This creates a distance vector of length equal to the number of training samples

3. **Neighbor Selection**:
   - Sort the distance vector in ascending order
   - Select the first k entries, which correspond to the k closest training instances
   - These k instances form the neighborhood of the test instance

4. **Decision Making**:
   - For classification: Count class frequencies among the k neighbors and assign the majority class
   - For regression: Calculate the mean or weighted mean of the target values of the k neighbors
   - In case of ties in classification (possible when k is even), either reduce k by 1 or use a distance-weighted voting scheme

5. **Result Interpretation**:
   - The confidence in a classification can be estimated by the proportion of neighbors belonging to the predicted class
   - For regression, the variance of the values among neighbors can indicate prediction uncertainty

### Training & Prediction Workflow

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and KNN
# For classification
clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])

# Fit the model
clf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = clf_pipeline.predict(X_test)
y_prob = clf_pipeline.predict_proba(X_test)  # Probability estimates

# For regression
reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])
reg_pipeline.fit(X_train, y_train)
y_pred_reg = reg_pipeline.predict(X_test)
```

## Implementation Details

### Code Structure

```python
import numpy as np
from collections import Counter
from scipy.spatial import distance

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean', weight='uniform'):
        """
        Initialize the KNN classifier.
        
        Parameters:
        -----------
        k : int
            Number of neighbors to consider
        distance_metric : str
            Distance metric to use ('euclidean', 'manhattan', 'minkowski', etc.)
        weight : str
            Weighting scheme ('uniform' or 'distance')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weight = weight
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Store the training data for later use during prediction.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
            Returns self
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
        
    def _calculate_distances(self, x):
        """
        Calculate distances from a point to all training instances.
        
        Parameters:
        -----------
        x : array-like of shape (n_features,)
            Test instance
            
        Returns:
        --------
        distances : array of shape (n_training_samples,)
            Distances to all training instances
        """
        if self.distance_metric == 'euclidean':
            return [distance.euclidean(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            return [distance.cityblock(x, x_train) for x_train in self.X_train]
        else:  # Default to Euclidean
            return [distance.euclidean(x, x_train) for x_train in self.X_train]
    
    def _get_weights(self, distances):
        """
        Calculate weights for the neighbors based on their distances.
        
        Parameters:
        -----------
        distances : array of shape (k,)
            Distances to k nearest neighbors
            
        Returns:
        --------
        weights : array of shape (k,)
            Weights for each neighbor
        """
        if self.weight == 'uniform':
            return np.ones(len(distances))
        elif self.weight == 'distance':
            # Avoid division by zero
            return 1.0 / (np.array(distances) + 1e-10)
        else:  # Default to uniform
            return np.ones(len(distances))
            
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Class labels for each data sample
        """
        X = np.array(X)
        y_pred = []
        
        # For each test instance
        for x in X:
            # Calculate distances to all training instances
            distances = self._calculate_distances(x)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get classes of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Get distances to k nearest neighbors
            k_nearest_distances = [distances[i] for i in k_indices]
            
            # Calculate weights
            weights = self._get_weights(k_nearest_distances)
            
            # If using weights, use weighted majority vote
            if self.weight == 'distance':
                # Create a dictionary to count weighted votes
                weighted_votes = {}
                for label, weight in zip(k_nearest_labels, weights):
                    if label in weighted_votes:
                        weighted_votes[label] += weight
                    else:
                        weighted_votes[label] = weight
                # Get the class with highest weighted votes
                prediction = max(weighted_votes, key=weighted_votes.get)
            else:
                # Use majority voting
                prediction = Counter(k_nearest_labels).most_common(1)[0][0]
                
            y_pred.append(prediction)
            
        return np.array(y_pred)
    
    def predict_proba(self, X):
        """
        Return probability estimates for each class.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        probas : array of shape (n_samples, n_classes)
            Probability of each sample for each class
        """
        X = np.array(X)
        probas = []
        
        # Get unique classes
        classes = np.unique(self.y_train)
        
        # For each test instance
        for x in X:
            # Calculate distances to all training instances
            distances = self._calculate_distances(x)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get classes of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Get distances to k nearest neighbors
            k_nearest_distances = [distances[i] for i in k_indices]
            
            # Calculate weights
            weights = self._get_weights(k_nearest_distances)
            
            # Calculate class probabilities
            class_probas = []
            for cls in classes:
                # Sum weights for instances of this class
                cls_weight = sum(w for l, w in zip(k_nearest_labels, weights) if l == cls)
                # Normalize by total weight
                prob = cls_weight / sum(weights)
                class_probas.append(prob)
                
            probas.append(class_probas)
            
        return np.array(probas)
```

### Setup Instructions

```bash
# Install required packages
pip install numpy scipy scikit-learn matplotlib

# For advanced usage with GPU acceleration
pip install cuml  # CUDA-accelerated implementation of KNN and other ML algorithms
```

## Hyperparameters & Optimization

### Key Hyperparameters

- **k (Number of Neighbors)**: The most critical hyperparameter, determining how many nearest neighbors influence the classification/regression
  - Smaller k: More sensitive to noise, captures finer patterns but risks overfitting
  - Larger k: Smoother decision boundaries, more robust but may underfit
  - Typically, an odd value of k is chosen for binary classification to avoid ties
  - The optimal k often depends on the dataset's noise level and complexity

- **Distance Metric**: The method used to calculate the similarity between instances
  - The choice depends on the nature of the features and domain knowledge
  - Some metrics perform better for specific types of data (e.g., cosine similarity for text)

- **Weight Function**: How to weight neighbors' contributions
  - 'uniform': All neighbors contribute equally
  - 'distance': Closer neighbors have greater influence (weight inversely proportional to distance)
  - Custom weighting schemes can be implemented to better reflect domain-specific notions of influence

- **Algorithm**: The method used to compute nearest neighbors
  - 'brute': Compute distances between all pairs of points (works for any distance metric)
  - 'kd_tree': Efficient for low-dimensional data but performance degrades in high dimensions
  - 'ball_tree': Generally faster than KD-tree in high dimensions
  - 'auto': Attempts to decide the most appropriate algorithm based on the input data

### Tuning Strategies

- **Cross-Validation**: Use k-fold cross-validation to evaluate different parameter configurations
  - Grid search for small parameter spaces
  - Random search for larger parameter spaces

- **Learning Curves**: Plot performance against k to identify the optimal value
  - Often, performance improves as k increases, then plateaus, and finally degrades

- **Elbow Method**: Plot error rates against k values and look for the "elbow point" where diminishing returns set in

- **Domain-Specific Considerations**:
  - In noisy domains, larger k values typically perform better
  - In domains with rare but important patterns, smaller k values may be necessary

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Define parameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Perform grid search
grid_search = GridSearchCV(
    pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Get best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Create model with best parameters
best_model = grid_search.best_estimator_
```

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Proportion of correct predictions (useful for balanced datasets)
- **Precision**: Ratio of true positives to all predicted positives
- **Recall (Sensitivity)**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes
- **Confusion Matrix**: Table showing TP, TN, FP, FN counts to better understand error patterns

### Regression Metrics

- **Mean Absolute Error (MAE)**: Average of absolute differences between predictions and actual values
- **Mean Squared Error (MSE)**: Average of squared differences (penalizes larger errors more)
- **Root Mean Squared Error (RMSE)**: Square root of MSE, interpretable in the original unit
- **R-squared**: Proportion of variance explained by the model
- **Mean Absolute Percentage Error (MAPE)**: Average of percentage errors (useful for measuring relative accuracy)

### Implementation Example

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For classification
y_pred = knn_clf.predict(X_test)
y_pred_proba = knn_clf.predict_proba(X_test)[:, 1]  # Probabilities for positive class

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# For regression
y_pred_reg = knn_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_reg)
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_reg)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
```

## Practical Examples

### Use Cases and Applications

KNN finds application across numerous domains due to its intuitive approach and flexibility:

- **Recommendation Systems**: Suggesting products/content based on similarity to items users previously liked
  - Example: Movie recommendations based on preferences of users with similar taste profiles
  - Implementation: Each user is represented as a point in feature space, and recommendations come from the preferences of nearby users

- **Image Recognition**: Classifying images based on visual similarity
  - Example: Handwritten digit recognition (MNIST dataset)
  - Implementation: Images are represented as flattened pixel vectors, and classification is based on similarity to known examples

- **Medical Diagnosis**: Disease classification based on patient symptoms and test results
  - Example: Diagnosing cancer types based on gene expression profiles
  - Implementation: Patients' clinical data forms the feature space, and diagnosis leverages the medical histories of similar cases

- **Anomaly Detection**: Identifying unusual patterns that don't conform to expected behavior
  - Example: Fraud detection in credit card transactions
  - Implementation: Transactions far from any clusters of normal behavior are flagged as potential fraud

- **Predictive Maintenance**: Forecasting equipment failures before they occur
  - Example: Predicting machine breakdowns based on sensor readings
  - Implementation: Current sensor patterns are compared to historical patterns preceding known failures

### Sample Implementation: Iris Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Test different k values
k_values = range(1, 31)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

# Plot accuracy vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(k_values[::2])  # Show every other k value
plt.show()

# Find the optimal k
optimal_k = k_values[np.argmax(accuracies)]
print(f"Optimal k value: {optimal_k}")

# Train and evaluate the model with optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Print detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Visualize decision boundaries (for 2D projection)
plt.figure(figsize=(12, 5))

# Use petal length and width for visualization
X_features = X[:, 2:4]  # Petal length and width
x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                    np.arange(y_min, y_max, 0.02))

# Train KNN on these two features
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_features, y)

# Predict across the grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_features[:, 0], X_features[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title(f'KNN Decision Boundaries (k={optimal_k})')

# Plot probability for class 1 (versicolor)
probs = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
probs = probs.reshape(xx.shape)
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, probs, 100, cmap=plt.cm.RdBu)
plt.colorbar()
plt.scatter(X_features[:, 0], X_features[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Class Probability (Versicolor)')

plt.tight_layout()
plt.show()
```

## Advanced Theory & Extensions

### Weighted KNN

Standard KNN gives equal importance to all k neighbors, which can be problematic when some neighbors are significantly closer than others. Weighted KNN addresses this by assigning weights to neighbors based on their distance:

$\hat{y} = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}$

Common weighting schemes include:
- Inverse distance: $w_i = \frac{1}{d(x, x_i)}$
- Inverse squared distance: $w_i = \frac{1}{d(x, x_i)^2}$
- Exponential decay: $w_i = e^{-d(x, x_i)}$

### Condensed Nearest Neighbors (CNN)

A method to reduce the size of the training set by selecting a subset that achieves the same classification performance. The algorithm:
1. Start with an empty set S
2. Add the first training example to S
3. For each remaining example, classify it using S as the training set
4. If misclassified, add it to S; otherwise, discard it
5. Repeat until all examples are correctly classified or added to S

This produces a minimal consistent subset that preserves decision boundaries.

### Local Weighted Learning (LWL)

A more sophisticated extension where a local model is built for each prediction point using weighted training instances. For regression:
1. Assign weights to all training points based on their distance to the query point
2. Fit a (usually simple) regression model using these weighted points
3. Use this local model to make a prediction for the query point

This approach combines KNN's locality with parametric modeling's flexibility.

### KNN with Dimensionality Reduction

High-dimensional spaces suffer from the "curse of dimensionality," where distance metrics become less meaningful. Combining KNN with dimensionality reduction techniques can improve performance:
- Principal Component Analysis (PCA)
- t-SNE
- Feature selection methods

### Fast Approximate Nearest Neighbors

For large datasets, exact nearest neighbor search becomes computationally prohibitive. Approximate methods trade slight accuracy for greatly improved speed:
- Locality-Sensitive Hashing (LSH)
- Hierarchical Navigable Small World (HNSW) graphs
- Product quantization

### Adaptive KNN

Rather than using a fixed k, adaptive methods determine the appropriate k for each query point:
- Use a radius-based approach (include all neighbors within a certain distance)
- Adjust k based on local density (smaller k in dense regions, larger k in sparse regions)
- Dynamically select k to minimize local leave-one-out error

## Computational Complexity

### Time Complexity

- **Training**: O(1) - KNN doesn't have a training phase, it simply stores the training data
- **Prediction (Brute Force)**:
  - Time: O(n×d) for a single query, where n is the number of training instances and d is the dimensionality
  - For batch prediction of m query points: O(m×n×d)
- **Prediction (KD-Tree or Ball Tree)**:
  - Time: O(log n) for low dimensions
  - Degrades to O(n) in high dimensions (typically >20)

### Space Complexity

- **Model Storage**: O(n×d) - Requires storing the entire training set
- **Additional Structures**:
  - KD-Tree: O(n)
  - Ball Tree: O(n)

### Memory Requirements

For a dataset with:
- n = 1 million instances
- d = 100 features
- 8 bytes per floating-point value

The memory requirement would be approximately:
n × d × 8 bytes = 10^6 × 100 × 8 bytes ≈ 800 MB

This highlights why KNN can be impractical for very large datasets without specialized approaches.

## Advantages & Limitations

### Advantages

- **Simplicity**: Conceptually straightforward and easy to understand
- **Non-parametric**: Makes no assumptions about the underlying data distribution
- **Adaptability**: Naturally adapts to complex decision boundaries
- **No Training Phase**: Can incorporate new data without retraining
- **Versatility**: Applicable to both classification and regression problems
- **Theoretical Guarantees**: As the training set size approaches infinity, the error rate is bounded by twice the Bayes error rate (the theoretical minimum)
- **Interpretability**: Predictions can be explained by examining the nearest neighbors

### Limitations

- **Computational Cost**: Prediction time scales poorly with dataset size
- **Memory Requirements**: Needs to store the entire training dataset
- **Curse of Dimensionality**: Performance degrades in high-dimensional spaces as distances become less meaningful
- **Sensitivity to Noisy Data**: Outliers can significantly impact predictions
- **Imbalanced Data**: Majority classes can dominate predictions
- **Scale Sensitivity**: Features with larger scales can dominate the distance calculations
- **Optimal K Selection**: Choosing the optimal k usually requires cross-validation
- **Cold Start Problem**: Performs poorly when there is limited training data

### Mitigating Limitations

- Scale features to ensure equal contribution to distance calculations
- Use weighted voting to reduce the impact of distant neighbors
- Perform dimensionality reduction for high-dimensional data
- Employ approximate nearest neighbor algorithms for large datasets
- For imbalanced data, use class-weighted approaches or adjust the neighborhood composition

## Further Reading
1. Cover, T.0 : (https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf)
