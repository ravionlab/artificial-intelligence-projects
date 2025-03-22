# Support Vector Machines (SVM)

## Table of Contents
- [Overview & Introduction](#overview--introduction)
- [Theoretical Foundations](#theoretical-foundations)
- [Algorithm Mechanics](#algorithm-mechanics)
- [Implementation Details](#implementation-details)
- [Hyperparameters & Optimization](#hyperparameters--optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Practical Examples](#practical-examples)
- [Advanced Theory](#advanced-theory)
- [Advantages & Limitations](#advantages--limitations)
- [Further Reading](#further-reading)

## Overview & Introduction
Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification, regression, and outlier detection. SVMs are particularly effective in high-dimensional spaces and cases where the number of dimensions exceeds the number of samples.

**Role in Supervised Learning**:  
SVMs excel at finding the optimal boundary that separates different classes by maximizing the margin between the closest points (support vectors) of different classes. They can perform both linear and non-linear classification through kernel methods.

### Historical Context
Developed by Vladimir Vapnik and colleagues in the 1990s, SVMs are based on statistical learning theory. They gained popularity due to their strong theoretical foundations and performance on complex classification tasks, especially before the rise of deep learning.

## Theoretical Foundations

### Conceptual Explanation
SVMs aim to find a hyperplane that best divides a dataset into classes by maximizing the margin between the closest points from each class. These closest points are called support vectors. For linearly separable data, the decision boundary is expressed as:

$w \cdot x + b = 0$

where:
- $w$: The weight vector perpendicular to the hyperplane
- $x$: Input feature vector
- $b$: Bias term

### Mathematical Formulation
**Primal Form**: The optimization objective is to minimize:

$\min_{w,b} \frac{1}{2} ||w||^2$

Subject to: $y_i(w \cdot x_i + b) \geq 1$ for all $i$

**Dual Form**: Using Lagrange multipliers, this becomes:

$\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)$

Subject to: $\alpha_i \geq 0$ and $\sum_{i=1}^{m} \alpha_i y_i = 0$

**Kernel Trick**: To handle non-linear boundaries, SVMs use kernel functions:

$K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$

### Assumptions
1. The optimal decision boundary maximizes the margin between classes
2. The classification problem is separable (though soft margin allows for some misclassifications)
3. Higher-dimensional mapping can make non-linearly separable data linearly separable

## Algorithm Mechanics

### Step-by-Step Process
1. **Data Preparation**: Normalize features and split data
2. **Kernel Selection**: Choose appropriate kernel function (linear, polynomial, RBF, etc.)
3. **Optimization**: Solve the quadratic programming problem to find support vectors
4. **Model Construction**: Create decision boundary using support vectors
5. **Classification**: Determine class based on position relative to the decision boundary

### Training & Prediction Workflow
```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Implementation Details

### Code Structure
```python
import numpy as np
from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True
        )
        
    def fit(self, X, y):
        return self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    @property
    def support_vectors_(self):
        return self.model.support_vectors_
```

### Setup Instructions
```bash
pip install numpy scikit-learn
```

## Hyperparameters & Optimization
- **C**: Regularization parameter (controls trade-off between smooth decision boundary and classifying training points correctly)
- **kernel**: Type of kernel function ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
- **gamma**: Kernel coefficient for 'rbf', 'poly' and 'sigmoid' ('scale', 'auto' or float)
- **degree**: Degree of the polynomial kernel function
- **class_weight**: Weights associated with classes (useful for imbalanced datasets)

**Tuning Strategies**:
- Grid search over C and gamma parameters
- Cross-validation to prevent overfitting

## Evaluation Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision and Recall**: Especially important for imbalanced datasets
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Practical Examples
**Use Cases**:
- Image classification
- Text categorization
- Handwriting recognition
- Biological sequence analysis
- Protein classification
- Face detection

**Example Datasets**: MNIST digits, cancer diagnosis, text sentiment analysis.

## Advanced Theory
**Soft Margin SVM**: Includes slack variables to allow for misclassifications:

$\min_{w,b,\xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{m} \xi_i$

Subject to: $y_i(w \cdot x_i + b) \geq 1 - \xi_i$ and $\xi_i \geq 0$ for all $i$

**Kernel Functions**:
- Linear: $K(x_i, x_j) = x_i \cdot x_j$
- Polynomial: $K(x_i, x_j) = (\gamma x_i \cdot x_j + r)^d$
- RBF (Gaussian): $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$
- Sigmoid: $K(x_i, x_j) = \tanh(\gamma x_i \cdot x_j + r)$

**SVM Regression**: Uses epsilon-insensitive loss function to perform regression.

## Advantages & Limitations
**Pros**:
- Effective in high-dimensional spaces
- Memory efficient as it uses only a subset of training points (support vectors)
- Versatile through different kernel functions
- Robust against overfitting, especially in high-dimensional space
- Works well with clear margin of separation

**Cons**:
- Not directly suited for multi-class classification (requires one-vs-rest or one-vs-one strategies)
- Computationally intensive for large datasets
- Requires careful tuning of hyperparameters
- Difficult to interpret compared to simpler models
- Feature scaling is crucial for performance

## Further Reading
1. Vapnik, V., *The Nature of Statistical Learning Theory*
2. [IBM: Support Vector Machine](https://www.ibm.com/topics/support-vector-machine)
3. [MathWorks: Support Vector Machine](https://www.mathworks.com/discovery/support-vector-machine.html)
4. [TechTarget: Support Vector Machine](https://www.techtarget.com/whatis/definition/support-vector-machine-SVM)
5. [Aylien: Support Vector Machines for Dummies](https://aylien.com/blog/support-vector-machines-for-dummies-a-simple-explanation)
6. [Mathematical Understanding of SVMs](https://shuzhanfan.github.io/2018/05/understanding-mathematics-behind-support-vector-machines/)
