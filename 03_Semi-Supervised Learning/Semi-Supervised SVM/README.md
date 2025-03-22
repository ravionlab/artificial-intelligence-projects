# Semi-Supervised Support Vector Machines (S3VM)

## Overview & Introduction
Semi-Supervised Support Vector Machines (S3VM), also known as Transductive SVMs (TSVMs), extend the traditional SVM framework to leverage unlabeled data. The algorithm aims to find a decision boundary that not only maximizes the margin between labeled data points but also passes through low-density regions of the input space, incorporating information from unlabeled examples.

**Role in Semi-Supervised Learning**:
S3VM addresses the common scenario where labeled data is scarce and expensive to obtain, while unlabeled data is abundant. It leverages the geometric structure revealed by unlabeled data to improve classification performance.

### Historical Context
Developed in the late 1990s by Vapnik and others as an extension of the highly successful Support Vector Machine algorithm. S3VM represented one of the first theoretically grounded approaches to semi-supervised learning and has inspired numerous variants and applications across different domains.

---

## Theoretical Foundations

### Conceptual Explanation
Traditional SVMs find a decision boundary that maximizes the margin between classes based on labeled data alone. S3VM extends this by additionally requiring the decision boundary to pass through regions with low data density, using the insight that natural class boundaries often occur in sparse regions of the feature space.

### Mathematical Formulation
**Objective Function**:
The S3VM optimization problem can be formulated as:

$$ \min_{w, b, y_u} \frac{1}{2} ||w||^2 + C_l \sum_{i=1}^{l} \max(0, 1-y_i(w \cdot x_i + b))^2 + C_u \sum_{j=l+1}^{l+u} \max(0, 1-y_j(w \cdot x_j + b))^2 $$

Where:
- $(w, b)$ defines the decision boundary
- $y_i$ are the known labels for labeled examples $x_i$
- $y_j$ are the unknown (to be determined) labels for unlabeled examples $x_j$
- $C_l$ and $C_u$ are regularization parameters for labeled and unlabeled data

**Kernel Extension**:
Like traditional SVMs, S3VM can be extended to nonlinear classification using kernels:

$$ f(x) = \sum_{i=1}^{l+u} \alpha_i y_i K(x_i, x) + b $$

Where $K(x_i, x)$ is a kernel function.

### Assumptions
1. **Low-Density Separation**: Decision boundaries should pass through regions of low data density.
2. **Cluster Assumption**: Points in the same cluster are likely to share the same label.
3. **Manifold Assumption**: The high-dimensional data lies approximately on a low-dimensional manifold.

---

## Algorithm Mechanics

### Step-by-Step Process
1. **Initial Boundary**: Train a standard SVM on labeled data only.
2. **Label Assignment**: Assign tentative labels to unlabeled points based on the current decision boundary.
3. **Continuous Optimization**: Solve a continuous optimization problem that includes both labeled and unlabeled data with their current labels.
4. **Label Switching**: Identify and switch labels of unlabeled points if doing so reduces the objective function.
5. **Iterative Refinement**: Repeat steps 3-4 until convergence or a maximum number of iterations is reached.

### Training & Prediction Workflow
```python
def train_s3vm(X_labeled, y_labeled, X_unlabeled, C_l=1.0, C_u=0.1, 
               kernel='linear', max_iter=100):
    # Step 1: Train initial SVM on labeled data
    initial_svm = SVC(C=C_l, kernel=kernel)
    initial_svm.fit(X_labeled, y_labeled)
    
    # Step 2: Get initial labels for unlabeled data
    y_unlabeled = initial_svm.predict(X_unlabeled)
    
    # Combined dataset
    X_combined = np.vstack((X_labeled, X_unlabeled))
    y_combined = np.concatenate((y_labeled, y_unlabeled))
    
    # Initialize annealing parameters
    T = 100.0  # Initial temperature
    T_min = 0.1  # Minimum temperature
    alpha = 0.9  # Cooling rate
    
    current_svm = SVC(C=C_l, kernel=kernel)
    current_svm.fit(X_combined, y_combined)
    current_objective = compute_objective(current_svm, X_combined, y_combined, 
                                         X_labeled, y_labeled, C_l, C_u)
    
    for iteration in range(max_iter):
        improved = False
        
        # Step 3 & 4: Try switching labels
        for i in range(len(X_unlabeled)):
            # Index in the combined dataset
            idx = i + len(X_labeled)
            
            # Try flipping this label
            y_combined_new = y_combined.copy()
            y_combined_new[idx] = -y_combined_new[idx]
            
            # Train with new labels
            new_svm = SVC(C=C_l, kernel=kernel)
            new_svm.fit(X_combined, y_combined_new)
            
            # Compute new objective
            new_objective = compute_objective(new_svm, X_combined, y_combined_new, 
                                            X_labeled, y_labeled, C_l, C_u)
            
            # Accept if better, or probabilistically based on temperature
            delta = new_objective - current_objective
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_svm = new_svm
                current_objective = new_objective
                y_combined = y_combined_new
                improved = True
        
        # Reduce temperature
        T = max(T * alpha, T_min)
        
        # Check for convergence
        if not improved:
            break
    
    return current_svm

def compute_objective(model, X, y, X_labeled, y_labeled, C_l, C_u):
    # Compute the S3VM objective function
    w = model.coef_[0]
    b = model.intercept_[0]
    
    # Margin term
    margin_term = 0.5 * np.sum(w ** 2)
    
    # Loss on labeled data
    labeled_indices = range(len(X_labeled))
    labeled_loss = 0
    for i in labeled_indices:
        loss = max(0, 1 - y[i] * (np.dot(w, X[i]) + b))
        labeled_loss += loss ** 2
    
    # Loss on unlabeled data
    unlabeled_indices = range(len(X_labeled), len(X))
    unlabeled_loss = 0
    for i in unlabeled_indices:
        loss = max(0, 1 - y[i] * (np.dot(w, X[i]) + b))
        unlabeled_loss += loss ** 2
    
    return margin_term + C_l * labeled_loss + C_u * unlabeled_loss
```

---

## Implementation Details

### Code Structure
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

class S3VM(BaseEstimator, ClassifierMixin):
    def __init__(self, C_l=1.0, C_u=0.1, kernel='linear', max_iter=100, 
                 switching_strategy='simulated_annealing'):
        self.C_l = C_l
        self.C_u = C_u
        self.kernel = kernel
        self.max_iter = max_iter
        self.switching_strategy = switching_strategy
    
    def fit(self, X, y):
        # Split into labeled and unlabeled
        labeled_mask = y != -1
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        X_unlabeled = X[~labeled_mask]
        
        # Initial SVM on labeled data only
        self.base_svm_ = SVC(C=self.C_l, kernel=self.kernel, probability=True)
        self.base_svm_.fit(X_labeled, y_labeled)
        
        # Get initial predictions for unlabeled data
        if len(X_unlabeled) > 0:
            y_unlabeled_pred = self.base_svm_.predict(X_unlabeled)
            
            # Combined data
            X_combined = np.vstack((X_labeled, X_unlabeled))
            y_combined = np.concatenate((y_labeled, y_unlabeled_pred))
            
            # Implement label switching strategy
            if self.switching_strategy == 'simulated_annealing':
                self.final_svm_ = self._simulated_annealing(X_combined, y_combined, 
                                                          len(X_labeled))
            elif self.switching_strategy == 'gradient_descent':
                self.final_svm_ = self._gradient_descent(X_combined, y_combined, 
                                                       len(X_labeled))
            else:
                self.final_svm_ = self.base_svm_
        else:
            self.final_svm_ = self.base_svm_
        
        return self
    
    def predict(self, X):
        return self.final_svm_.predict(X)
    
    def predict_proba(self, X):
        return self.final_svm_.predict_proba(X)
    
    def _simulated_annealing(self, X, y, n_labeled):
        # Implementation of simulated annealing strategy for S3VM
        # ...
        pass
    
    def _gradient_descent(self, X, y, n_labeled):
        # Implementation of gradient descent strategy for S3VM
        # ...
        pass
    
    def _compute_objective(self, svm, X, y, n_labeled):
        # Calculate the S3VM objective function
        # ...
        pass
```

### Setup Instructions
```bash
pip install numpy scikit-learn cvxopt
```

---

## Hyperparameters & Optimization

- **C_l**: Regularization parameter for labeled data.
- **C_u**: Regularization parameter for unlabeled data.
- **Kernel**: Type of kernel function (linear, rbf, polynomial).
- **Switching Strategy**: Method for optimizing unlabeled data labels.
- **Balancing Constraint**: Enforcing class balance in unlabeled data predictions.

**Tuning Strategies**:
- Grid search on a small validation set for C_l and C_u.
- Gradually increase the weight of unlabeled data (C_u) during training.
- Use model selection criteria that account for both labeled and unlabeled data.

---

## Evaluation Metrics

- **Classification Accuracy**: Performance on held-out test data.
- **Transductive Accuracy**: Performance on the unlabeled training data (if ground truth is available).
- **Decision Boundary Visualization**: Qualitative assessment of how well the boundary separates classes and passes through low-density regions.

---

## Practical Examples

**Dataset**: Text Document Classification with sparse labeled instances.
**Use Case**: Protein Structure Prediction where labeled examples are obtained through expensive laboratory experiments.

---

## Advanced Theory

**Non-Convexity and Optimization Challenges**:
The S3VM objective function is non-convex due to the unknown labels of unlabeled data, making global optimization challenging. Various approaches address this:

1. **Continuous Relaxation**: Replace binary labels with continuous values and gradually force them toward binary solutions.
2. **Deterministic Annealing**: Begin with an easy optimization problem and gradually transform it into the target problem.
3. **Convex-Concave Procedures (CCCP)**: Decompose the non-convex problem into a series of convex sub-problems.

**Theoretical Guarantees**:
Under certain conditions regarding the data distribution and the number of unlabeled examples, S3VM can achieve significantly better generalization performance than standard SVMs with the same number of labeled examples.

**Relationship to Manifold Regularization**:
S3VM can be viewed as a special case of manifold regularization where the regularization term encourages decision boundaries to pass through low-density regions.

**Multi-Class Extensions**:
Extensions of S3VM to multi-class problems typically use approaches like one-vs-all or one-vs-one strategies, though specialized formulations exist that directly address the multi-class setting.

---

## Advantages & Limitations

**Pros**:
- Theoretical foundation in statistical learning theory.
- Effectively utilizes unlabeled data when the low-density assumption holds.
- Can significantly improve performance over supervised SVMs when labeled data is scarce.
- Extensions can incorporate prior knowledge about class distributions.

**Cons**:
- Non-convex optimization makes finding the global optimum challenging.
- Computationally intensive, especially for large unlabeled datasets.
- Performance depends heavily on the validity of the low-density assumption.
- Sensitive to initial labeling of unlabeled data.

---

## Further Reading

1. Vapnik, V., & Sterin, A. (1977). "On structural risk minimization or overall risk in a problem of pattern recognition."
2. Joachims, T. (1999). "Transductive Inference for Text Classification using Support Vector Machines."
3. Chapelle, O., & Zien, A. (2005). "Semi-Supervised Classification by Low Density Separation."
4. Li, Y. F., & Zhou, Z. H. (2015). "Towards Making Unlabeled Data Never Hurt."

---

# Co-Training
*(Described in a separate README with focus on multi-view learning and classifier agreements.)*

# Self-Training
*(Described in a separate README with focus on single-classifier iterative learning.)*
