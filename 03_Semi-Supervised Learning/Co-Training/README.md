# Co-Training

## Overview & Introduction
Co-Training is a semi-supervised learning algorithm that leverages multiple views of the data to improve classification performance when labeled data is scarce. Introduced by Blum and Mitchell in 1998, it operates under the assumption that different feature subsets (views) can provide complementary information about the same instances.

**Role in Semi-Supervised Learning**:
Co-Training bridges the gap between supervised and unsupervised learning by using a small set of labeled data along with a large pool of unlabeled data to train multiple classifiers that teach each other.

### Historical Context
Co-Training was initially developed for web page classification tasks where natural feature splits exist (e.g., webpage text and hyperlink structure). Its effectiveness has since been demonstrated across various domains including natural language processing, bioinformatics, and computer vision.

---

## Theoretical Foundations

### Conceptual Explanation
The algorithm works on the principle that if two views of the data are conditionally independent given the class label, then a classifier trained on one view can provide reliable labels for instances that another classifier is uncertain about. This creates a virtuous cycle where classifiers iteratively improve each other's performance.

### Mathematical Formulation
Given:
- A small set of labeled examples $L = \{(x_1, y_1), (x_2, y_2), ..., (x_l, y_l)\}$
- A large pool of unlabeled examples $U = \{x_{l+1}, x_{l+2}, ..., x_{l+u}\}$
- Each example $x_i$ can be represented by two distinct feature sets (views): $x_i = (x_i^{(1)}, x_i^{(2)})$

The objective is to learn two classifiers $h_1$ and $h_2$ for each view such that they maximize:
$$ P(h_1(x^{(1)}) = h_2(x^{(2)}) = y) $$

### Assumptions
1. **View Sufficiency**: Each view should be sufficient to learn the target concept on its own.
2. **View Independence**: The views should be conditionally independent given the class label.
3. **Compatible Views**: The target functions over each view should predict the same labels for most examples.

---

## Algorithm Mechanics

### Step-by-Step Process
1. **Initial Training**: Train separate classifiers $h_1$ and $h_2$ on the labeled data $L$ using the respective views $x^{(1)}$ and $x^{(2)}$.
2. **Confidence Assessment**: Each classifier evaluates the unlabeled data and identifies the most confidently labeled examples.
3. **Cross-Teaching**: The most confidently labeled examples from each classifier are added to the labeled set for the other classifier.
4. **Iterative Learning**: Retrain both classifiers with their augmented labeled datasets.
5. **Termination**: Repeat steps 2-4 until a stopping criterion is met (e.g., no more confident predictions, maximum iterations).

### Training & Prediction Workflow
```python
def co_training(L, U, max_iterations=50, k=5):
    # Initialize separate feature views
    L1 = [(x[0], y) for x, y in L]  # First view
    L2 = [(x[1], y) for x, y in L]  # Second view
    
    # Train initial classifiers
    h1 = train_classifier(L1)
    h2 = train_classifier(L2)
    
    for iteration in range(max_iterations):
        if len(U) == 0:
            break
        
        # Get most confident predictions from each classifier
        U1 = [(x[0], h1.predict_proba(x[0])) for x in U]
        U2 = [(x[1], h2.predict_proba(x[1])) for x in U]
        
        # Sort by confidence
        U1_conf = sorted(U1, key=lambda x: np.max(x[1]), reverse=True)
        U2_conf = sorted(U2, key=lambda x: np.max(x[1]), reverse=True)
        
        # Select top k most confident examples
        new_L1 = [(U[i][0], np.argmax(U2_conf[i][1])) for i in range(min(k, len(U2_conf)))]
        new_L2 = [(U[i][1], np.argmax(U1_conf[i][1])) for i in range(min(k, len(U1_conf)))]
        
        # Add newly labeled examples to training sets
        L1.extend(new_L1)
        L2.extend(new_L2)
        
        # Remove labeled examples from unlabeled pool
        labeled_indices = set([U.index(x) for x, _ in U1_conf[:k]] + 
                             [U.index(x) for x, _ in U2_conf[:k]])
        U = [U[i] for i in range(len(U)) if i not in labeled_indices]
        
        # Retrain classifiers
        h1 = train_classifier(L1)
        h2 = train_classifier(L2)
    
    # Final prediction can combine both classifiers
    return h1, h2
```

---

## Implementation Details

### Code Structure
```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CoTraining(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator1, base_estimator2, k=5, max_iterations=50):
        self.base_estimator1 = base_estimator1
        self.base_estimator2 = base_estimator2
        self.k = k  # Number of examples to add in each iteration
        self.max_iterations = max_iterations
        
    def fit(self, X, y, X_unlabeled):
        # Split features into two views
        X1, X2 = self._split_views(X)
        X1_u, X2_u = self._split_views(X_unlabeled)
        
        # Filter only labeled examples
        labeled_indices = np.where(y != -1)[0]
        X1_l = X1[labeled_indices]
        X2_l = X2[labeled_indices]
        y_l = y[labeled_indices]
        
        # Initial training on labeled data
        self.base_estimator1.fit(X1_l, y_l)
        self.base_estimator2.fit(X2_l, y_l)
        
        U = list(zip(X1_u, X2_u))  # Unlabeled pool
        L1 = list(zip(X1_l, y_l))  # Labeled data for view 1
        L2 = list(zip(X2_l, y_l))  # Labeled data for view 2
        
        for _ in range(self.max_iterations):
            if len(U) == 0:
                break
            
            # Implement the co-training steps as in the workflow
            # ...
            
        return self
    
    def predict(self, X):
        X1, X2 = self._split_views(X)
        pred1 = self.base_estimator1.predict(X1)
        pred2 = self.base_estimator2.predict(X2)
        # Combine predictions (can use voting, averaging, etc.)
        return np.where(self.base_estimator1.predict_proba(X1)[:,1] > 
                       self.base_estimator2.predict_proba(X2)[:,1], pred1, pred2)
    
    def _split_views(self, X):
        # Implementation depends on how views are defined
        # For example, split features in half
        mid = X.shape[1] // 2
        return X[:, :mid], X[:, mid:]
```

### Setup Instructions
```bash
pip install numpy scikit-learn
```

---

## Hyperparameters & Optimization

- **Base Classifiers**: Algorithms used for each view (e.g., Decision Trees, Naive Bayes).
- **k**: Number of examples to label in each iteration.
- **Confidence Threshold**: Minimum confidence required to add a prediction to the labeled set.
- **Maximum Iterations**: Controls the termination of the algorithm.

**Tuning Strategies**:
- Cross-validation on the initial labeled set to select appropriate base classifiers.
- Grid search for optimal values of k and confidence thresholds.

---

## Evaluation Metrics

- **Classification Accuracy**: Performance on a held-out test set.
- **Learning Curve**: Performance improvement with increasing amounts of pseudo-labeled data.
- **Agreement Rate**: Consistency between the two view classifiers.

---

## Practical Examples

**Dataset**: News Article Classification (text content and metadata views).
**Use Case**: Email Spam Detection (email text and header information as separate views).

---

## Advanced Theory

**Multi-View Learning Extensions**:
Co-Training can be extended to more than two views, leading to multi-view learning approaches where information from multiple sources or feature representations is combined.

**Theoretical Guarantees**:
Under the view independence assumption, Blum and Mitchell proved that co-training can learn from few labeled examples if the initial weak classifiers perform better than random guessing.

---

## Advantages & Limitations

**Pros**:
- Efficiently uses unlabeled data when natural view splits exist.
- Can significantly reduce the need for labeled examples.
- Provides a principled way to combine information from multiple sources.

**Cons**:
- Requires natural or effective feature splits.
- Strong assumptions about view independence rarely hold in practice.
- Performance depends heavily on the quality of the initial classifiers.

---

## Further Reading

1. Blum, A., & Mitchell, T. (1998). "Combining labeled and unlabeled data with co-training."
2. Zhou, Z. H., & Li, M. (2005). "Tri-training: Exploiting unlabeled data using three classifiers."
3. Wang, W., & Zhou, Z. H. (2010). "A new analysis of co-training."

---

# Self-Training
*(Described in a separate README with focus on single-classifier iterative learning.)*

# Semi-Supervised SVM
*(Described in a separate README with focus on margin maximization in the presence of unlabeled data.)*
