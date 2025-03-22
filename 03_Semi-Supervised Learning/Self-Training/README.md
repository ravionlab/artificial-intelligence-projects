# Self-Training

## Overview & Introduction
Self-Training is one of the simplest and most intuitive semi-supervised learning algorithms, operating on a single classifier that iteratively improves itself by incorporating its own most confident predictions into its training set. This approach is sometimes called self-teaching or self-labeling and is considered a wrapper method in semi-supervised learning.

**Role in Semi-Supervised Learning**:
Self-Training serves as a fundamental bridge between supervised and unsupervised learning by leveraging a large pool of unlabeled data alongside a limited set of labeled examples to improve model performance.

### Historical Context
Self-Training has deep roots in machine learning, with early applications dating back to the 1960s. It gained renewed attention in the early 2000s with the increasing availability of unlabeled data and has been successfully applied to various domains including natural language processing, image recognition, and bioinformatics.

---

## Theoretical Foundations

### Conceptual Explanation
The core idea behind Self-Training is simple: a model trained on labeled data makes predictions on unlabeled data, and the most confident predictions are then added to the training set as if they were true labels. This process iterates, with the model gradually learning from its own predictions.

### Mathematical Formulation
Given:
- A small set of labeled examples $L = \{(x_1, y_1), (x_2, y_2), ..., (x_l, y_l)\}$
- A large pool of unlabeled examples $U = \{x_{l+1}, x_{l+2}, ..., x_{l+u}\}$
- A base classifier $h$ and a confidence measure $c(h, x)$

The objective is to iteratively expand $L$ by adding the most confidently predicted examples from $U$ until a stopping criterion is met.

### Assumptions
1. **Low-Density Separation**: Decision boundaries should lie in low-density regions of the input space.
2. **Cluster Assumption**: Points in the same cluster likely belong to the same class.
3. **Smoothness Assumption**: Points that are close to each other are likely to have the same label.

---

## Algorithm Mechanics

### Step-by-Step Process
1. **Initial Training**: Train a classifier $h$ on the labeled dataset $L$.
2. **Prediction**: Use $h$ to make predictions on the unlabeled dataset $U$.
3. **Confidence Assessment**: Calculate confidence scores $c(h, x)$ for each prediction.
4. **Selection**: Select subset $S \subset U$ containing examples with confidence above a threshold $\theta$.
5. **Augmentation**: Add selected examples with their predicted labels to the labeled set: $L = L \cup \{(x, h(x)) | x \in S\}$.
6. **Removal**: Remove selected examples from the unlabeled pool: $U = U \setminus S$.
7. **Retraining**: Retrain the classifier on the augmented labeled set.
8. **Iteration**: Repeat steps 2-7 until a stopping criterion is met.

### Training & Prediction Workflow
```python
def self_training(X_labeled, y_labeled, X_unlabeled, base_classifier, 
                 confidence_threshold=0.8, max_iterations=50):
    # Initialize labeled and unlabeled pools
    L = list(zip(X_labeled, y_labeled))
    U = list(X_unlabeled)
    
    # Create a copy of the base classifier
    model = clone(base_classifier)
    
    for iteration in range(max_iterations):
        # Train model on current labeled data
        X_l, y_l = zip(*L)
        model.fit(X_l, y_l)
        
        if len(U) == 0:
            break
        
        # Predict on unlabeled data
        X_u = np.array(U)
        predictions = model.predict(X_u)
        
        # Get confidence scores (implementation depends on classifier)
        if hasattr(model, "predict_proba"):
            confidences = np.max(model.predict_proba(X_u), axis=1)
        else:
            # For models without probabilistic output, use distance to decision boundary
            confidences = np.abs(model.decision_function(X_u))
        
        # Find examples with confidence above threshold
        confident_indices = np.where(confidences >= confidence_threshold)[0]
        
        if len(confident_indices) == 0:
            # No confident predictions, terminate early
            break
        
        # Add confident predictions to labeled set
        for idx in confident_indices:
            L.append((U[idx], predictions[idx]))
        
        # Remove from unlabeled pool
        U = [U[i] for i in range(len(U)) if i not in confident_indices]
        
    return model
```

---

## Implementation Details

### Code Structure
```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone

class SelfTrainingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, threshold=0.8, max_iter=50, 
                 increment=10, verbose=0):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.max_iter = max_iter
        self.increment = increment  # Number of examples to add per iteration
        self.verbose = verbose
        
    def fit(self, X, y):
        # Separate labeled and unlabeled data
        labeled_mask = y != -1
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        X_unlabeled = X[~labeled_mask]
        
        # Keep track of labeled and pseudo-labeled data
        self.X_train = X_labeled.copy()
        self.y_train = y_labeled.copy()
        self.unlabeled_indices = np.where(~labeled_mask)[0]
        
        # Initialize estimator
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(self.X_train, self.y_train)
        
        n_iter = 0
        while n_iter < self.max_iter and len(self.unlabeled_indices) > 0:
            # Predict on unlabeled data
            probabilities = self.estimator_.predict_proba(X_unlabeled)
            max_proba = np.max(probabilities, axis=1)
            pred_labels = np.argmax(probabilities, axis=1)
            
            # Find most confident predictions
            confident_idx = np.argsort(max_proba)[::-1]
            confident_idx = confident_idx[:min(self.increment, len(confident_idx))]
            confident_idx = confident_idx[max_proba[confident_idx] >= self.threshold]
            
            if len(confident_idx) == 0:
                if self.verbose:
                    print(f"No confident predictions at iteration {n_iter}")
                break
                
            if self.verbose:
                print(f"Adding {len(confident_idx)} new labels at iteration {n_iter}")
                
            # Add to labeled dataset
            new_X = X_unlabeled[confident_idx]
            new_y = pred_labels[confident_idx]
            self.X_train = np.vstack((self.X_train, new_X))
            self.y_train = np.append(self.y_train, new_y)
            
            # Remove from unlabeled dataset
            X_unlabeled = np.delete(X_unlabeled, confident_idx, axis=0)
            self.unlabeled_indices = np.delete(self.unlabeled_indices, confident_idx)
            pred_labels = np.delete(pred_labels, confident_idx)
            
            # Retrain the model
            self.estimator_ = clone(self.base_estimator)
            self.estimator_.fit(self.X_train, self.y_train)
            
            n_iter += 1
            
        return self
    
    def predict(self, X):
        return self.estimator_.predict(X)
    
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
```

### Setup Instructions
```bash
pip install numpy scikit-learn
```

---

## Hyperparameters & Optimization

- **Base Classifier**: Any classifier capable of providing confidence scores.
- **Confidence Threshold**: Minimum confidence required to add a prediction to the labeled set.
- **Selection Strategy**: How many examples to add per iteration (fixed number or adaptive).
- **Stopping Criterion**: Maximum iterations or convergence condition.

**Tuning Strategies**:
- Use validation set to optimize confidence threshold.
- Implement early stopping if performance on validation set starts to degrade.
- Experiment with different base classifiers to find the most suitable for the task.

---

## Evaluation Metrics

- **Classification Accuracy**: Performance on a held-out test set.
- **Pseudo-Labeling Accuracy**: Correctness of assigned pseudo-labels (if ground truth is available).
- **Learning Curve**: Performance improvement with increasing amounts of pseudo-labeled data.

---

## Practical Examples

**Dataset**: Text Classification with small labeled corpus and large unlabeled corpus.
**Use Case**: Medical Image Classification where annotation requires expensive expert knowledge.

---

## Advanced Theory

**Theoretical Analysis**:
Self-Training can be analyzed within the framework of Expectation-Maximization (EM) algorithms, where the model parameters are the "parameters" and the unknown labels are the "missing data."

**Connections to Other Methods**:
Self-Training shares similarities with Expectation-Maximization and can be viewed as a special case of co-training where both classifiers are identical.

---

## Advantages & Limitations

**Pros**:
- Simplicity: Easy to implement and understand.
- Flexibility: Works with any base classifier.
- Universality: Applicable to many types of problems.

**Cons**:
- Error Propagation: Incorrect pseudo-labels can reinforce themselves.
- Confirmation Bias: The model tends to strengthen its initial biases.
- Parameter Sensitivity: Performance heavily depends on confidence threshold.

---

## Further Reading

1. Yarowsky, D. (1995). "Unsupervised Word Sense Disambiguation Rivaling Supervised Methods."
2. Triguero, I., García, S., & Herrera, F. (2015). "Self-labeled techniques for semi-supervised learning: taxonomy, software and empirical study."
3. Chapelle, O., Schölkopf, B., & Zien, A. (2006). "Semi-Supervised Learning."

---

# Co-Training
*(Described in a separate README with focus on multi-view learning and classifier agreements.)*

# Semi-Supervised SVM
*(Described in a separate README with focus on margin maximization in the presence of unlabeled data.)*
