# Logistic Regression

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
Logistic regression is a fundamental supervised learning algorithm used for binary and multi-class classification tasks. Despite its name, it's a classification algorithm that models the probability of an instance belonging to a particular class.

**Role in Supervised Learning**:  
Logistic regression is employed for classification tasks where the goal is to predict categorical outcomes. It estimates probabilities using a logistic function to transform a linear combination of features.

### Historical Context
Developed in the 1950s for biological applications, logistic regression became popular in statistical modeling before becoming a staple in machine learning. Its probabilistic approach makes it valuable for applications requiring probability estimates rather than just class labels.

## Theoretical Foundations

### Conceptual Explanation
Logistic regression uses the sigmoid function (logistic function) to transform linear predictions into probability values between 0 and 1. The decision boundary is represented as:

$p(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}} = \sigma(z)$

where:
- $p(y=1|x)$: Probability that the outcome is 1 given input $x$
- $\beta_0, \beta_1, \dots, \beta_n$: Model parameters
- $\sigma(z)$: Sigmoid function, where $z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$

### Mathematical Formulation
**Cost Function**: The objective is to minimize the log loss (cross-entropy):

$J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\beta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\beta(x^{(i)}))]$

where $h_\beta(x)$ is the sigmoid function applied to the linear combination of features.

**Gradient Descent**: Parameters are updated as:

$\beta_j := \beta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\beta(x^{(i)}) - y^{(i)}) x_j^{(i)}$

### Assumptions
1. Linear relationship between log-odds (logit) and features
2. Independence of observations
3. Minimal multicollinearity among features
4. Large sample size relative to the number of features

## Algorithm Mechanics

### Step-by-Step Process
1. **Data Preparation**: Split data into features ($X$) and target ($y$)
2. **Parameter Initialization**: Initialize coefficients $\beta$
3. **Prediction Calculation**: Compute sigmoid function for linear combination of features
4. **Cost Calculation**: Compute log loss
5. **Optimization**: Adjust $\beta$ using gradient descent or advanced optimization methods
6. **Classification**: Predict class based on probability threshold (typically 0.5)

### Training & Prediction Workflow
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # Probability estimates
```

## Implementation Details

### Code Structure
```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Update weights
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return [1 if p > threshold else 0 for p in probabilities]
```

### Setup Instructions
```bash
pip install numpy scikit-learn
```

## Hyperparameters & Optimization
- **C**: Inverse of regularization strength (smaller values = stronger regularization)
- **Solver**: Optimization algorithm (e.g., 'liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga')
- **Penalty**: Type of regularization ('l1', 'l2', 'elasticnet', or 'none')
- **Max Iterations**: Maximum number of iterations for solver convergence

**Tuning Strategies**:
- Use cross-validation to find optimal regularization strength
- Select different solvers based on dataset size and characteristics

## Evaluation Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve, measuring discriminative ability
- **Log Loss**: Measures the performance of a classifier where output is a probability value

## Practical Examples
**Use Cases**:
- Credit scoring and loan approval
- Disease diagnosis (present/absent)
- Email spam detection
- Customer churn prediction
- Marketing campaign response prediction

**Example Dataset**: Credit risk assessment, medical diagnosis, or marketing response prediction.

## Advanced Theory
**Multinomial Logistic Regression**: Extension for multi-class classification:

$P(y=k|x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}$

**Regularization**: Adding penalty terms to prevent overfitting:
- L1 regularization (Lasso): $J(\beta) + \lambda \sum_{j=1}^{n} |\beta_j|$
- L2 regularization (Ridge): $J(\beta) + \lambda \sum_{j=1}^{n} \beta_j^2$

## Advantages & Limitations
**Pros**:
- Outputs well-calibrated probabilities
- Interpretable coefficients
- Efficient training even for large datasets
- Less prone to overfitting compared to complex models
- Handles non-linear decision boundaries through feature engineering

**Cons**:
- Assumes linearity in log-odds space
- May underperform on complex, highly non-linear relationships
- Sensitive to highly correlated features
- Feature scaling often required for optimal performance
- Struggles with imbalanced datasets (requires adjustment)

## Further Reading
1. Hosmer, D.W., Lemeshow, S., *Applied Logistic Regression*
2. [AWS: What is Logistic Regression?](https://aws.amazon.com/what-is/logistic-regression/)
3. [Encord: What is Logistic Regression?](https://encord.com/blog/what-is-logistic-regression/)
4. [TechTarget: Logistic Regression](https://www.techtarget.com/searchbusinessanalytics/definition/logistic-regression)
5. [Coursera: Logistic Regression](https://www.coursera.org/articles/logistic-regression)
