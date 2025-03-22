# Linear Regression

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
Linear regression is a foundational supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. It predicts continuous outcomes and serves as a cornerstone for understanding more complex models.

**Role in Supervised Learning**:  
Linear regression is employed for regression tasks where the goal is to predict a numerical value. It assumes a linear relationship between input features and the target variable.

### Historical Context
Developed in the early 19th century by Francis Galton and later formalized by Karl Pearson, linear regression remains widely used due to its simplicity and interpretability.

## Theoretical Foundations

### Conceptual Explanation
The algorithm identifies the best-fit line by minimizing the sum of squared residuals (errors) between observed and predicted values. This line is represented as:

$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon$

where:
- $y$: Dependent variable
- $\beta_0$: Y-intercept
- $\beta_1, \dots, \beta_n$: Coefficients of independent variables $x_1, \dots, x_n$
- $\epsilon$: Error term

### Mathematical Formulation
**Cost Function**: The objective is to minimize the Mean Squared Error (MSE):

$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\beta(x^{(i)}) - y^{(i)})^2$

where $h_\beta(x)$ is the hypothesis function.

**Optimal Coefficients**: Using the least squares method, coefficients are calculated as:

$\beta = (X^T X)^{-1} X^T y$

where $X$ is the feature matrix and $y$ is the target vector.

### Assumptions
1. Linearity between features and target
2. Independence of errors
3. Homoscedasticity (constant variance of errors)
4. Normality of residuals

## Algorithm Mechanics

### Step-by-Step Process
1. **Data Preparation**: Split data into features ($X$) and target ($y$)
2. **Parameter Initialization**: Initialize coefficients $\beta$
3. **Cost Calculation**: Compute MSE
4. **Optimization**: Adjust $\beta$ using gradient descent or closed-form solutions
5. **Prediction**: Use the learned coefficients to predict new values

### Training & Prediction Workflow
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Implementation Details

### Code Structure
```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        
    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        
    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.coef_
```

### Setup Instructions
```bash
pip install numpy scikit-learn
```

## Hyperparameters & Optimization
- **Fit Intercept**: Whether to calculate the intercept (default: `True`)
- **Normalize**: Standardize features (optional)

**Tuning Strategies**:
- Use regularization (Ridge/Lasso) to prevent overfitting

## Evaluation Metrics
- **Mean Squared Error (MSE)**: Measures average squared error
- **R-squared**: Proportion of variance explained by the model

## Practical Examples
**Dataset**: Boston Housing, Advertising Budget vs Sales  
**Use Case**: Predicting house prices based on square footage and location

## Advanced Theory
**Normal Equation Derivation**:

$\nabla J(\beta) = 0 \Rightarrow X^T (X \beta - y) = 0$

Solving yields $\beta = (X^T X)^{-1} X^T y$.

## Advantages & Limitations
**Pros**:
- Interpretable coefficients
- Computationally efficient

**Cons**:
- Sensitive to outliers
- Assumes linearity

## Further Reading
1. Hastie, T., *The Elements of Statistical Learning*
2. [Linear Regression Use Cases (TechTarget)](https://www.techtarget.com/searchenterpriseai/definition/linear-regression)
3. [Beginner's Guide to Linear Regression](https://www.turing.com/kb/beginners-guide-to-complete-mathematical-intuition-behind-linear-regression-algorithms)
4. [IBM: Linear Regression](https://www.ibm.com/topics/linear-regression)
5. [AWS: What is Linear Regression?](https://aws.amazon.com/what-is/linear-regression/)
