# Decision Trees

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
Decision Trees are versatile supervised learning algorithms used for both classification and regression tasks. They create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. The model resembles a tree structure, with nodes, branches, and leaves.

**Role in Supervised Learning**:  
Decision Trees partition the feature space into regions, each assigned a prediction value. They serve as fundamental building blocks for more advanced ensemble methods like Random Forests and Gradient Boosting.

### Historical Context
The concept of decision trees has roots in decision theory and has been used in various forms since the 1960s. Notable algorithms include ID3 (Iterative Dichotomiser 3) by Ross Quinlan in the 1970s, followed by C4.5 and CART (Classification and Regression Trees) by Breiman et al. in the 1980s.

## Theoretical Foundations

### Conceptual Explanation
A Decision Tree recursively splits the data into subsets based on the value of an attribute, aiming to create subsets that are as homogeneous as possible with respect to the target variable. Each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label or a continuous value.

### Mathematical Formulation
**Split Criteria for Classification**:

1. **Gini Impurity**: Measures the probability of misclassifying a randomly chosen element:
   $Gini(D) = 1 - \sum_{i=1}^{c} (p_i)^2$

2. **Entropy**: Measures the level of disorder or uncertainty:
   $Entropy(D) = -\sum_{i=1}^{c} p_i \log_2(p_i)$

3. **Information Gain**: Reduction in entropy after splitting:
   $IG(D, A) = Entropy(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} Entropy(D_v)$

**Split Criteria for Regression**:

1. **Mean Squared Error (MSE)**:
   $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$

2. **Mean Absolute Error (MAE)**:
   $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \bar{y}|$

### Assumptions
1. The entire feature space can be recursively partitioned using axis-aligned splits
2. The training data can represent the underlying distribution adequately
3. Simpler trees generalize better (Occam's razor principle)

## Algorithm Mechanics

### Step-by-Step Process
1. **Start at the Root**: Begin with the entire dataset
2. **Find Best Split**: Evaluate all features and possible split points using a criterion (Gini, entropy, MSE)
3. **Split the Data**: Divide the data based on the best split
4. **Recursive Splitting**: Repeat steps 2-3 for each child node until stopping criteria are met
5. **Create Leaf Nodes**: Assign class labels (classification) or average values (regression) to leaf nodes

### Training & Prediction Workflow
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# For classification
clf_model = DecisionTreeClassifier(criterion='gini', max_depth=5)
clf_model.fit(X_train, y_train)
y_pred_clf = clf_model.predict(X_test)

# For regression
reg_model = DecisionTreeRegressor(criterion='mse', max_depth=5)
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)
```

## Implementation Details

### Code Structure
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

class MyDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion
        )
        
    def fit(self, X, y):
        return self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def visualize(self, feature_names=None, class_names=None, filename='tree.dot'):
        export_graphviz(
            self.model,
            out_file=filename,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        print(f"Tree exported to {filename}. Convert to PDF/PNG using: dot -Tpdf {filename} -o tree.pdf")
```

### Setup Instructions
```bash
pip install numpy scikit-learn graphviz
```

## Hyperparameters & Optimization
- **max_depth**: Maximum depth of the tree
- **min_samples_split**: Minimum number of samples required to split an internal node
- **min_samples_leaf**: Minimum number of samples required to be at a leaf node
- **max_features**: Number of features to consider when looking for the best split
- **criterion**: Function to measure the quality of a split ('gini'/'entropy' for classification, 'mse'/'mae' for regression)
- **splitter**: Strategy used to choose the split at each node ('best'/'random')

**Tuning Strategies**:
- Grid search with cross-validation
- Start with small trees and gradually increase complexity
- Use cost-complexity pruning to find optimal tree size

## Evaluation Metrics
- **Classification**: Accuracy, precision, recall, F1-score, confusion matrix
- **Regression**: Mean squared error, mean absolute error, R-squared
- **Tree Complexity**: Number of nodes, tree depth

## Practical Examples
**Use Cases**:
- Credit risk assessment
- Medical diagnosis
- Customer segmentation
- Fraud detection
- Recommendation systems
- Predictive maintenance

**Example Datasets**: Titanic survival prediction, loan approval, sales forecasting.

## Advanced Theory
**CART Algorithm**: Uses a greedy approach to find the best split:
1. For each feature, examine all possible split points
2. Choose the split that minimizes the weighted sum of child node impurities

**Pruning Techniques**:
1. **Pre-pruning**: Stop tree growth early
2. **Post-pruning**: Build full tree, then remove nodes to reduce complexity
3. **Cost-complexity pruning**: Gradually remove nodes that provide minimal complexity reduction

**Handling Missing Values**:
1. **Surrogate splits**: Use alternative splitting rules when the primary split feature is missing
2. **Weighted splits**: Distribute observations with missing values across child nodes

## Advantages & Limitations
**Pros**:
- Easy to understand and interpret
- Requires little data preprocessing (no normalization needed)
- Can handle both numerical and categorical data
- Handles non-linear relationships well
- Automatically performs feature selection
- Robust to outliers

**Cons**:
- Prone to overfitting (especially with deep trees)
- Can create biased trees if some classes dominate
- Instability: small variations in data can result in different trees
- Struggles with capturing complex relationships with a single tree
- Diagonal decision boundaries are difficult to represent efficiently

## Further Reading
1. Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J., *Classification and Regression Trees*
2. Quinlan, J.R., *C4.5: Programs for Machine Learning*
3. [IBM: Decision Trees](https://www.ibm.com/topics/decision-trees)
4. [TechTarget: Decision Tree in Machine Learning](https://www.techtarget.com/searchenterpriseai/definition/decision-tree-in-machine-learning)
5. [Simplilearn: The Power of Decision Trees in Machine Learning](https://www.simplilearn.com/the-power-of-decision-trees-in-machine-learning-article)
6. [KDnuggets: Decision Tree Algorithm Explained](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)
