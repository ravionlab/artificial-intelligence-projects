# Artificial Neural Network (ANN)

## Overview & Introduction
Artificial Neural Networks (ANNs) are computational models inspired by the structure and function of biological neural networks in the human brain. They serve as the foundation for deep learning and consist of interconnected nodes (neurons) organized in layers that process information.

**Role in Machine Learning**:
ANNs are versatile algorithms capable of addressing both regression and classification tasks. They excel at recognizing patterns in complex, high-dimensional data where traditional algorithms struggle.

### Historical Context
The concept originated in the 1940s with McCulloch and Pitts' mathematical model of neurons. After periods of reduced interest, ANNs experienced a renaissance in the 1980s with the development of backpropagation. The modern deep learning revolution began in the 2010s with advances in computing power and data availability.

---

## Theoretical Foundations

### Conceptual Explanation
ANNs learn by adjusting connection strengths (weights) between neurons. Each neuron receives inputs, applies weights, sums the results, passes this through an activation function, and transmits the output to the next layer. Through iterative weight adjustments, the network learns to map inputs to desired outputs.

The basic computation at each neuron is:
$$ z = \sum_{i=1}^{n} w_i x_i + b $$
$$ a = f(z) $$

where:
- $$ z $$: Weighted sum of inputs
- $$ w_i $$: Weight of the connection from input $$ x_i $$
- $$ b $$: Bias term
- $$ f $$: Activation function
- $$ a $$: Neuron output (activation)

### Mathematical Formulation
**Forward Propagation**:
Information flows from input to output layers. For each layer $$ l $$:
$$ Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]} $$
$$ A^{[l]} = f^{[l]}(Z^{[l]}) $$

where $$ A^{[l]} $$ is the activation of layer $$ l $$, and $$ A^{[0]} = X $$ (the input).

**Cost Function**:
For regression problems, typically Mean Squared Error:
$$ J(W, b) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 $$

For classification, often Cross-Entropy Loss:
$$ J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{C} y_j^{(i)} \log(\hat{y}_j^{(i)}) $$

where $$ C $$ is the number of classes.

**Backpropagation**:
The algorithm for computing gradients and updating weights:
$$ \frac{\partial J}{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} \cdot (A^{[l-1]})^T $$
$$ \frac{\partial J}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l](i)} $$

where $$ dZ^{[l]} $$ represents the error in layer $$ l $$.

### Activation Functions
- **Sigmoid**: $$ f(z) = \frac{1}{1 + e^{-z}} $$
- **Tanh**: $$ f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $$
- **ReLU**: $$ f(z) = \max(0, z) $$
- **Softmax**: $$ f(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $$ (for output layer in classification)

---

## Algorithm Mechanics

### Network Architecture
1. **Input Layer**: Receives raw data features
2. **Hidden Layers**: Intermediate layers that perform computations
3. **Output Layer**: Produces the final prediction

### Step-by-Step Process
1. **Initialization**: Set weights and biases to small random values
2. **Forward Propagation**: Compute outputs through the network
3. **Loss Calculation**: Measure prediction error
4. **Backpropagation**: Calculate gradients of loss with respect to weights
5. **Weight Update**: Adjust weights using an optimization algorithm
6. **Iteration**: Repeat steps 2-5 until convergence

### Training & Prediction Workflow
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create model
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(1)  # For regression (or use appropriate units for classification)
])

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predict
y_pred = model.predict(X_test)
```

---

## Implementation Details

### Code Structure
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dims):
        self.L = len(layer_dims) - 1  # Number of layers
        self.parameters = {}
        
        # Initialize weights and biases
        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    
    def forward_propagation(self, X):
        caches = []
        A = X
        
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = np.maximum(0, Z)  # ReLU activation
            caches.append((A_prev, W, b, Z))
        
        # Output layer (linear for regression)
        W = self.parameters["W" + str(self.L)]
        b = self.parameters["b" + str(self.L)]
        Z = np.dot(W, A) + b
        A = Z  # Linear activation for regression
        caches.append((A, W, b, Z))
        
        return A, caches
    
    def backward_propagation(self, y, AL, caches):
        gradients = {}
        m = y.shape[1]
        
        # Output layer
        dAL = -(y - AL)  # Derivative of MSE
        A_prev, W, b, Z = caches[self.L-1]
        dZ = dAL
        gradients["dW" + str(self.L)] = (1/m) * np.dot(dZ, A_prev.T)
        gradients["db" + str(self.L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        # Hidden layers
        for l in reversed(range(1, self.L)):
            A_prev, W, b, Z = caches[l-1]
            dZ = dA_prev * (Z > 0)  # Derivative of ReLU
            gradients["dW" + str(l)] = (1/m) * np.dot(dZ, A_prev.T)
            gradients["db" + str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA_prev = np.dot(W.T, dZ)
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] -= learning_rate * gradients["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * gradients["db" + str(l)]
    
    def train(self, X, y, learning_rate=0.01, iterations=1000):
        costs = []
        
        for i in range(iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = np.mean((y - AL) ** 2)  # MSE
            costs.append(cost)
            
            # Backward propagation
            gradients = self.backward_propagation(y, AL, caches)
            
            # Update parameters
            self.update_parameters(gradients, learning_rate)
            
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
        
        return costs
    
    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return AL
```

### Setup Instructions
```bash
# For NumPy implementation
pip install numpy

# For TensorFlow/Keras implementation
pip install tensorflow
```

---

## Hyperparameters & Optimization

### Key Hyperparameters
- **Learning Rate**: Controls step size during optimization (typically 0.001 - 0.1)
- **Number of Hidden Layers**: Depth of the network
- **Neurons per Layer**: Width of the network
- **Batch Size**: Number of samples processed before weight update
- **Epochs**: Number of complete passes through the training dataset
- **Activation Functions**: Functions applied to layer outputs

### Optimization Algorithms
- **Gradient Descent**: Basic weight update using gradients
- **Stochastic Gradient Descent (SGD)**: Updates weights using individual samples
- **Mini-batch Gradient Descent**: Updates weights using small batches
- **Adam**: Adaptive learning rate optimization
- **RMSprop**: Adaptive learning rate method that addresses vanishing/exploding gradients

### Regularization Techniques
- **Dropout**: Randomly deactivates neurons during training
- **L1/L2 Regularization**: Adds penalty terms to the loss function
- **Batch Normalization**: Normalizes layer inputs to stabilize learning

---

## Evaluation Metrics

### For Regression
- **Mean Squared Error (MSE)**: Average of squared differences
- **Mean Absolute Error (MAE)**: Average of absolute differences
- **R-squared**: Proportion of variance explained by the model

### For Classification
- **Accuracy**: Proportion of correct predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

---

## Practical Examples

### Use Cases
- **Image Classification**: Identifying objects in images
- **Natural Language Processing**: Sentiment analysis, language translation
- **Time Series Forecasting**: Stock prices, weather prediction
- **Recommender Systems**: Product recommendations in e-commerce

### Datasets
- **Regression**: Boston Housing, Bike Sharing Demand
- **Classification**: MNIST, CIFAR-10, Iris, Breast Cancer

---

## Advanced Theory

### Vanishing/Exploding Gradients
Gradients becoming extremely small or large during backpropagation, hindering learning. Addressed through careful initialization, normalization techniques, and alternative activation functions.

### Weight Initialization Strategies
- **Xavier/Glorot Initialization**: Variance scaled by fan-in and fan-out
- **He Initialization**: Variance scaled by fan-in, suited for ReLU activations

### Learning Rate Schedules
- **Step Decay**: Reducing learning rate at predefined intervals
- **Exponential Decay**: Continuously decreasing learning rate
- **Warm-up**: Gradually increasing then decreasing learning rate

---

## Advantages & Limitations

### Advantages
- **Universal Approximation**: Can approximate any continuous function
- **Feature Learning**: Automatically learns relevant features
- **Scalability**: Performance improves with more data and compute
- **Versatility**: Applicable to diverse problem domains

### Limitations
- **Data Hungry**: Requires large datasets for optimal performance
- **Computationally Intensive**: Training can be resource-demanding
- **Black Box Nature**: Limited interpretability
- **Hyperparameter Sensitivity**: Performance depends on careful tuning

---

## Further Reading
1. Goodfellow, I., et al., *Deep Learning*. MIT Press, 2016.
2. Nielsen, M., *Neural Networks and Deep Learning*. Online Book.
3. Aggarwal, C. C., *Neural Networks and Deep Learning*. Springer, 2018.
