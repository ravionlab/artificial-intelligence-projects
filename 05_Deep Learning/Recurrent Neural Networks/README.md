# Recurrent Neural Networks (RNN)

A comprehensive guide to Recurrent Neural Networks (RNNs), covering theoretical foundations, architectures, implementation details, hyperparameters, and evaluation metrics.

---

## Table of Contents

- [Overview & Introduction](#overview--introduction)
- [Theoretical Foundations](#theoretical-foundations)
  - [Conceptual Explanation](#conceptual-explanation)
  - [Mathematical Formulation](#mathematical-formulation)
    - [Basic RNN Cell](#basic-rnn-cell)
    - [LSTM Cell](#lstm-cell)
    - [GRU Cell](#gru-cell)
  - [Backpropagation Through Time (BPTT)](#backpropagation-through-time-bptt)
- [Algorithm Mechanics](#algorithm-mechanics)
  - [RNN Architectures](#rnn-architectures)
  - [Step-by-Step Process](#step-by-step-process)
  - [TensorFlow/Keras Example](#tensorflowkeras-example)
- [Implementation Details](#implementation-details)
  - [Code Structure Overview](#code-structure-overview)
  - [Setup Instructions](#setup-instructions)
- [Hyperparameters & Optimization](#hyperparameters--optimization)
- [Evaluation Metrics](#evaluation-metrics)

---

## Overview & Introduction

Recurrent Neural Networks (RNNs) are specialized neural network architectures designed to recognize patterns in sequential data. Unlike traditional feedforward networks, RNNs include cycles via recurrent connections that provide an internal "memory," making them well-suited for tasks where sequence order matters.

### Key Applications
- **Sequence Modeling**: Time-series prediction and natural language processing.
- **Speech Recognition**: Interpreting audio signals.
- **Text Generation**: Generating human-like text.

### Historical Context
- **1980s**: Conceptualization of RNNs.
- **Challenges**: Early RNNs suffered from vanishing/exploding gradients.
- **Breakthroughs**: The development of LSTM (1997) and GRU (2014) architectures addressed these issues, enabling robust sequence modeling.

---

## Theoretical Foundations

### Conceptual Explanation

RNNs process sequences by maintaining a hidden state that evolves over time. At each time step, the network:
1. **Receives** the current input and the previous hidden state.
2. **Updates** the hidden state based on this combined information.
3. **Generates** an output from the updated hidden state.
4. **Propagates** the hidden state to the next time step.

### Mathematical Formulation

#### Basic RNN Cell
The computation for a basic RNN cell at time step *t* is as follows:
\[
h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
\]
\[
y_t = g(W_{hy} h_t + b_y)
\]
- **\(f\)**: Activation function (typically tanh or ReLU).
- **\(g\)**: Task-specific activation function (e.g., softmax for classification).

#### LSTM Cell
LSTM networks introduce gating mechanisms to handle long-term dependencies:

1. **Forget Gate**:
   \[
   f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
   \]
2. **Input Gate & Candidate Memory**:
   \[
   i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
   \]
   \[
   \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
   \]
3. **Cell State Update**:
   \[
   C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
   \]
4. **Output Gate**:
   \[
   o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   \]
   \[
   h_t = o_t * \tanh(C_t)
   \]

#### GRU Cell
GRUs simplify the gating mechanism:

1. **Update Gate**:
   \[
   z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
   \]
2. **Reset Gate**:
   \[
   r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
   \]
3. **Candidate Activation**:
   \[
   \tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t] + b)
   \]
4. **Hidden State Update**:
   \[
   h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
   \]

### Backpropagation Through Time (BPTT)

BPTT adapts standard backpropagation to RNNs by “unfolding” the network over time:
\[
\frac{\partial E}{\partial W} = \sum_{t=1}^{T} \frac{\partial E_t}{\partial W}
\]
This method applies the chain rule across time steps, allowing the model to learn from temporal dependencies.

---

## Algorithm Mechanics

### RNN Architectures

- **One-to-One**: Standard feedforward network.
- **One-to-Many**: Single input producing sequence output (e.g., image captioning).
- **Many-to-One**: Sequence input yielding a single output (e.g., sentiment analysis).
- **Many-to-Many (Synchronized)**: Sequence input and output of the same length.
- **Many-to-Many (Encoder-Decoder)**: Input and output sequences can differ in length (e.g., machine translation).

### Step-by-Step Process

1. **Initialize** the hidden state (typically with zeros).
2. **Iterate** over each time step:
   - Process the input \( x_t \) and previous hidden state \( h_{t-1} \).
   - Update the hidden state \( h_t \) and generate the output \( y_t \).
3. **Compute Loss** by comparing outputs with targets.
4. **Backpropagate** the error through time.
5. **Update Weights** using gradient descent or its variants.

### TensorFlow/Keras Example

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Define a Many-to-One RNN model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, features)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression or appropriate classification output
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions
y_pred = model.predict(X_test)
```

---

## Implementation Details

### Code Structure Overview

Below is an example of a simple RNN implementation using NumPy:

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # Store model dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def forward(self, inputs):
        """
        Performs a forward pass through the RNN.
        
        Args:
            inputs: List of input vectors for each time step [x_1, x_2, ..., x_T]
            
        Returns:
            h_states: List of hidden states
            y_preds: List of predictions
        """
        h_states = []
        y_preds = []
        h_prev = np.zeros((self.hidden_size, 1))
        
        for x in inputs:
            # Reshape input if necessary
            x = x.reshape(-1, 1) if x.ndim == 1 else x
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h_prev) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            h_states.append(h)
            y_preds.append(y)
            h_prev = h
        
        return h_states, y_preds
    
    def backward(self, inputs, targets, h_states, y_preds, learning_rate=0.01):
        """
        Backward pass through time (BPTT).
        (Detailed implementation as provided in the original document.)
        """
        # Initialize gradient accumulators
        # (Implementation details go here.)
        pass
    
    def train_sequence(self, inputs, targets, epochs=100, learning_rate=0.01):
        """
        Train the RNN on a sequence of inputs and targets.
        """
        losses = []
        for epoch in range(epochs):
            h_states, y_preds = self.forward(inputs)
            loss = self.backward(inputs, targets, h_states, y_preds, learning_rate)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        return losses

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        # LSTM implementation with gating mechanisms goes here.
        pass
```

### Setup Instructions

- **NumPy Implementation**:
  ```bash
  pip install numpy
  ```
- **TensorFlow/Keras Implementation**:
  ```bash
  pip install tensorflow
  ```
- **PyTorch Implementation**:
  ```bash
  pip install torch
  ```

---

## Hyperparameters & Optimization

### Key Hyperparameters
- **Hidden Layer Size**: Number of units in the hidden layer.
- **Sequence Length/Truncation**: Maximum number of time steps for BPTT.
- **Learning Rate**: Step size for gradient updates.
- **Number of Layers**: Depth of the RNN (stacked layers).
- **Dropout Rate**: Regularization to prevent overfitting.
- **Gradient Clipping Threshold**: Limits gradient magnitude to stabilize training.

### Optimization Techniques
- **Gradient Clipping**: Prevents exploding gradients by scaling down large gradients.
- **Layer Normalization**: Normalizes inputs across a layer to improve convergence.
- **Bidirectional Processing**: Uses information from both past and future (when applicable).
- **Teacher Forcing & Scheduled Sampling**: Techniques that help guide training by leveraging ground truth inputs.
- **Attention Mechanisms**: Allow the model to focus on key parts of the input sequence.

### Handling Long Sequences
- **Truncated BPTT**: Limit backpropagation to a fixed number of time steps.
- **Stateful RNNs**: Preserve hidden states across batches for long sequences.
- **Hierarchical RNNs**: Process sequences at multiple time scales.

---

## Evaluation Metrics

### For Sequence Classification
- **Accuracy**: The proportion of correctly classified sequences.
- **Precision, Recall, F1 Score**: Standard classification metrics.
- **Confusion Matrix**: Detailed breakdown of model performance across classes.

### For Sequence Generation
- **Perplexity**: How well the model predicts the next element in a sequence.
- **BLEU Score**: Evaluates generated text against reference text (common in NLP).
- **ROUGE Score**: Focuses on recall in summarization tasks.
- **Word Error Rate (WER)**: Commonly used in speech recognition evaluations.

---
