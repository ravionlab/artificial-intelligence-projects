# Convolutional Neural Network (CNN)

## Overview & Introduction
Convolutional Neural Networks (CNNs) are specialized deep learning architectures primarily designed for processing grid-like data, particularly images. They have revolutionized computer vision and are now applied to various domains requiring pattern recognition in structured data.

**Role in Deep Learning**:
CNNs excel at tasks involving spatial hierarchies or patterns in data, automatically learning spatial feature hierarchies from low-level features (like edges) to high-level concepts (like faces).

### Historical Context
Inspired by studies of the visual cortex, CNNs were introduced in the 1980s by Kunihiko Fukushima's Neocognitron. LeCun et al. developed LeNet-5 in the 1990s for digit recognition. The breakthrough moment came in 2012 when AlexNet dramatically won the ImageNet competition, triggering the deep learning revolution in computer vision.

---

## Theoretical Foundations

### Conceptual Explanation
CNNs leverage three key architectural elements:
1. **Convolutional Layers**: Apply filters to detect local patterns
2. **Pooling Layers**: Reduce dimensionality while retaining important features
3. **Fully Connected Layers**: Interpret extracted features for final prediction

Unlike standard neural networks, CNNs maintain spatial relationships in data through local connectivity and parameter sharing, making them particularly effective for visual tasks.

### Mathematical Formulation

**Convolution Operation**:
The core operation in a CNN is discrete convolution:
$$ (f * g)(t) = \sum_{a=-\infty}^{\infty} f(a) \cdot g(t-a) $$

In 2D (for images):
$$ (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n) $$

where $I$ is the input image, $K$ is the kernel (filter), and $(i,j)$ are the coordinates.

**Feature Map**: The output of applying a filter to the input:
$$ z_{ij}^{l} = \sum_{m=0}^{f_h-1} \sum_{n=0}^{f_w-1} w_{mn}^{l} \cdot a_{(i+m)(j+n)}^{l-1} + b^{l} $$
$$ a_{ij}^{l} = f(z_{ij}^{l}) $$

where:
- $f_h$ and $f_w$ are filter height and width
- $w_{mn}^{l}$ is the weight at position $(m,n)$ of filter in layer $l$
- $a_{ij}^{l-1}$ is the activation from previous layer
- $b^{l}$ is the bias term
- $f$ is the activation function

**Pooling Operation**:
Max pooling with a 2×2 filter and stride 2:
$$ a_{ij}^{l} = \max(a_{2i,2j}^{l-1}, a_{2i,2j+1}^{l-1}, a_{2i+1,2j}^{l-1}, a_{2i+1,2j+1}^{l-1}) $$

### Key Concepts

**Filters/Kernels**: Small weight matrices that detect specific features (e.g., edges, textures).

**Stride**: The step size when sliding the filter across the input.

**Padding**: Adding zeros around the input to control output dimensions:
- "Valid padding": No padding
- "Same padding": Output has same dimensions as input

**Receptive Field**: The region in the input that affects a particular neuron in the output.

---

## Algorithm Mechanics

### CNN Architecture Components

1. **Input Layer**: Holds the raw pixel values (e.g., 224×224×3 for RGB images)

2. **Convolutional Layer**: Applies filters to extract features:
   - Output size: $((n + 2p - f)/s + 1) \times ((n + 2p - f)/s + 1) \times k$
   - Where $n$ is input size, $p$ is padding, $f$ is filter size, $s$ is stride, $k$ is number of filters

3. **Activation Layer**: Applies non-linearity (typically ReLU)

4. **Pooling Layer**: Reduces spatial dimensions while retaining important information
   - Common types: max pooling, average pooling

5. **Fully Connected Layer**: Connects all neurons to all activations from previous layer

6. **Output Layer**: Produces final predictions (e.g., softmax for classification)

### Step-by-Step Process

1. **Initialization**: Set weights for convolutional filters, fully connected layers
2. **Forward Propagation**:
   - Pass input through convolutional layers to extract features
   - Apply pooling to reduce dimensions
   - Flatten feature maps and pass through fully connected layers
3. **Loss Calculation**: Compute error between predictions and ground truth
4. **Backward Propagation**: Calculate gradients of loss with respect to weights
5. **Weight Update**: Adjust weights using an optimization algorithm
6. **Iteration**: Repeat steps 2-5 until convergence

### Training & Prediction Workflow

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Predict
y_pred = model.predict(X_test)
```

---

## Implementation Details

### Code Structure

```python
import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_shape, padding='valid', stride=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_shape = input_shape  # (channels, height, width)
        self.padding = padding
        self.stride = stride
        
        # Initialize filters with He initialization
        self.filters = np.random.randn(
            num_filters, 
            input_shape[0],  # input channels
            filter_size, 
            filter_size
        ) * np.sqrt(2 / (input_shape[0] * filter_size * filter_size))
        
        self.biases = np.zeros(num_filters)
        
    def forward(self, input_data):
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        # Calculate output dimensions
        if self.padding == 'same':
            pad = (self.filter_size - 1) // 2
            padded_input = np.pad(input_data, 
                                  ((0,0), (0,0), (pad,pad), (pad,pad)), 
                                  mode='constant')
            output_height = height
            output_width = width
        else:  # 'valid' padding
            padded_input = input_data
            output_height = (height - self.filter_size) // self.stride + 1
            output_width = (width - self.filter_size) // self.stride + 1
            
        # Initialize output volume
        output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        
        # Perform convolution
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(0, output_height, self.stride):
                    for j in range(0, output_width, self.stride):
                        # Extract region to apply filter to
                        input_region = padded_input[
                            b, 
                            :, 
                            i:i+self.filter_size, 
                            j:j+self.filter_size
                        ]
                        # Apply filter
                        output[b, f, i//self.stride, j//self.stride] = np.sum(
                            input_region * self.filters[f]
                        ) + self.biases[f]
                        
        return output
    
    def backward(self, grad_output, learning_rate):
        batch_size, num_filters, output_height, output_width = grad_output.shape
        
        # Initialize gradients
        grad_input = np.zeros_like(self.input)
        grad_filters = np.zeros_like(self.filters)
        grad_biases = np.zeros_like(self.biases)
        
        # Calculate gradients
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        # Update filter gradients
                        input_region = self.input[
                            b, 
                            :, 
                            i*self.stride:i*self.stride+self.filter_size, 
                            j*self.stride:j*self.stride+self.filter_size
                        ]
                        grad_filters[f] += input_region * grad_output[b, f, i, j]
                        grad_biases[f] += grad_output[b, f, i, j]
                        
                        # Update input gradients
                        grad_input[
                            b, 
                            :, 
                            i*self.stride:i*self.stride+self.filter_size, 
                            j*self.stride:j*self.stride+self.filter_size
                        ] += self.filters[f] * grad_output[b, f, i, j]
        
        # Update parameters
        self.filters -= learning_rate * grad_filters
        self.biases -= learning_rate * grad_biases
        
        return grad_input

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, input_data):
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        # Calculate output dimensions
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, channels, output_height, output_width))
        
        # Store max indices for backpropagation
        self.max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, output_height):
                    for j in range(0, output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Extract region
                        region = input_data[b, c, h_start:h_end, w_start:w_end]
                        
                        # Get max value and its position
                        max_val = np.max(region)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        
                        # Store in output
                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = max_idx
        
        return output
    
    def backward(self, grad_output, learning_rate=None):
        batch_size, channels, output_height, output_width = grad_output.shape
        _, _, input_height, input_width = self.input.shape
        
        # Initialize gradient with respect to input
        grad_input = np.zeros_like(self.input)
        
        # Distribute gradients
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        
                        # Get stored indices
                        max_i, max_j = self.max_indices[b, c, i, j]
                        
                        # Pass gradient to input
                        grad_input[b, c, h_start + max_i, w_start + max_j] = grad_output[b, c, i, j]
        
        return grad_input
```

### Setup Instructions

```bash
# For NumPy implementation
pip install numpy

# For TensorFlow/Keras implementation
pip install tensorflow

# For PyTorch implementation
pip install torch torchvision
```

---

## Hyperparameters & Optimization

### Key CNN Hyperparameters
- **Number of Filters**: Controls feature extraction capacity
- **Filter Size**: Determines receptive field size
- **Stride**: Affects feature map resolution
- **Padding**: Controls output dimensions
- **Pooling Size**: Determines downsampling factor
- **Learning Rate**: Controls weight update magnitude
- **Batch Size**: Number of samples processed before weight update
- **Epochs**: Number of complete passes through the training dataset

### Architectural Considerations
- **Network Depth**: Number of convolutional layers
- **Filter Progression**: Typically increasing number of filters with depth
- **Skip Connections**: As in ResNet to mitigate vanishing gradients
- **Bottleneck Layers**: Reducing dimensions before expensive operations

### Optimization Techniques
- **Data Augmentation**: Random transforms to increase effective dataset size
- **Transfer Learning**: Using pre-trained networks
- **Batch Normalization**: Normalizing layer inputs to stabilize training
- **Dropout**: Randomly deactivating neurons to prevent overfitting
- **Early Stopping**: Halting training when validation error stops improving

---

## Evaluation Metrics

- **Accuracy**: Proportion of correctly classified instances
- **Precision**: True positives divided by predicted positives
- **Recall**: True positives divided by actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: For object detection and segmentation tasks
- **Mean Average Precision (mAP)**: For object detection
- **Top-k Accuracy**: Prediction is considered correct if true label is among k most probable classes

---

## Practical Examples

### Common CNN Architectures
- **LeNet-5**: First successful CNN for digit recognition
- **AlexNet**: Breakthrough architecture that won ILSVRC 2012
- **VGGNet**: Deep network with uniform 3×3 filters
- **GoogLeNet (Inception)**: Introduced inception modules
- **ResNet**: Introduced residual connections for training very deep networks
- **MobileNet**: Efficient architecture for mobile devices

### Applications
- **Image Classification**: Identifying objects in images
- **Object Detection**: Localizing and classifying multiple objects (YOLO, SSD)
- **Semantic Segmentation**: Pixel-wise classification (U-Net, FCN)
- **Face Recognition**: Identifying individuals from facial features
- **Medical Imaging**: Diagnosing diseases from X-rays, MRIs, etc.

### Datasets
- **MNIST**: Handwritten digits
- **CIFAR-10/100**: Small images of common objects
- **ImageNet**: Large-scale dataset with millions of labeled images
- **COCO**: Common Objects in Context for object detection
- **Pascal VOC**: Visual Object Classes for object detection and segmentation

---

## Advanced Theory

### Feature Visualization
Techniques to visualize what features CNNs learn:
- **Activation Maximization**: Generate inputs that maximize neuron activations
- **Class Activation Mapping (CAM)**: Highlight regions contributing to classification
- **Gradient-weighted Class Activation Mapping (Grad-CAM)**: Uses gradients for better localization

### Adversarial Examples
Inputs with subtle perturbations that cause misclassification, revealing vulnerabilities in CNN models.

### Model Pruning and Quantization
Techniques to reduce model size and improve inference speed:
- **Weight Pruning**: Removing less important connections
- **Quantization**: Reducing precision of weights
- **Knowledge Distillation**: Training smaller networks to mimic larger ones

---

## Advantages & Limitations

### Advantages
- **Automatic Feature Extraction**: No manual feature engineering required
- **Parameter Sharing**: Efficient use of parameters
- **Translation Invariance**: Robust to position shifts in input
- **Hierarchical Learning**: Extracts features at multiple levels of abstraction

### Limitations
- **Data Requirements**: Need large labeled datasets
- **Computational Cost**: Training can be resource-intensive
- **Black Box Nature**: Limited interpretability
- **Spatial Hierarchy Assumption**: May not be optimal for non-grid data
- **Vulnerability to Adversarial Attacks**: Small perturbations can cause misclassification

---

## Further Reading
1. LeCun, Y., et al., *Gradient-based learning applied to document recognition*, Proceedings of the IEEE, 1998.
2. Krizhevsky, A., et al., *ImageNet classification with deep convolutional neural networks*, NIPS, 2012.
3. He, K., et al., *Deep residual learning for image recognition*, CVPR, 2016.
4. Goodfellow, I., et al., *Deep Learning*, MIT Press, 2016.
5. Zeiler, M.D., Fergus, R., *Visualizing and understanding convolutional networks*, ECCV, 2014.
