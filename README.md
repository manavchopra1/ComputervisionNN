# Computer Vision Neural Network Project

## Overview

This project demonstrates the implementation of a neural network for digit recognition using two different approaches:
1. **NumPy Implementation**: Building a neural network from scratch using only NumPy
2. **TensorFlow/Keras Implementation**: Using modern deep learning frameworks

The project uses the MNIST dataset to classify handwritten digits (0-9).

## Dataset

- **MNIST Dataset**: 28x28 pixel grayscale images of handwritten digits
- **Training Data**: 60,000 images
- **Test Data**: 10,000 images
- **Classes**: 10 (digits 0-9)

## Approach 1: NumPy Implementation (From Scratch)

### Architecture
- **Input Layer**: 784 neurons (28×28 flattened pixels)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation

### Key Components

#### 1. Data Preprocessing
```python
# Normalize pixel values to [0,1] range
X_train = X_train / 255.0
X_dev = X_dev / 255.0
```

#### 2. Neural Network Functions
- `init_params()`: Initialize weights and biases
- `ReLU()`: Rectified Linear Unit activation function
- `softmax()`: Softmax activation for output layer
- `forward_prop()`: Forward propagation
- `backward_prop()`: Backward propagation with gradient computation
- `update_params()`: Parameter updates using gradient descent

#### 3. Training Process
```python
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
```

### Challenges with NumPy Implementation

#### 1. Extensive Hyperparameter Tuning Required
- **Learning Rate**: Required careful tuning (0.10 used)
- **Network Architecture**: Limited to simple feedforward network
- **Training Iterations**: 500 iterations needed for convergence
- **Initialization**: Manual weight initialization with random values

#### 2. Performance Limitations
- **Final Accuracy**: ~83.12% on training data
- **Training Time**: Slower due to manual implementation
- **Scalability**: Limited to simple architectures

#### 3. Manual Implementation Challenges
- **Gradient Computation**: Manual implementation of backpropagation
- **Activation Functions**: Manual implementation of ReLU and Softmax
- **Loss Function**: Manual implementation of cross-entropy loss
- **Optimization**: Basic gradient descent without advanced optimizers

## Approach 2: TensorFlow/Keras Implementation

### Architecture
- **Input Layer**: Flatten layer (784 neurons)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation

### Key Advantages

#### 1. Built-in Optimizations
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

#### 2. Advanced Training Features
- **Adam Optimizer**: Adaptive learning rate
- **Categorical Crossentropy**: Optimized loss function
- **Batch Processing**: Efficient training with mini-batches
- **Validation Split**: Built-in validation during training

#### 3. Superior Performance
- **Final Accuracy**: 97.48% on test data
- **Training Time**: Significantly faster
- **Convergence**: Achieved in just 10 epochs

## Performance Comparison

| Metric | NumPy Implementation | TensorFlow/Keras Implementation |
|--------|---------------------|--------------------------------|
| **Training Accuracy** | ~83.12% | ~99.30% |
| **Test Accuracy** | Not evaluated | 97.48% |
| **Training Time** | Slower (500 iterations) | Faster (10 epochs) |
| **Hyperparameter Tuning** | Extensive manual tuning required | Minimal tuning needed |
| **Implementation Complexity** | High (manual backpropagation) | Low (built-in functions) |
| **Scalability** | Limited | Highly scalable |

## Key Findings

### 1. Framework Efficiency
- **TensorFlow/Keras** provides significantly better performance with minimal effort
- **NumPy** implementation requires extensive hyperparameter tuning and manual optimization

### 2. Accuracy Improvement
- **TensorFlow/Keras**: 97.48% test accuracy
- **NumPy**: ~83% training accuracy (limited evaluation)

### 3. Development Time
- **TensorFlow/Keras**: Rapid development with built-in optimizations
- **NumPy**: Time-consuming manual implementation and tuning

### 4. Learning Value
- **NumPy**: Excellent for understanding neural network fundamentals
- **TensorFlow/Keras**: Better for production and research applications

## Technical Details

### NumPy Implementation Challenges
1. **Manual Gradient Computation**: Required implementing backpropagation from scratch
2. **Hyperparameter Sensitivity**: Learning rate and network size required extensive tuning
3. **Limited Architecture**: Simple feedforward network without advanced features
4. **Performance Bottlenecks**: No vectorization optimizations

### TensorFlow/Keras Advantages
1. **Optimized Operations**: Built-in vectorized operations
2. **Advanced Optimizers**: Adam optimizer with adaptive learning rates
3. **Regularization**: Built-in dropout and regularization techniques
4. **GPU Acceleration**: Automatic GPU utilization when available

## Conclusion

This project demonstrates the evolution of neural network implementation from manual NumPy coding to modern deep learning frameworks. While the NumPy implementation provides valuable insights into neural network fundamentals, the TensorFlow/Keras approach offers superior performance, faster development, and better scalability for real-world applications.

The comparison highlights the importance of using appropriate tools for different use cases:
- **Educational/Understanding**: NumPy implementation
- **Production/Research**: TensorFlow/Keras implementation

## Files

- `computervisionNN.ipynb`: Complete Jupyter notebook with both implementations
- `README.md`: This documentation file

## Requirements

```python
numpy
pandas
matplotlib
seaborn
tensorflow
keras
```

## Usage

1. Open `computervisionNN.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells to see both implementations
3. Compare the performance and implementation complexity

## Future Improvements

1. **CNN Implementation**: Add Convolutional Neural Network for better image recognition
2. **Data Augmentation**: Implement data augmentation techniques
3. **Advanced Architectures**: Experiment with ResNet, VGG, or other architectures
4. **Hyperparameter Optimization**: Use automated hyperparameter tuning
5. **Model Deployment**: Add model saving and deployment capabilities