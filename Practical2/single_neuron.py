# Implement a Single Neuron Model manually using Python (NumPy only)
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (needed for gradient descent)
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# Initialize weights and bias randomly
np.random.seed(1)
weights = np.random.uniform(-1, 1, (2, 1))
bias = np.random.uniform(-1, 1, (1,))
learning_rate = 0.1
epochs = 10000

# Predictions BEFORE training
print("Predictions BEFORE training:")
untrained_output = sigmoid(np.dot(X, weights) + bias)
for i, pred in enumerate(untrained_output):
    print(f"Input: {X[i]} => Predicted: {round(pred.item(), 4)} => Class: {int(pred.item() >= 0.5)}")

# Training the neuron using gradient descent
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    output = sigmoid(z)

    # Calculate error
    error = y - output

    # Backward pass (gradient descent update)
    adjustment = error * sigmoid_derivative(output)
    weights += learning_rate * np.dot(X.T, adjustment)
    bias += learning_rate * np.sum(adjustment)

# Predictions AFTER training
print("\nPredictions AFTER training:")
trained_output = sigmoid(np.dot(X, weights) + bias)
for i, pred in enumerate(trained_output):
    print(f"Input: {X[i]} => Predicted: {round(pred.item(), 4)} => Class: {int(pred.item() >= 0.5)}")

# Result
# Before training, the neuron behaves randomly because the weights are initialized randomly.
# After training, it learns to approximate the AND function using gradient descent,
# adjusting weights and bias manually (without using any deep learning library).
