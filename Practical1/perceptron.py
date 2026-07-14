# Simulate a Perceptron using NumPy
import numpy as np

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# Training data for OR gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

# Train the perceptron
p = Perceptron(input_size=2)
p.train(X, y, epochs=10)

# Test predictions
print("Predictions:")
for xi in X:
    print(f"Input: {xi}, Output: {p.predict(xi)}")
