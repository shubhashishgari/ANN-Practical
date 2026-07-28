# Implement a Feedforward Neural Network using TensorFlow (Keras)
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

# Training data for XOR gate
# XOR is NOT linearly separable, so a single neuron/perceptron cannot learn it.
# A feedforward network with a hidden layer is needed.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Building the Feedforward Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=2, activation='relu'),   # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')               # Output layer
])
# Sequential: stacks layers linearly, data flows forward through each layer (hence "feedforward").
# Hidden layer: 4 neurons, ReLU activation, learns intermediate feature combinations.
# Output layer: 1 neuron, sigmoid activation, outputs a probability between 0 and 1.

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Predictions BEFORE training
print("Predictions BEFORE training:")
untrained_predictions = model.predict(X, verbose=0)
for i, pred in enumerate(untrained_predictions):
    print(f"Input: {X[i]} => Predicted: {round(pred.item(), 4)} => Class: {int(pred.item() >= 0.5)}")

# Training the model
model.fit(X, y, epochs=2000, verbose=0)

# Predictions AFTER training
print("\nPredictions AFTER training:")
predictions = model.predict(X, verbose=0)
for i, pred in enumerate(predictions):
    print(f"Input: {X[i]} => Predicted: {round(pred.item(), 4)} => Class: {int(pred.item() >= 0.5)}")

# Result
# Before training, the model behaves randomly.
# After training, the feedforward network learns the XOR pattern,
# which a single neuron/perceptron alone cannot do since XOR is not linearly separable.
