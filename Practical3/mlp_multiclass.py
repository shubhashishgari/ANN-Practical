# Implement a Keras MLP for Multiclass Classification
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)

# 1. Load the dataset
# Iris dataset: 150 flower samples, 4 input features, 3 output classes
# (Setosa, Versicolor, Virginica)
iris = load_iris()
X = iris.data
y = iris.target

# 2. Scale the input features
# Neural networks train much better when inputs are on a similar scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. One-hot encode the labels
# Multiclass classification needs each label turned into a vector,
# e.g. class 0 -> [1,0,0], class 1 -> [0,1,0], class 2 -> [0,0,1]
y_encoded = tf.keras.utils.to_categorical(y, num_classes=3)

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# 5. Build the MLP (Multi-Layer Perceptron)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=4, activation='relu'),   # Hidden layer
    tf.keras.layers.Dense(3, activation='softmax')               # Output layer
])
# Hidden layer: 8 neurons, ReLU activation, learns feature combinations.
# Output layer: 3 neurons (one per class), softmax activation converts
# outputs into probabilities across all 3 classes that sum to 1.

# 6. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# categorical_crossentropy: standard loss function for multiclass classification.

# 7. Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# 8. Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {round(accuracy * 100, 2)}%")

# 9. Make predictions on a few test samples
predictions = model.predict(X_test[:5], verbose=0)
print("\nSample Predictions:")
for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    actual_class = np.argmax(y_test[i])
    print(f"Predicted: {predicted_class} ({iris.target_names[predicted_class]}) "
          f"=> Actual: {actual_class} ({iris.target_names[actual_class]})")
