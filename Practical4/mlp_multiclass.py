import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)
iris = load_iris()
X = iris.data
y = iris.target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_encoded = tf.keras.utils.to_categorical(y, num_classes=3)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=4, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=100, verbose=0)


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {round(accuracy * 100, 2)}%")

predictions = model.predict(X_test[:5], verbose=0)
print("\nSample Predictions:")
for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    actual_class = np.argmax(y_test[i])
    print(f"Predicted: {predicted_class} ({iris.target_names[predicted_class]}) "
          f"=> Actual: {actual_class} ({iris.target_names[actual_class]})")
