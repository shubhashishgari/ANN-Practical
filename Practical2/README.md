# Practical 2: Single Neuron Model (Implemented Manually)

A single neuron implemented manually using only NumPy — no deep learning libraries (TensorFlow/Keras) used. The neuron's forward pass, error calculation, and weight updates (gradient descent) are all written from scratch.

## Files

- `single_neuron.py` — the main program

## How It Works

1. **Sigmoid Activation**: Squashes the neuron's weighted sum into a value between 0 and 1, making it suitable for binary classification.
2. **Manual Forward Pass**: `output = sigmoid(X · weights + bias)`
3. **Manual Backward Pass (Gradient Descent)**:
   - Computes the error: `error = y - output`
   - Computes the adjustment using the derivative of sigmoid: `error * sigmoid_derivative(output)`
   - Updates weights and bias directly: `weights += learning_rate * X.T · adjustment`
4. **Training Data**: The neuron is trained on the AND gate truth table, which is linearly separable.
5. **Before vs After Training**: Predictions are shown both before training (random weights) and after training (learned weights), to show the neuron actually learning the pattern.

## Difference from Practical 1 (Perceptron)

| | Practical 1: Perceptron | Practical 2: Single Neuron |
|---|---|---|
| Activation | Step function (hard 0/1) | Sigmoid (smooth 0–1) |
| Output | Binary only | Probability-like value |
| Learning rule | Simple error-based update | Gradient descent using derivative |

## Example Output

```
Predictions BEFORE training:
Input: [0. 0.] => Predicted: 0.269 => Class: 0
Input: [0. 1.] => Predicted: 0.3638 => Class: 0
Input: [1. 0.] => Predicted: 0.2376 => Class: 0
Input: [1. 1.] => Predicted: 0.3263 => Class: 0

Predictions AFTER training:
Input: [0. 0.] => Predicted: 0.0002 => Class: 0
Input: [0. 1.] => Predicted: 0.0555 => Class: 0
Input: [1. 0.] => Predicted: 0.0555 => Class: 0
Input: [1. 1.] => Predicted: 0.9339 => Class: 1
```

## How to Run

```bash
python single_neuron.py
```

Requires only NumPy:

```bash
pip install numpy
```
