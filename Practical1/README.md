# Perceptron Simulation using NumPy

A simple Python implementation of a single-layer Perceptron, built from scratch using NumPy (no machine learning libraries).

## What is a Perceptron?

A perceptron is the simplest type of artificial neural network unit. It takes in inputs, multiplies them by weights, adds a bias, and passes the result through an activation function to produce an output (0 or 1). It "learns" by adjusting its weights and bias based on the error between its prediction and the actual target.

## Files

- `perceptron.py` — the main program

## How It Works

1. **Step Function**: Converts the weighted sum into a binary output (1 if the sum is ≥ 0, otherwise 0).
2. **Perceptron Class**:
   - `predict(x)` — calculates the weighted sum of inputs plus bias, then applies the step function.
   - `train(X, y, epochs)` — repeatedly loops through the training data, comparing predictions to actual targets, and updates the weights and bias to reduce error.
3. **Training Data**: The perceptron is trained on a logic gate (OR gate in this example), which is linearly separable and can be learned by a single perceptron.
4. **Testing**: After training, the perceptron predicts outputs for all input combinations.

## Example Output

```
Predictions:
Input: [0 0], Output: 0
Input: [0 1], Output: 1
Input: [1 0], Output: 1
Input: [1 1], Output: 1
```

## How to Run

```bash
python perceptron.py
```

Requires only NumPy:

```bash
pip install numpy
```

## Note

A single perceptron can only learn patterns that are **linearly separable** (like AND, OR gates). It cannot learn the XOR gate, since XOR is not linearly separable — this requires a multi-layer network (like a Multi-Layer Perceptron).
