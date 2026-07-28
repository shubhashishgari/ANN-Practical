# Practical 3: Feedforward Neural Network (using TensorFlow/Keras)

A feedforward neural network built using TensorFlow/Keras, trained on the XOR gate — a classic problem that a single neuron/perceptron cannot solve, since XOR is not linearly separable.

## Files

- `feedforward_nn.py` - the main program

## How It Works

1. **Architecture**: A `Sequential` model where data flows forward through each layer with no loops (hence "feedforward").
   - **Hidden layer**: 8 neurons, ReLU activation - learns intermediate combinations of the inputs.
   - **Output layer**: 1 neuron, sigmoid activation - outputs a probability between 0 and 1.
2. **Compiling**: Uses the Adam optimizer and binary cross-entropy loss, standard for binary classification tasks.
3. **Training Data**: The XOR gate truth table. Unlike AND/OR, XOR outputs 1 only when the two inputs differ.
4. **Before vs After Training**: Predictions are printed before training (random weights, near-random output) and after training (network has learned the XOR pattern).

## Why XOR (and Why Not Just a Single Neuron)

XOR is **not linearly separable** - no single straight line can separate the 0s and 1s in its input space. A single perceptron/neuron (Practicals 1 and 2) can only learn linearly separable patterns like AND/OR. A feedforward network solves this because the hidden layer lets the network combine inputs non-linearly before producing the final output.

## Example Output

```
Predictions BEFORE training:
Input: [0. 0.] => Predicted: 0.5 => Class: 1
Input: [0. 1.] => Predicted: 0.4835 => Class: 0
Input: [1. 0.] => Predicted: 0.3896 => Class: 0
Input: [1. 1.] => Predicted: 0.479 => Class: 0

Predictions AFTER training:
Input: [0. 0.] => Predicted: 0.1183 => Class: 0
Input: [0. 1.] => Predicted: 0.9426 => Class: 1
Input: [1. 0.] => Predicted: 0.9422 => Class: 1
Input: [1. 1.] => Predicted: 0.0572 => Class: 0
```

## Debugging Journey: Why the Network Needed Tuning

The first version of this code (4 hidden neurons, 500 epochs, no seed) gave an inconsistent result:

```
Predictions AFTER training:
Input: [0. 0.] => Predicted: 0.4999 => Class: 0
Input: [0. 1.] => Predicted: 0.9527 => Class: 1
Input: [1. 0.] => Predicted: 0.4998 => Class: 0
Input: [1. 1.] => Predicted: 0.0473 => Class: 0
```

The correct XOR output should be `0,1,1,0`. Here `Input: [1,0]` was predicted as `0` instead of `1` - the network had only partially learned the pattern.

**Why it failed:** neural network weights start off randomly, and with only 4 hidden neurons the network sometimes settles into a "local minimum" - a set of weights where the error is low-ish but not actually correct, and where it gets stuck rather than continuing to improve. 500 epochs also wasn't enough to reliably escape that.

**Fix 1 — More epochs:** Increased from 500 to 2000 to give gradient descent more chances to keep improving. This alone did not fully fix it — the network still made 1 mistake, since it was already stuck in a local minimum, not just "still learning."

**Fix 2 — More hidden neurons (4 → 8):** More neurons in the hidden layer give the network more capacity to represent the XOR pattern in different ways, making it far less likely to get stuck in a bad local minimum.

**Fix 3 — Fixed random seed (`tf.random.set_seed(42)`):** Ensures the weights start from the same point every time the code is run, so the result is reproducible — important so the code behaves consistently when your professor runs it too, instead of occasionally producing a wrong prediction due to random luck.

**Result after all three fixes:**

```
Predictions AFTER training:
Input: [0. 0.] => Predicted: 0.1183 => Class: 0
Input: [0. 1.] => Predicted: 0.9426 => Class: 1
Input: [1. 0.] => Predicted: 0.9422 => Class: 1
Input: [1. 1.] => Predicted: 0.0572 => Class: 0
```

All 4 predictions now correctly match the XOR truth table.

## Key Learnings 

- **Random weight initialization means results can vary between runs.** This is why a fixed seed is often used in teaching/demo code — it makes results reproducible, though in real-world training this randomness is normal and usually not a problem across many training runs.
- **More epochs alone doesn't guarantee a correct result** if the network is stuck in a local minimum — sometimes the network capacity (number of neurons) needs to increase too, not just the training time.
- **Hidden layer size affects representational capacity.** Too few neurons can make even a solvable problem like XOR unreliable to learn; more neurons give the network more "room" to find a correct solution, though too many can risk overfitting on larger, real-world datasets.
- **XOR is a benchmark problem** in neural network teaching specifically because it's the simplest example that *requires* a hidden layer — it's a good way to demonstrate the difference between a single neuron (Practicals 1–2) and a true feedforward network.

## How to Run

```bash
python feedforward_nn.py
```

Requires TensorFlow and NumPy:

```bash
pip install tensorflow numpy
```
