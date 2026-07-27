# Practical 4: Keras MLP for Multiclass Classification

A simple Multi-Layer Perceptron (MLP) built with Keras (TensorFlow), trained to classify flowers into one of **three** species using the classic Iris dataset.

## Files

- `mlp_multiclass.py` — the main program

## Dataset

The **Iris dataset**: 150 flower samples, each with 4 input features (sepal length, sepal width, petal length, petal width) and 1 of 3 output classes: Setosa, Versicolor, Virginica.

## How It Works

1. **Load the dataset** — `load_iris()` gives us the 4 features (`X`) and the class label (`y`) for each of the 150 samples.
2. **Scale the features** — `StandardScaler` rescales all 4 features to a similar range (mean 0, unit variance). Neural networks train much faster and more reliably when inputs aren't on wildly different scales.
3. **One-hot encode the labels** — Multiclass problems need labels as vectors, not single numbers. E.g. class 0 becomes `[1,0,0]`, class 1 becomes `[0,1,0]`, class 2 becomes `[0,0,1]`. This is what lets the output layer compare directly against 3 output neurons.
4. **Train/test split** — 80% of the data is used to train the model, 20% is held back to fairly test it on data it hasn't seen.
5. **Build the MLP**:
   - **Hidden layer**: 8 neurons, ReLU activation — learns combinations of the 4 input features.
   - **Output layer**: 3 neurons, **softmax** activation — converts raw outputs into probabilities for each of the 3 classes that sum to 1 (e.g. `[0.05, 0.90, 0.05]` means "90% confident it's Versicolor").
6. **Compile** — uses the Adam optimizer and `categorical_crossentropy` loss, the standard loss function for multiclass classification (compare to `binary_crossentropy` used in earlier practicals for 2-class problems).
7. **Train** — `model.fit()` runs 100 epochs over the training data, adjusting weights to reduce the loss.
8. **Evaluate** — `model.evaluate()` checks accuracy on the unseen test data.
9. **Predict** — `np.argmax()` picks the class with the highest probability from the softmax output, which is then compared against the actual label.

## Key Concept for Viva: Softmax vs Sigmoid

| | Sigmoid (Practicals 1–3) | Softmax (this practical) |
|---|---|---|
| Used for | Binary classification (2 classes) | Multiclass classification (3+ classes) |
| Output | Single value between 0–1 | Vector of probabilities across all classes, summing to 1 |
| Example | `0.87` → "87% chance of class 1" | `[0.05, 0.90, 0.05]` → "90% chance of class 1 out of 3" |

## Debugging Journey: Why Scaling and Epochs Were Adjusted

The first version of this code (50 epochs, no feature scaling) gave a poor result:

```
Test Accuracy: 16.67%

Sample Predictions:
Predicted: 1 (versicolor) => Actual: 1 (versicolor)
Predicted: 2 (virginica) => Actual: 0 (setosa)
Predicted: 1 (versicolor) => Actual: 2 (virginica)
Predicted: 2 (virginica) => Actual: 1 (versicolor)
Predicted: 2 (virginica) => Actual: 1 (versicolor)
```

16.67% is roughly what you'd get from random guessing across 3 classes (1/3 ≈ 33%, and this was even worse) — the model wasn't learning at all.

**Why it failed:** the 4 Iris features are on very different scales — petal length ranges roughly 1–7 cm, but sepal width only ranges roughly 2–4.4 cm. When features have very different ranges, the network's weight updates get dominated by whichever feature has the largest raw numbers, so it struggles to learn a balanced pattern from all 4 features together.

**Fix 1 — Feature scaling:** Added `StandardScaler`, which rescales every feature to have mean 0 and standard deviation 1. Now no single feature can dominate just because of its raw number size — the network can weigh all 4 features fairly based on their actual predictive value.

**Fix 2 — More epochs:** Increased from 50 to 100. One epoch = one full pass through the training data. With only 50 passes, the model hadn't yet converged (its weights hadn't settled into a good solution). More passes gave gradient descent more chances to gradually reduce the error.

**Result after both fixes:**

```
Test Accuracy: 90.0%

Sample Predictions:
Predicted: 1 (versicolor) => Actual: 1 (versicolor)
Predicted: 0 (setosa) => Actual: 0 (setosa)
Predicted: 2 (virginica) => Actual: 2 (virginica)
Predicted: 2 (virginica) => Actual: 1 (versicolor)
Predicted: 2 (virginica) => Actual: 1 (versicolor)
```

## Key Learnings 

- **Feature scaling matters a lot** for neural networks. Unscaled features with very different ranges can make training unstable or prevent it from converging at all — this is one of the most common reasons a model "doesn't learn."
- **Epochs control how much the model learns from the same data.** Too few epochs and the model is undertrained (hasn't found a good solution yet); too many can eventually lead to overfitting (memorizing training data instead of generalizing). 100 was enough here without overfitting on such a small dataset.
- **Low accuracy isn't always a sign of a wrong architecture.** Here, the model design (8 hidden neurons, softmax output) was fine the whole time — the real issues were in data preprocessing and training duration, not the network itself. This is a common and important distinction to be able to explain: model architecture vs. data preparation vs. training configuration are three separate things to check when debugging.

## How to Run

```bash
python mlp_multiclass.py
```

Requires TensorFlow and scikit-learn:

```bash
pip install tensorflow scikit-learn
```
