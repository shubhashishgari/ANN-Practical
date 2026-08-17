# Practical 6: Keras MLP for Regression

An MLP built with Keras, trained to predict disease progression from patient health measurements, a regression problem. Compares 3 activation functions and 3 loss functions to see which combination performs best.

## Files

- `mlp_regression.py`
- `loss_curves_activation.png`: generated when the script runs
- `loss_curves_loss_function.png`: generated when the script runs

## Dataset

**Diabetes dataset** (`sklearn.datasets.load_diabetes`): 442 patient records, 10 numeric input features (age, sex, BMI, blood pressure, and 6 blood serum measurements), predicting a continuous target, a quantitative measure of disease progression one year after baseline. Built into scikit-learn, loads with no internet connection or manual download required. All features are already numeric with no missing values, so no encoding or imputation was needed.

**Why this dataset fits a regression MLP:** the target is continuous (not a category), the relationship between the 10 features and the target is not simply linear (justifying a neural network over plain linear regression), and the small feature count (10) keeps the model simple enough to clearly observe the effect of changing activation/loss functions, which is the actual point of this practical.

## How It Works

1. **Load & explore:** loads the dataset, prints summary statistics and confirms there are no missing values.
2. **Scale features and target:** `StandardScaler` is applied to both the input features *and* the target. Scaling the target matters here specifically because Huber loss's behavior depends on the scale of the errors (see justification below).
3. **Train/test split:** 80% training, 20% held out for testing. A further 20% of the training set is set aside during `fit()` as a validation set, purely for plotting loss curves.
4. **Model architecture:** `Input(10) → Dense(64) → Dense(32) → Dense(1, linear)`. Two hidden layers as required. The output layer uses `linear` activation (not sigmoid/softmax) because regression output must be able to take any real value, not just 0–1 or a probability.
5. **Activation experiment:** trains 3 identical models (same architecture, optimizer, epochs, batch size, loss=MSE), varying only the hidden layer activation: ReLU, Sigmoid, Tanh.
6. **Loss function experiment:** takes the best-performing activation from step 5, and trains 2 more models with that same activation but MAE and Huber loss instead of MSE (MSE was already covered in step 5, so together this covers all 3 required loss functions without retraining a duplicate model).
7. **Evaluate:** for every model, predictions are inverse-scaled back to the original units, then MSE, RMSE, MAE, and R² are computed on the test set.
8. **Compare:** builds a results table across all 5 trained models, and plots validation loss curves for both experiments.

## Why These Activation Functions

- **ReLU**: standard default for hidden layers, cheap to compute, avoids vanishing gradients, the usual first choice.
- **Sigmoid**: included as a contrast, saturates at extreme values, which can slow learning in deeper networks, but was worth testing given this network is shallow (2 hidden layers) where that weakness matters less.
- **Tanh**: zero-centered version of sigmoid, sometimes trains faster than sigmoid since its output isn't always positive.

## Why These Loss Functions

- **MSE (Mean Squared Error)**: the standard regression loss, penalizes larger errors disproportionately, appropriate as a baseline since there's no strong reason to expect extreme outliers in this dataset.
- **MAE (Mean Absolute Error)**: penalizes all errors linearly rather than squaring them, making it less sensitive to outliers than MSE, useful to test since medical measurement data like this can have some patient outliers.
- **Huber Loss**: a middle ground, behaves like MSE for small errors and like MAE for large ones, combining MSE's smooth gradient near zero with MAE's robustness to outliers. Included specifically to see whether the dataset actually benefits from that outlier-robustness or not.

## Results (actual output from running the script)

```
Comparison Table:
        Experiment Activation Loss Function   MAE  RMSE     R2
   Activation-relu       relu           mse 44.53 58.47 0.3547
Activation-sigmoid    sigmoid           mse 41.76 53.09 0.4680
   Activation-tanh       tanh           mse 47.04 59.91 0.3226
          Loss-mae    sigmoid           mae 42.68 54.03 0.4489
        Loss-huber    sigmoid         huber 41.94 53.48 0.4602

Best overall model: activation=sigmoid, loss=mse, R2=0.468, MAE=41.76, RMSE=53.09
```

## What Went Wrong and What I Learned

**Bug: duplicate row in results table.** The first version appended the best-activation result to the results list a second time after already including it from the activation loop, so it appeared twice in the comparison table. Fixed by removing the redundant append. The sigmoid+MSE model already represents that combination from the activation experiment, so it doesn't need to be added again for the loss experiment.

**Finding: ReLU and Tanh overfit on this dataset, Sigmoid didn't.**

Looking at `loss_curves_activation.png`, the ReLU and Tanh validation loss curves both drop initially, then climb steadily back up after roughly epoch 20, classic overfitting, where the model keeps improving on training data but gets worse on unseen data. Sigmoid's validation loss, by contrast, stays flat and stable for the full 100 epochs.

**Why this likely happened:** the dataset only has 442 samples total, and only ~283 of those go to actual training after the test and validation splits. With just 2 hidden layers of 64 and 32 neurons, ReLU and Tanh have enough capacity to start memorizing that small training set, while Sigmoid's saturating behavior (it compresses large values toward 0 or 1) acts as a mild, built-in brake on how sharply it can fit the training data, which incidentally helped it generalize better here, even though that's not usually the reason to pick Sigmoid.

**This directly affected which activation "won":** Sigmoid had the best R² not necessarily because it's a fundamentally better activation function in general, but because ReLU and Tanh's extra flexibility became a liability on this specific small dataset. On a larger dataset, this result could easily flip.

**Finding: the two loss-curve plots aren't directly comparable to each other.** MSE, MAE, and Huber compute error differently, so their raw loss values sit on completely different numeric scales (visible in `loss_curves_loss_function.png`: Huber's curve sits far below MSE and MAE's, not because it's "better" but because Huber's formula produces smaller numbers for the same actual prediction error). This is why the final judgment of "which model is best" in the comparison table uses MAE/RMSE/R² computed after inverse-scaling back to real units. Those are the only numbers that are fairly comparable across differently-trained models. Comparing raw training loss values across different loss functions would have been a misleading way to pick a winner.

## Analysis (for the required write-up)

- **Best activation function:** Sigmoid, based on R², but see the overfitting finding above; this is dataset-size-dependent, not a general rule.
- **Best loss function:** MSE and Huber performed almost identically (R² 0.468 vs 0.460); MAE was slightly behind. This suggests the dataset doesn't have severe outliers that would make MAE/Huber's robustness clearly pay off. MSE's simpler behavior was enough.
- **Best combination:** Sigmoid + MSE, R² = 0.468, MAE = 41.76, RMSE = 53.09.
- **Did activation function affect convergence?** Yes clearly, see the overfitting finding above; ReLU/Tanh converged faster initially but diverged later, while Sigmoid converged slower but stayed stable.
- **Did any loss function show more robustness to outliers?** Not meaningfully in this case. Huber's slight edge over MAE and near-tie with MSE suggests this dataset doesn't have outliers severe enough to make that robustness clearly show up in the results.
- **Overfitting/underfitting from the curves:** ReLU and Tanh show overfitting after ~epoch 20 (see above). Sigmoid shows a well-fit curve, loss drops and plateaus without climbing back up, suggesting neither significant overfitting nor underfitting for that specific model.

## Important Note on the Deliverable Format

The assignment specifies submitting a **Google Colab notebook** with dataset description, preprocessing, architecture, experiments, plots, evaluation, comparison table, and interpretation all included together. This repo follows the `.py` + `README.md` structure used for the other practicals, and the README above covers everything the notebook would need (description, justification, results, plots, analysis), but if your professor specifically wants the `.ipynb` file itself, you'll need to either paste this code into Google Colab directly (it will run as-is) or ask me to convert it into a notebook file.

## How to Run

```bash
python mlp_regression.py
```

```bash
pip install tensorflow scikit-learn pandas matplotlib
```
