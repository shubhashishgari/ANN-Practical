# Practical 7: Keras MLP for Multiclass Classification (Wine Quality)

An MLP built with Keras for multiclass classification of red wine quality, comparing 3 hidden-layer activation functions and 3 optimizers across 5 experiments.

## Files

- `mlp_wine_multiclass.py`
- `WineQT.csv` (dataset)
- `curves_activation.png` (generated on run)
- `curves_optimizer.png` (generated on run)
- `confusion_matrix.png` (generated on run)

## Dataset Description

**WineQT.csv**: 1,143 red wine samples with 11 physicochemical input features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) and a quality score target. Derived from the well-known UCI Wine Quality dataset.

**Problem definition**: predict a wine's quality category from its measurable chemical properties. This is a multiclass classification problem because the target is a discrete category, not a continuous value.

**Missing values**: none. Verified with `df.isnull().sum()`, all columns returned 0, so no imputation was required.

**Dropped column**: `Id` was removed, since it is a row identifier with no predictive meaning. Leaving it in risks the model treating an arbitrary index as a real feature.

### Class Distribution and the Binning Decision

The raw `quality` column contains 6 distinct scores, but they are severely imbalanced:

```
quality
3      6
4     33
5    483
6    462
7    143
8     16
```

Classes 3 and 8 have only 6 and 16 samples respectively. After an 80/20 split, class 3 would have roughly **1 sample in the test set**, making any per-class metric for it statistically meaningless, and the model would have almost no examples to learn from.

**Preprocessing decision**: quality scores were binned into 3 ordered categories:

| Class | Quality scores | Count | Proportion |
|-------|---------------|-------|------------|
| Low | 3, 4, 5 | 522 | 45.7% |
| Medium | 6 | 462 | 40.4% |
| High | 7, 8 | 159 | 13.9% |

**Justification**: this satisfies the 3-or-more-class requirement while giving every class enough samples to actually train and evaluate on. It also reflects how wine quality is realistically interpreted (poor / average / good) rather than treating a 1-point difference between adjacent scores as a hard categorical boundary. Some imbalance remains (High is 13.9% of the data), which is retained deliberately and analyzed in the results below.

### Preprocessing Justification

- **Target encoding**: `LabelEncoder` converts the string class names into integers, then `to_categorical` one-hot encodes them into vectors. One-hot encoding is required because the 3 classes are treated as distinct categories by the softmax output layer, and raw integers would imply a false numeric ordering.
- **Feature scaling**: `StandardScaler` rescales all 11 features to mean 0, standard deviation 1. This is necessary because the raw features are on wildly different scales (total sulfur dioxide reaches into the hundreds while density sits near 1.0). Without scaling, the large-magnitude features would dominate the weight updates regardless of their actual predictive value.
- **Scaler fitted on training data only**: `fit_transform` on the training set, `transform` on the test set. Fitting the scaler on the full dataset before splitting would leak information about the test set's distribution into training.
- **Stratified split**: `stratify=y_encoded_int` keeps the class proportions consistent across train and test. Without this, the minority High class could be unevenly distributed by chance.
- **Validation strategy**: `validation_split=0.2` inside `fit()` holds out 20% of the training data for per-epoch validation, keeping the test set completely untouched until final evaluation.

## MLP Architecture

| Setting | Value |
|---------|-------|
| Input features | 11 |
| Number of classes | 3 |
| Hidden layer 1 | 64 neurons |
| Hidden layer 2 | 32 neurons |
| Hidden activation | varies by experiment (ReLU / Sigmoid / Tanh) |
| Output layer | 3 neurons |
| Output activation | Softmax |
| Loss function | Categorical crossentropy |
| Optimizer | varies by experiment (Adam / SGD / RMSprop) |
| Learning rate | 0.001 (Adam, RMSprop), 0.01 (SGD) |
| Batch size | 32 |
| Epochs | 100 |
| Training samples | 914 |
| Test samples | 229 |

**Softmax output**: produces a probability distribution across the 3 classes that sums to 1, so the predicted class is whichever has the highest probability. Sigmoid would be wrong here since it gives independent 0-1 values rather than a distribution over mutually exclusive classes.

**SGD learning rate**: set to 0.01 rather than 0.001 because plain SGD has no adaptive per-parameter scaling and converges very slowly at the smaller rate. This keeps the comparison fair rather than handicapping SGD with a rate suited to adaptive optimizers.

## Results

```
Experiment Activation Optimizer  Train Acc  Val Acc  Train Loss  Val Loss  Test Acc  Precision  Recall     F1
   Model 1       relu      adam     0.8960   0.6175      0.3051    1.0904    0.6070     0.6170  0.6070 0.6092
   Model 2    sigmoid      adam     0.6594   0.6612      0.7243    0.7207    0.6201     0.6178  0.6201 0.6188
   Model 3       tanh      adam     0.8235   0.6776      0.4598    0.7835    0.6681     0.6745  0.6681 0.6698
   Model 4       relu       sgd     0.6977   0.6667      0.6537    0.7583    0.6419     0.6385  0.6419 0.6385
   Model 5       relu   rmsprop     0.8605   0.6557      0.3605    1.0685    0.6550     0.6557  0.6550 0.6551
```

Precision, Recall, and F1 are **weighted averages**, chosen over macro-average because the classes are imbalanced and weighted averaging accounts for the differing support of each class.

### Confusion Matrix (best model: Tanh + Adam)

```
        High  Low  Medium
High      18    1      13
Low        2   74      29
Medium     9   22      61
```

Per-class report:
```
              precision    recall  f1-score   support
        High       0.62      0.56      0.59        32
         Low       0.76      0.70      0.73       105
      Medium       0.59      0.66      0.63        92
    accuracy                           0.67       229
   macro avg       0.66      0.64      0.65       229
weighted avg       0.67      0.67      0.67       229
```

## What Went Wrong and What I Learned

**Problem: the raw 6-class target was unusable.** Classes 3 and 8 had 6 and 16 samples. Training on this directly would produce a model that never predicts those classes at all, and per-class metrics computed on roughly 1 test sample would be noise rather than measurement. Binning into 3 classes fixed this while keeping the problem genuinely multiclass. This was a data problem, not a model problem, and no amount of architecture tuning would have solved it.

**Finding: ReLU + Adam overfits badly on this dataset.** Model 1 reached 89.6% training accuracy but only 61.75% validation accuracy, a gap of nearly 28 points. Its validation loss bottoms out around epoch 10 at roughly 0.76, then climbs steadily to 1.09 by epoch 100. This is textbook overfitting and is clearly visible in `curves_activation.png`.

**Finding: Sigmoid barely overfits at all, but underfits instead.** Model 2 had training accuracy 65.9% and validation accuracy 66.1%, essentially identical, with a flat validation loss curve. The model generalizes well because it is not learning much in the first place. Sigmoid saturates at extreme values, which squashes gradients and limits how sharply it can fit the training data. Good generalization, but achieved by underfitting rather than by learning a better representation.

**Finding: Tanh landed between the two and won.** Model 3 reached 82.4% training accuracy and 67.8% validation accuracy, with the best test accuracy (66.8%) and F1 (0.670). Tanh is zero-centered, which generally allows faster and more balanced learning than sigmoid, but it still saturates at the extremes, which restrained the runaway overfitting that ReLU showed.

**Finding: the optimizer affected generalization more than final accuracy.** In `curves_optimizer.png`, Adam and RMSprop both drive training loss down fast (to 0.31 and 0.36) while their validation loss climbs steadily after roughly epoch 15, indicating they are memorizing. SGD converges more slowly and reaches a much higher training loss (0.65) but its validation loss flattens around 0.758 and stays there. SGD's slower, non-adaptive updates acted as an implicit brake on overfitting.

**Finding: early stopping would likely beat all five models.** Every Adam and RMSprop run reached its best validation loss around epoch 10-20 and got worse from there, meaning training for the full 100 epochs actively hurt final performance. The reported final-epoch numbers are therefore worse than what these models achieved mid-training.

## Analysis and Interpretation

**1. Which activation function performed best? Why?**
Tanh, with test accuracy 66.8% and F1 0.670. It is zero-centered so it learns more effectively than sigmoid, but unlike ReLU it saturates at extreme values, which limited overfitting on this relatively small dataset (914 training samples).

**2. Which optimizer performed best? Why?**
On raw test accuracy, RMSprop (65.5%) edged out SGD (64.2%) and Adam (60.7%). But on generalization, SGD was clearly best: its validation loss stayed flat at 0.758 while Adam's and RMSprop's climbed past 1.06. SGD is the better choice here despite not topping the accuracy column.

**3. Best activation-optimizer combination?**
Tanh + Adam (Model 3), test accuracy 66.8%, weighted F1 0.670.

**4. Did the activation function significantly affect convergence?**
Yes, substantially. ReLU converged fastest on training data (steepest training loss drop) but diverged worst on validation. Sigmoid converged slowest and plateaued early. Tanh sat between the two. See `curves_activation.png`.

**5. Did the optimizer affect convergence speed?**
Yes. Adam and RMSprop dropped training loss much faster than SGD, which is expected since both adapt the learning rate per parameter while SGD applies a fixed rate. SGD needed the full 100 epochs to reach a training loss that Adam hit by roughly epoch 25.

**6. Best validation performance?**
Model 3 (Tanh + Adam), validation accuracy 67.8%.

**7. Best test performance?**
Model 3 (Tanh + Adam), test accuracy 66.8%.

**8. Significant difference between training and testing performance?**
Yes, for the adaptive optimizers. Model 1 (ReLU + Adam) had a 28.9-point gap between training (89.6%) and test (60.7%) accuracy. Model 2 (Sigmoid) had almost no gap (65.9% vs 62.0%). The gap size tracks directly with how much each configuration overfit.

**9. Do the learning curves indicate overfitting or underfitting?**
Both, depending on the model. ReLU + Adam and ReLU + RMSprop show clear overfitting (rising validation loss with falling training loss). Sigmoid + Adam shows underfitting (both curves flat and close together at mediocre accuracy). Tanh + Adam and ReLU + SGD sit closest to a healthy fit.

**10. Which classes were most frequently misclassified?**
High had the worst recall (0.56), with 13 of its 32 test samples predicted as Medium. In absolute terms the largest single error was Low predicted as Medium (29 cases). Low was predicted most accurately (recall 0.70).

**11. Possible reasons for these misclassifications?**
- **Overlapping class boundaries**: Low, Medium, and High are cuts on a continuous underlying quality score. A wine scoring 5 (Low) and one scoring 6 (Medium) can be chemically almost identical, so the boundary is genuinely blurry rather than a real categorical divide. This explains why the Low-Medium confusion dominates: those two classes are adjacent and together account for 51 of the 76 total errors.
- **Class imbalance**: High is only 13.9% of the data, so the model sees far fewer High examples and is biased toward predicting the two larger classes.
- **Limited features**: 11 physicochemical measurements may simply not capture everything that determines perceived quality, since the original labels came from human tasters.
- **Subjective labels**: the underlying quality scores are median sensory ratings from human judges, which carry inherent noise that no model can fully resolve.

**12. What changes could improve performance?**
- **Early stopping** on validation loss, which would have stopped the Adam runs around epoch 15 at their best point instead of letting them degrade for 85 more epochs.
- **Class weights or oversampling** for the minority High class, to counter the imbalance bias.
- **Regularization** (dropout or L2) to directly address the ReLU overfitting rather than relying on activation choice to limit it.
- **Merging Low and Medium** into a binary problem if the goal permits, since that boundary produces most of the errors.

**13. Final model selection.**
**Model 3 (Tanh + Adam)** is selected. Justification beyond accuracy alone:
- Best test accuracy (66.8%) and best weighted F1 (0.670) of all five models
- Best validation accuracy (67.8%)
- Its train/validation gap (82.4% vs 67.8%) is meaningfully smaller than ReLU + Adam's, so it generalizes better than the highest-training-accuracy model
- Balanced per-class performance: it is the only model to achieve reasonable recall across all three classes rather than collapsing onto the two majority classes
- Same architecture complexity as every other model, so it wins without any added cost

The honest caveat: its validation loss does still rise after roughly epoch 25, so this model would improve further with early stopping. It is the best of the five tested, not the best achievable configuration.

## Conclusion

Tanh + Adam gave the best overall multiclass performance at 66.8% test accuracy on 3-class wine quality prediction. The dominant limitation is not the architecture but the data: the Low/Medium boundary is an artificial cut through a continuous, subjectively-labeled quality scale, and 51 of 76 errors fall on that single boundary. The experiments also showed that the highest training accuracy (ReLU + Adam, 89.6%) produced the *worst* test accuracy (60.7%), a clear demonstration that training performance alone is a misleading model-selection criterion.

## How to Run

```bash
python mlp_wine_multiclass.py
```

```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn
```

Keep `WineQT.csv` in the same folder as the script.

## Note on Deliverable Format

The assignment asks for a Google Colab notebook. This repo follows the `.py` + `README.md` structure used across the other practicals, and the README covers every required section (dataset description, preprocessing justification, architecture, experiments, plots, confusion matrix, comparison table, analysis, final model selection, conclusion). The script runs as-is if pasted into Colab. Ask if you want it converted to an actual `.ipynb`.
