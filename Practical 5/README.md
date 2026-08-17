# Practical 5: CNN for Binary Image Classification (Cats vs Dogs)

A Convolutional Neural Network (CNN) built with TensorFlow/Keras, trained to classify images as either a cat or a dog.

## Files

- `cnn_cats_dogs.py`

## Dataset

The official small "Cats vs Dogs" filtered dataset from the TensorFlow tutorials: 2,000 training images, 1,000 validation images, split into `cats/` and `dogs/` folders. Downloads and extracts automatically on first run, no manual download needed.

## How It Works

1. **Download & extract:** `get_file(..., extract=True)` downloads and unzips the dataset.
2. **Load images:** `image_dataset_from_directory()` reads images from folders and labels them automatically based on folder name, resizing all images to 150x150.
3. **Data augmentation:** `RandomFlip`, `RandomRotation`, `RandomZoom` randomly alter training images each epoch to increase effective training variety.
4. **Build the CNN**:
   - `Rescaling(1./255)`: scales pixel values from 0-255 to 0-1.
   - Three `Conv2D` + `MaxPooling2D` blocks (32→64→128 filters): detect increasingly complex visual patterns.
   - `Flatten`: converts 2D feature maps into a 1D vector.
   - `Dense(128, relu)`: combines detected features.
   - `Dropout(0.5)`: randomly disables neurons during training to reduce overfitting.
   - `Dense(1, sigmoid)`: final binary output (cat vs dog).
5. **Compile:** Adam optimizer, binary cross-entropy loss.
6. **Train:** 10 epochs, validated after each epoch.
7. **Evaluate & predict:** reports validation accuracy and prints sample predictions with confidence scores.

## What Went Wrong and What I Learned

**Bug: scalar conversion errors in the prediction loop:**
The first version used:
```python
predicted_class = class_names[int(predictions[i] >= 0.5)]
actual_class = class_names[int(labels[i])]
```
Both threw `TypeError: only 0-dimensional arrays can be converted to Python scalars`, since `predictions[i]` and `labels[i]` are NumPy/TensorFlow array objects, not plain Python numbers. Fixed by calling `.item()` on each before converting, which correctly pulls out a single scalar value.

**Deprecation warning:**
Passing `input_shape` directly into the first `Rescaling` layer triggered a Keras warning that this pattern is discouraged. Fixed by adding an explicit `tf.keras.layers.Input(shape=(150, 150, 3))` as the first layer instead.

**Dataset access limitation:** this code was written and structurally tested (data pipeline, augmentation, CNN layers, training loop, prediction output) using a small synthetic image set, since the sandbox environment used to write this couldn't reach Google's dataset servers to download the real Cats vs Dogs images. Run on the real dataset with internet access, and expect actual training. A small CNN like this one typically reaches roughly 75-85% validation accuracy on this dataset after 10-15 epochs, noticeably higher than the ~50% seen when testing structure only on random synthetic images.

## How to Run

```bash
python cnn_cats_dogs.py
```

```bash
pip install tensorflow
```
