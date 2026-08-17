import tensorflow as tf
import os

tf.random.set_seed(42)

dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip", origin=dataset_url, extract=True)

base_dir = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

img_size = (150, 150)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size, label_mode="binary"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir, image_size=img_size, batch_size=batch_size, label_mode="binary"
)

class_names = train_ds.class_names

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Rescaling(1./255),
    data_augmentation,

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"\nValidation Accuracy: {round(accuracy * 100, 2)}%")

print("\nSample Predictions:")
for images, labels in val_ds.take(1):
    predictions = model.predict(images, verbose=0)
    for i in range(5):
        predicted_class = class_names[int(predictions[i].item() >= 0.5)]
        actual_class = class_names[int(labels[i].numpy().item())]
        print(f"Predicted: {predicted_class} (confidence: {round(predictions[i].item(), 3)}) "
              f"=> Actual: {actual_class}")
