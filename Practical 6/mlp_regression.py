import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(42)
np.random.seed(42)

data = load_diabetes(as_frame=True)
df = data.frame
print(df.describe())
print("\nMissing values per column:\n", df.isnull().sum())

X = data.data.values
y = data.target.values

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)


def build_model(activation, input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(32, activation=activation),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    return model


def run_experiment(name, activation, loss, X_train, y_train, X_test, y_test):
    model = build_model(activation, X_train.shape[1])
    model.compile(optimizer='adam', loss=loss, metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "name": name,
        "activation": activation,
        "loss": loss,
        "history": history,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


results = []

for activation in ['relu', 'sigmoid', 'tanh']:
    result = run_experiment(
        f"Activation-{activation}", activation, 'mse',
        X_train, y_train, X_test, y_test
    )
    results.append(result)

best_activation_result = max(results, key=lambda r: r["r2"])
best_activation = best_activation_result["activation"]
print(f"\nBest activation based on R^2: {best_activation}")

for loss in ['mae', 'huber']:
    result = run_experiment(
        f"Loss-{loss}", best_activation, loss,
        X_train, y_train, X_test, y_test
    )
    results.append(result)

table = pd.DataFrame([{
    "Experiment": r["name"],
    "Activation": r["activation"],
    "Loss Function": r["loss"],
    "MAE": round(r["mae"], 2),
    "RMSE": round(r["rmse"], 2),
    "R2": round(r["r2"], 4)
} for r in results])

print("\nComparison Table:")
print(table.to_string(index=False))

plt.figure(figsize=(10, 5))
for r in results[:3]:
    plt.plot(r["history"].history['val_loss'], label=r["activation"])
plt.title("Validation Loss by Activation Function (loss=MSE)")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss (MSE, scaled target)")
plt.legend()
plt.savefig("loss_curves_activation.png")
plt.close()

plt.figure(figsize=(10, 5))
for r in [best_activation_result, results[3], results[4]]:
    plt.plot(r["history"].history['val_loss'], label=r["loss"])
plt.title(f"Validation Loss by Loss Function (activation={best_activation})")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss (scaled target)")
plt.legend()
plt.savefig("loss_curves_loss_function.png")
plt.close()

best_overall = max(results, key=lambda r: r["r2"])
print(f"\nBest overall model: activation={best_overall['activation']}, "
      f"loss={best_overall['loss']}, R2={round(best_overall['r2'], 4)}, "
      f"MAE={round(best_overall['mae'], 2)}, RMSE={round(best_overall['rmse'], 2)}")
