"""
mnist_fgsm_expanded.py

MNIST CNN trained with standard and adversarial (FGSM) training.
Evaluates robustness across multiple epsilon values and compares
model performance before and after adversarial fine-tuning.

Usage:
    python mnist_fgsm_expanded.py
"""

import os
import json
import random
import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
VAL_SPLIT = 0.1
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

MODEL_PATH = "cnn_mnist_fgsm.h5"
ADV_MODEL_PATH = "cnn_mnist_fgsm_adv_trained.h5"
RESULTS_PATH = "experiment_results.json"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist():
    """
    Load and preprocess MNIST.

    Returns:
        x_train, y_train_cat, x_test, y_test_cat  — normalised arrays
        y_train_int, y_test_int                    — integer labels
    """
    (x_train, y_train_int), (x_test, y_test_int) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train_cat = to_categorical(y_train_int, 10)
    y_test_cat = to_categorical(y_test_int, 10)

    print(f"Train: {x_train.shape}  Test: {x_test.shape}")
    return x_train, y_train_cat, x_test, y_test_cat, y_train_int, y_test_int


def split_validation(x_train, y_train_cat, val_split: float = VAL_SPLIT):
    """Split the first `val_split` fraction of training data into a validation set."""
    n_val = int(x_train.shape[0] * val_split)
    return (
        x_train[n_val:], y_train_cat[n_val:],   # training subset
        x_train[:n_val], y_train_cat[:n_val],    # validation subset
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_cnn(input_shape=(28, 28, 1), num_classes: int = 10) -> tf.keras.Model:
    """
    Build a simple CNN for MNIST classification.

    Architecture: Conv(32) -> Pool -> Conv(64) -> Pool -> Conv(128)
                  -> Flatten -> Dense(128) -> Softmax(10)
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape, name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Conv2D(128, (3, 3), activation="relu", name="conv3"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="mnist_cnn")

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: tf.keras.Model,
    x_train, y_train,
    x_val=None, y_val=None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tf.keras.callbacks.History:
    """
    Compile and train `model`. Uses EarlyStopping and ModelCheckpoint
    when validation data is provided.
    """
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fit_kwargs = dict(epochs=epochs, batch_size=batch_size, verbose=2)

    if x_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (x_val, y_val)
        fit_kwargs["callbacks"] = [
            callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss"),
        ]

    return model.fit(x_train, y_train, **fit_kwargs)


def adversarial_finetune(
    model: tf.keras.Model,
    x_train, y_train_cat,
    epsilon: float = 0.1,
    batch_size: int = 256,
    epochs: int = 2,
) -> tf.keras.callbacks.History:
    """
    Fine-tune `model` on a 50/50 mix of clean and FGSM-perturbed training images.

    This is a simple but effective adversarial training strategy.
    """
    print(f"Generating adversarial training set (ε={epsilon}) ...")
    adv_x = generate_adversarial_dataset(model, x_train, y_train_cat, epsilon=epsilon, batch_size=batch_size)

    combined_x = np.vstack([x_train, adv_x])
    combined_y = np.vstack([y_train_cat, y_train_cat])

    # Shuffle
    perm = np.random.permutation(len(combined_x))
    combined_x, combined_y = combined_x[perm], combined_y[perm]

    print(f"Combined dataset: {combined_x.shape}")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model.fit(combined_x, combined_y, batch_size=batch_size, epochs=epochs, verbose=2)


# ---------------------------------------------------------------------------
# FGSM attack
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 0.1,
) -> tf.Tensor:
    """
    Fast Gradient Sign Method (FGSM).

    Perturbs `images` by `epsilon` in the direction that maximises cross-entropy loss.

    Args:
        model:   Keras model.
        images:  Float array in [0, 1], shape (N, H, W, C).
        labels:  One-hot labels, shape (N, num_classes).
        epsilon: Perturbation magnitude.

    Returns:
        Adversarial images clipped to [0, 1].
    """
    images_var = tf.Variable(tf.cast(images, tf.float32))
    labels_tf = tf.cast(labels, tf.float32)

    with tf.GradientTape() as tape:
        preds = model(images_var, training=False)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(labels_tf, preds)
        )

    grads = tape.gradient(loss, images_var)
    adv = images_var + epsilon * tf.sign(grads)
    return tf.clip_by_value(adv, 0.0, 1.0)


def generate_adversarial_dataset(
    model: tf.keras.Model,
    x: np.ndarray,
    y_cat: np.ndarray,
    epsilon: float = 0.1,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Generate adversarial examples for an entire dataset in mini-batches.

    Returns a numpy array with the same shape as `x`.
    """
    adv_batches = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        adv_batch = fgsm_attack(model, x[start:end], y_cat[start:end], epsilon)
        adv_batches.append(adv_batch.numpy())
    return np.vstack(adv_batches)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: tf.keras.Model,
    x: np.ndarray,
    y_cat: np.ndarray,
    y_int: np.ndarray = None,
    batch_size: int = 256,
    label: str = "",
):
    """
    Predict and print accuracy + classification report.

    Returns:
        accuracy (float), predicted classes (np.ndarray), true classes (np.ndarray)
    """
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    preds = np.argmax(probs, axis=1)
    true = y_int if y_int is not None else np.argmax(y_cat, axis=1)

    acc = accuracy_score(true, preds)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Accuracy: {acc:.4f}")
    print(classification_report(true, preds, digits=4))
    return acc, preds, true


def sweep_epsilons(
    model: tf.keras.Model,
    x_test, y_test_cat, y_test_int,
    epsilons=None,
) -> dict:
    """
    Evaluate model under each epsilon in `epsilons`.

    Returns:
        dict mapping epsilon -> accuracy
    """
    if epsilons is None:
        epsilons = EPSILONS

    results = {}
    for eps in epsilons:
        print(f"\n--- ε = {eps} ---")
        if eps == 0.0:
            acc, _, _ = evaluate(model, x_test, y_test_cat, y_test_int, label=f"ε={eps}")
        else:
            adv = generate_adversarial_dataset(model, x_test, y_test_cat, epsilon=eps, batch_size=512)
            acc, _, _ = evaluate(model, adv, y_test_cat, y_test_int, label=f"ε={eps}")
        results[eps] = acc
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(history, title: str = "") -> None:
    hist = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    axes[0].plot(hist.get("loss", []), label="train")
    if "val_loss" in hist:
        axes[0].plot(hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist.get("accuracy", []), label="train")
    if "val_accuracy" in hist:
        axes[1].plot(hist["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm, classes, normalize: bool = False, title: str = "Confusion matrix"
) -> None:
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), ha="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_epsilon(results_before: dict, results_after: dict = None) -> None:
    eps = sorted(results_before.keys())
    plt.figure(figsize=(8, 5))
    plt.plot(eps, [results_before[e] for e in eps], marker="o", label="Before adv training")
    if results_after:
        plt.plot(eps, [results_after[e] for e in eps], marker="x", label="After adv training")
    plt.title("Accuracy vs FGSM Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_adversarial_examples(
    originals, adversarials, labels_true, preds_adv=None, n: int = 6
) -> None:
    """
    Show n rows of: [original | adversarial | perturbation × 10].
    """
    n = min(n, len(originals))
    idxs = np.random.choice(len(originals), size=n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    col_titles = ["Original", "Adversarial", "Perturbation ×10"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)

    for row, idx in enumerate(idxs):
        orig = originals[idx].squeeze()
        adv = adversarials[idx].squeeze()
        pert = adv - orig

        axes[row, 0].imshow(orig, cmap="gray")
        axes[row, 0].set_ylabel(f"label={labels_true[idx]}", fontsize=8)

        adv_title = f"pred={preds_adv[idx]}" if preds_adv is not None else ""
        axes[row, 1].imshow(adv, cmap="gray")
        axes[row, 1].set_ylabel(adv_title, fontsize=8)

        axes[row, 2].imshow(pert * 10 + 0.5, cmap="gray", vmin=0, vmax=1)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds()

    # ------ Data ------
    x_train, y_train_cat, x_test, y_test_cat, y_train_int, y_test_int = load_mnist()
    x_tr, y_tr, x_val, y_val = split_validation(x_train, y_train_cat)
    print(f"Training: {x_tr.shape[0]}  Validation: {x_val.shape[0]}")

    # ------ Build & train clean model ------
    model = build_cnn(input_shape=x_train.shape[1:])
    model.summary()

    print("\n=== Clean training ===")
    history_clean = train(model, x_tr, y_tr, x_val, y_val, epochs=EPOCHS)
    plot_training_history(history_clean, title="Clean training")

    # ------ Baseline evaluation ------
    print("\n=== Baseline evaluation (clean test set) ===")
    evaluate(model, x_test, y_test_cat, y_test_int, label="Clean")

    # ------ Epsilon sweep (before adversarial training) ------
    print("\n=== Epsilon sweep — before adversarial training ===")
    eps_before = sweep_epsilons(model, x_test, y_test_cat, y_test_int)
    plot_accuracy_vs_epsilon(eps_before)

    # ------ Visualise adversarial examples (ε=0.1) ------
    adv_vis = generate_adversarial_dataset(model, x_test[:500], y_test_cat[:500], epsilon=0.1)
    _, preds_adv_vis, _ = evaluate(model, adv_vis, y_test_cat[:500], y_test_int[:500], label="Adv ε=0.1")
    visualize_adversarial_examples(x_test[:500], adv_vis, y_test_int[:500], preds_adv=preds_adv_vis)

    # ------ Confusion matrices (before) ------
    print("\n=== Confusion matrices — before adversarial training ===")
    _, preds_clean, true = evaluate(model, x_test, y_test_cat, y_test_int, label="Clean")
    adv_full = generate_adversarial_dataset(model, x_test, y_test_cat, epsilon=0.1, batch_size=512)
    _, preds_adv_full, _ = evaluate(model, adv_full, y_test_cat, y_test_int, label="Adv ε=0.1")

    plot_confusion_matrix(confusion_matrix(true, preds_clean), list(range(10)),
                          title="Clean — before adv training")
    plot_confusion_matrix(confusion_matrix(true, preds_adv_full), list(range(10)),
                          title="Adversarial ε=0.1 — before adv training")

    # ------ Save clean model ------
    model.save(MODEL_PATH)
    print(f"Clean model saved → {MODEL_PATH}")

    # ------ Adversarial fine-tuning ------
    print("\n=== Adversarial fine-tuning ===")
    model_adv = tf.keras.models.load_model(MODEL_PATH)
    history_adv = adversarial_finetune(model_adv, x_tr, y_tr, epsilon=0.1, batch_size=512, epochs=2)
    plot_training_history(history_adv, title="Adversarial fine-tuning")

    # ------ Evaluation after adversarial training ------
    print("\n=== Evaluation — after adversarial training ===")
    evaluate(model_adv, x_test, y_test_cat, y_test_int, label="Clean (after)")
    adv_after = generate_adversarial_dataset(model_adv, x_test, y_test_cat, epsilon=0.1, batch_size=512)
    _, preds_adv_after, true_after = evaluate(model_adv, adv_after, y_test_cat, y_test_int, label="Adv ε=0.1 (after)")

    plot_confusion_matrix(confusion_matrix(true_after, preds_adv_after), list(range(10)),
                          title="Adversarial ε=0.1 — after adv training")

    # ------ Epsilon sweep (after adversarial training) ------
    print("\n=== Epsilon sweep — after adversarial training ===")
    eps_after = sweep_epsilons(model_adv, x_test, y_test_cat, y_test_int)
    plot_accuracy_vs_epsilon(eps_before, eps_after)

    # ------ Visualise adversarial examples from fine-tuned model ------
    adv_vis_after = generate_adversarial_dataset(model_adv, x_test[:300], y_test_cat[:300], epsilon=0.1)
    _, preds_vis_after, _ = evaluate(model_adv, adv_vis_after, y_test_cat[:300], y_test_int[:300], label="Adv ε=0.1 (after)")
    visualize_adversarial_examples(x_test[:300], adv_vis_after, y_test_int[:300], preds_adv=preds_vis_after)

    # ------ Save adversarially trained model ------
    model_adv.save(ADV_MODEL_PATH)
    print(f"Adversarially trained model saved → {ADV_MODEL_PATH}")

    # ------ Persist experiment results ------
    results = {
        "date": str(datetime.datetime.now()),
        "seed": SEED,
        "epochs_clean": EPOCHS,
        "epsilons_tested": EPSILONS,
        "accuracy_before": {str(k): float(v) for k, v in eps_before.items()},
        "accuracy_after":  {str(k): float(v) for k, v in eps_after.items()},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {RESULTS_PATH}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
