import numpy as np, os, warnings, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings("ignore"); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf; tf.get_logger().setLevel('ERROR')
from tensorflow import keras; from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import pandas as pd

os.makedirs("outputs", exist_ok=True)
np.random.seed(42); tf.random.set_seed(42)

# ── Load sklearn's built-in digits (1797 samples, 8×8, upscale to 28×28) ──
print("Loading sklearn digits dataset and upscaling to 28×28...")
digits = load_digits()
X_raw = digits.images  # (1797, 8, 8)
y_raw = digits.target.astype(np.int32)

# Upscale 8×8 → 28×28
X_up = np.array([zoom(img, 28/8) for img in X_raw])
X_up = X_up / X_up.max()  # normalise to [0,1]
X_up = X_up[..., np.newaxis].astype(np.float32)  # (1797, 28, 28, 1)

print(f"Dataset: {X_up.shape}, classes: {np.unique(y_raw)}")

# Augment by adding noise variants to get more samples
augmented_X, augmented_y = [X_up], [y_raw]
for sigma in [0.05, 0.08, 0.03]:
    rng = np.random.RandomState(int(sigma*1000))
    noisy = np.clip(X_up + rng.randn(*X_up.shape)*sigma, 0, 1).astype(np.float32)
    augmented_X.append(noisy); augmented_y.append(y_raw)
X_aug = np.concatenate(augmented_X, axis=0)
y_aug = np.concatenate(augmented_y, axis=0)
print(f"After augmentation: {X_aug.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.15, stratify=y_aug, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, stratify=y_train, random_state=42)
print(f"Train:{X_train.shape} Val:{X_val.shape} Test:{X_test.shape}")

# ── EDA plots ──
labels = [str(i) for i in range(10)]
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
fig.suptitle("Handwritten Digit Samples (sklearn digits, upscaled to 28×28)", fontsize=13, fontweight="bold")
for d in range(10):
    idxs = np.where(y_train==d)[0][:2]
    for row, idx in enumerate(idxs[:2]):
        ax=axes[row,d]; ax.imshow(X_train[idx].squeeze(), cmap="gray"); ax.axis("off")
        if row==0: ax.set_title(str(d), fontsize=10)
plt.tight_layout(); plt.savefig("outputs/sample_images.png", dpi=150); plt.close()

unique, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(10,4))
plt.bar([str(u) for u in unique], counts, color="#3498db", edgecolor="white")
plt.title("Class Distribution (Training Set)", fontsize=12, fontweight="bold"); plt.xlabel("Digit"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("outputs/class_distribution.png", dpi=150); plt.close()
print("EDA plots saved")

# ── Models ──
def build_simple_cnn():
    inp = keras.Input((28,28,1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation="softmax")(x)
    m = keras.Model(inp, out, name="SimpleCNN")
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def build_deep_cnn():
    inp = keras.Input((28,28,1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation="softmax")(x)
    m = keras.Model(inp, out, name="DeepCNN")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

cb = [EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
      ReduceLROnPlateau(factor=0.5, patience=4, verbose=0)]

print("Training Simple CNN...")
m1 = build_simple_cnn()
h1 = m1.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val,y_val), callbacks=cb, verbose=0)
v1 = max(h1.history['val_accuracy'])
print(f"  Best val_acc: {v1:.4f}")

print("Training Deep CNN...")
m2 = build_deep_cnn()
h2 = m2.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val,y_val), callbacks=cb, verbose=0)
v2 = max(h2.history['val_accuracy'])
print(f"  Best val_acc: {v2:.4f}")

# ── Evaluate ──
results = {}; best_m, best_a, best_p = None, 0, None
for name, m in [("Simple CNN", m1), ("Deep CNN", m2)]:
    yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, yp); f1w = f1_score(y_test, yp, average="weighted")
    results[name] = {"Test Accuracy": acc, "Test F1 (weighted)": f1w}
    print(f"\n{name}: Acc={acc:.4f}  F1={f1w:.4f}")
    print(classification_report(y_test, yp, target_names=labels))
    if acc > best_a: best_a, best_m, best_p = acc, m, yp

# Confusion matrix
cm = confusion_matrix(y_test, best_p)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Best CNN — Confusion Matrix", fontsize=13, fontweight="bold")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix_best.png", dpi=150); plt.close()

# Training history
fig, ax = plt.subplots(1,2, figsize=(13,4))
fig.suptitle("CNN Training History", fontsize=13, fontweight="bold")
ax[0].plot(h1.history["accuracy"], "b-", label="Simple Train")
ax[0].plot(h1.history["val_accuracy"], "b--", label="Simple Val")
ax[0].plot(h2.history["accuracy"], "r-", label="Deep Train")
ax[0].plot(h2.history["val_accuracy"], "r--", label="Deep Val")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend(fontsize=8)
ax[1].plot(h1.history["loss"], "b-", label="Simple Train")
ax[1].plot(h1.history["val_loss"], "b--", label="Simple Val")
ax[1].plot(h2.history["loss"], "r-", label="Deep Train")
ax[1].plot(h2.history["val_loss"], "r--", label="Deep Val")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("outputs/training_history.png", dpi=150); plt.close()

# Per-class accuracy
per_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10,4))
plt.bar(labels, per_acc*100, color="#2ecc71", edgecolor="white")
plt.title("Per-Class Accuracy (%)", fontsize=13, fontweight="bold"); plt.ylabel("Accuracy (%)"); plt.ylim(0,115)
for i,(l,a) in enumerate(zip(labels, per_acc)):
    plt.text(i, a*100+1, f"{a*100:.1f}%", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/per_class_accuracy.png", dpi=150); plt.close()

# Inference demo
rng_inf = np.random.RandomState(77)
idxs = rng_inf.choice(len(X_test), 16, replace=False)
yp16 = np.argmax(best_m.predict(X_test[idxs], verbose=0), axis=1)
fig, axes = plt.subplots(4, 4, figsize=(8,8))
fig.suptitle("Model Predictions (green=correct, red=wrong)", fontsize=11, fontweight="bold")
for i, (ax2, idx2) in enumerate(zip(axes.flatten(), idxs)):
    cmap = "Greens" if yp16[i]==y_test[idx2] else "Reds"
    ax2.imshow(X_test[idx2].squeeze(), cmap=cmap)
    ax2.set_title(f"P:{yp16[i]} T:{y_test[idx2]}", fontsize=9); ax2.axis("off")
plt.tight_layout(); plt.savefig("outputs/inference_demo.png", dpi=150); plt.close()

# Misclassified
wrong = np.where(y_test != best_p)[0][:20]
if len(wrong) > 0:
    n_cols=5; n_rows=(len(wrong)+4)//5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    fig.suptitle("Misclassified Samples", fontsize=12, fontweight="bold")
    axes_flat = axes.flatten() if n_rows>1 else axes
    for i, idx2 in enumerate(wrong):
        axes_flat[i].imshow(X_test[idx2].squeeze(), cmap="Reds")
        axes_flat[i].set_title(f"T:{y_test[idx2]} P:{best_p[idx2]}", fontsize=7); axes_flat[i].axis("off")
    for j in range(i+1, len(axes_flat)): axes_flat[j].axis("off")
    plt.tight_layout(); plt.savefig("outputs/misclassified_best_cnn.png", dpi=150); plt.close()

# Save model
best_m.save("outputs/best_character_recognition_model.keras")

# Model comparison
df = pd.DataFrame(results).T; df.to_csv("outputs/model_comparison.csv")
plt.figure(figsize=(8,4))
x = np.arange(2); w = 0.35
plt.bar(x-w/2, [results[n]["Test Accuracy"]*100 for n in results], w, label="Accuracy", color="#3498db")
plt.bar(x+w/2, [results[n]["Test F1 (weighted)"]*100 for n in results], w, label="F1-Score", color="#2ecc71")
plt.xticks(x, list(results.keys())); plt.title("Model Comparison", fontsize=13, fontweight="bold")
plt.ylabel("Score (%)"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150); plt.close()

print(f"\n🏆 Best Test Accuracy: {best_a:.4f} ({best_a*100:.2f}%)")
print("✅ All Task 3 outputs saved!")
