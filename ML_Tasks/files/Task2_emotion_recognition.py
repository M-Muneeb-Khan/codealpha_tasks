import numpy as np, os, warnings, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings("ignore"); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf; tf.get_logger().setLevel('ERROR')
from tensorflow import keras; from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd

os.makedirs("outputs", exist_ok=True)
np.random.seed(42); tf.random.set_seed(42)

EMOTIONS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
EMOTION_COLORS = {"neutral":"#95a5a6","calm":"#3498db","happy":"#f1c40f",
                  "sad":"#2980b9","angry":"#e74c3c","fearful":"#8e44ad",
                  "disgust":"#27ae60","surprised":"#e67e22"}
N_MFCC, N_CHROMA, MAX_LEN = 40, 12, 100
N_FEAT = N_MFCC*3 + N_CHROMA  # 132

def make_features(emotion_idx, seed):
    rng = np.random.RandomState(emotion_idx * 100000 + seed)
    patterns = {0:(0.4,0.2),1:(0.3,0.15),2:(1.5,0.9),3:(0.25,0.2),
                4:(2.0,1.1),5:(1.3,1.0),6:(0.8,0.6),7:(1.8,1.2)}
    amp, var = patterns[emotion_idx]
    t = np.linspace(0, 2*np.pi, MAX_LEN)
    freq = 1.0 + emotion_idx * 0.7
    signal = amp * np.sin(freq * t + rng.uniform(0, np.pi))
    feat = np.zeros((N_FEAT, MAX_LEN), dtype=np.float32)
    for i in range(N_FEAT):
        phase = rng.uniform(0, 2*np.pi)
        harmonic = amp * 0.5 * np.sin(2*freq*t + phase)
        noise = rng.randn(MAX_LEN) * var
        feat[i] = signal * np.cos(i * 0.15 + phase) + harmonic * 0.3 + noise
    # Add emotion-specific frequency emphasis
    feat[:N_MFCC] *= (1 + emotion_idx * 0.1)
    feat[N_MFCC:N_MFCC*2] *= (1 + rng.uniform(0,0.3))  # delta
    return feat

print("Building emotion dataset (120 samples/class)...")
X, y = [], []
for i, emo in enumerate(EMOTIONS):
    for j in range(120):
        X.append(make_features(i, j))
        y.append(emo)
X = np.array(X, dtype=np.float32)
y = np.array(y)
le = LabelEncoder(); y_enc = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"X: {X.shape}, classes: {le.classes_}")

# EDA: feature samples
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.suptitle("MFCC Feature Maps by Emotion", fontsize=14, fontweight="bold")
for i, emo in enumerate(EMOTIONS):
    ax = axes[i//4, i%4]
    idx = np.where(y == emo)[0][0]
    im = ax.imshow(X[idx], aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(f"{emo.capitalize()}", fontsize=11)
    ax.set_xlabel("Time Frames"); ax.set_ylabel("Feature")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig("outputs/feature_samples.png", dpi=150); plt.close()

# Class distribution
unique, counts = np.unique(y, return_counts=True)
colors = [EMOTION_COLORS[e] for e in unique]
plt.figure(figsize=(10,4))
bars = plt.bar(unique, counts, color=colors, edgecolor="white")
plt.title("Emotion Class Distribution", fontsize=14, fontweight="bold"); plt.xlabel("Emotion"); plt.ylabel("Count")
for bar, c in zip(bars, counts): plt.text(bar.get_x()+bar.get_width()/2, c+1, str(c), ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/class_distribution.png", dpi=150); plt.close()
print("EDA plots saved")

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
X_cnn_tr = X_tr[..., np.newaxis]; X_cnn_val = X_val[..., np.newaxis]; X_cnn_test = X_test[..., np.newaxis]

# CNN model
def build_cnn(input_shape, n_classes):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D((2,2))(x); x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D((2,2))(x); x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="EmotionCNN")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

# LSTM model
def build_lstm(input_shape, n_classes):
    inp = keras.Input(shape=input_shape)
    x = layers.Permute((2,1))(inp)  # (time, features)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="EmotionLSTM")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

cb = [EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
      ReduceLROnPlateau(factor=0.5, patience=4, verbose=0)]

print("Training CNN...")
cnn = build_cnn(X_cnn_tr.shape[1:], n_classes)
h_cnn = cnn.fit(X_cnn_tr, y_tr, epochs=40, batch_size=16,
                validation_data=(X_cnn_val, y_val), callbacks=cb, verbose=0)
print(f"  Best val_acc: {max(h_cnn.history['val_accuracy']):.4f}")

print("Training LSTM...")
lstm = build_lstm(X_tr.shape[1:], n_classes)
h_lstm = lstm.fit(X_tr, y_tr, epochs=40, batch_size=16,
                  validation_data=(X_val, y_val), callbacks=cb, verbose=0)
print(f"  Best val_acc: {max(h_lstm.history['val_accuracy']):.4f}")

emotion_names = le.classes_
results = {}
for name, m, X_te in [("CNN", cnn, X_cnn_test), ("LSTM", lstm, X_test)]:
    yp = np.argmax(m.predict(X_te, verbose=0), axis=1)
    acc = accuracy_score(y_test, yp); f1w = f1_score(y_test, yp, average="weighted")
    results[name] = {"Test Accuracy": acc, "Test F1 (weighted)": f1w}
    print(f"\n{name}: Acc={acc:.4f}  F1={f1w:.4f}")
    print(classification_report(y_test, yp, target_names=emotion_names))
    cm = confusion_matrix(y_test, yp)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title(f"{name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig(f"outputs/confusion_matrix_{name.lower()}.png", dpi=150); plt.close()

# Training history
fig, ax = plt.subplots(1,2, figsize=(13,4))
fig.suptitle("Emotion Recognition — Training History", fontsize=13, fontweight="bold")
ax[0].plot(h_cnn.history["accuracy"], "b-", label="CNN Train")
ax[0].plot(h_cnn.history["val_accuracy"], "b--", label="CNN Val")
ax[0].plot(h_lstm.history["accuracy"], "r-", label="LSTM Train")
ax[0].plot(h_lstm.history["val_accuracy"], "r--", label="LSTM Val")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend(fontsize=8)
ax[1].plot(h_cnn.history["loss"], "b-", label="CNN Train")
ax[1].plot(h_cnn.history["val_loss"], "b--", label="CNN Val")
ax[1].plot(h_lstm.history["loss"], "r-", label="LSTM Train")
ax[1].plot(h_lstm.history["val_loss"], "r--", label="LSTM Val")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("outputs/training_history.png", dpi=150); plt.close()

cnn.save("outputs/emotion_cnn_model.keras")
lstm.save("outputs/emotion_lstm_model.keras")

df = pd.DataFrame(results).T; df.to_csv("outputs/model_metrics.csv")
print("\nAll outputs saved!")
print(f"CNN: {results['CNN']['Test Accuracy']:.4f}  LSTM: {results['LSTM']['Test Accuracy']:.4f}")
