import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- Load Pickle Data ---
with open('processed_mfcc/ravdess_mfcc.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract features and labels
features = [entry[0] for entry in data]  # MFCC arrays
labels = [entry[1] for entry in data]    # integer labels

# --- Pad MFCC arrays to the same length ---
max_len = max(feat.shape[0] for feat in features)  # maximum frames length (e.g. 93)
num_coeffs = features[0].shape[1]                   # MFCC coefficients (e.g. 13)

X = np.array([
    np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode='constant') if f.shape[0] < max_len else f[:max_len]
    for f in features
])

# --- Normalize each sample individually (zero mean, unit variance) ---
X = np.array([(x - np.mean(x)) / (np.std(x) + 1e-10) for x in X])

# --- Add channel dimension for Conv2D ---
X = X[..., np.newaxis]  # shape: (samples, max_len, num_coeffs, 1)

# --- One-hot encode labels ---
num_classes = max(labels) + 1
y = to_categorical(labels, num_classes=num_classes)

# --- Split dataset into train and test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Build CNN model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(max_len, num_coeffs, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train model ---
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# --- Save the trained model ---
model.save('models/cnn_emotion_model.keras')
print("✅ Model saved as cnn_emotion_model.keras")

# --- Pause/Filler Detection Function ---
def detect_pauses(mfcc, energy_threshold=0.01, min_pause_frames=5):
    """
    Detects pauses in an MFCC array based on low energy.
    Returns a list of (start, end) frame indices for detected pauses.
    """
    frame_energy = np.mean(np.abs(mfcc), axis=1)
    low_energy = frame_energy < energy_threshold

    pauses = []
    start = None
    for i, is_pause in enumerate(low_energy):
        if is_pause and start is None:
            start = i
        elif not is_pause and start is not None:
            if i - start >= min_pause_frames:
                pauses.append((start, i - 1))
            start = None
    # Handle pause at end
    if start is not None and len(mfcc) - start >= min_pause_frames:
        pauses.append((start, len(mfcc) - 1))
    return pauses

# --- Example: Detect pauses in the first sample ---
sample_mfcc = features[0]
pauses = detect_pauses(sample_mfcc)
print(f"Detected {len(pauses)} pause(s) in first sample: {pauses}")
