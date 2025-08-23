import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Convert pixels to numpy arrays
def process_pixels(pixel_string):
    pixels = np.array(pixel_string.split(), dtype='float32')
    return pixels.reshape(48, 48, 1)  # shape (48,48,1) for CNN


# Load dataset
data = pd.read_csv("fer2013/fer2013.csv")

# Apply conversion
images = np.array([process_pixels(p) for p in data['pixels']])
labels = data['emotion'].values

# Normalize images (0-1)
images = images / 255.0

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Save as numpy files
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Preprocessing complete.")
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
