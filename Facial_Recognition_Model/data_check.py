import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("fer2013/fer2013.csv")
print("Dataset shape:", data.shape)
print(data.head())
print("Unique emotion labels:", data['emotion'].unique())

# Show one image
example = data.iloc[0]
pixels = np.array(example['pixels'].split(), dtype='float32').reshape(48, 48)
plt.imshow(pixels, cmap='gray')
plt.title(f"Emotion: {example['emotion']}")
plt.show()
