import fuzzycmeans as fcm
from skimage import io, color
import numpy as np
from skimage.morphology import disk, opening
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the image
filename = 'h.jpg'  # Replace with the path to your image
I = io.imread(filename)

# Convert to grayscale if the image is RGB
if len(I.shape) == 3 and I.shape[2] == 3:
    I = color.rgb2gray(I)

# Define the number of clusters (2 for brain and tumor)
num_clusters = 2

# Reshape the image to a 1D array (feature vector)
num_pixels = I.size
feature_vector = I.reshape(num_pixels, 1)

# Fuzzy C-Means clustering
fcm_instance = fcm.FCM(n_clusters=num_clusters)
fcm_instance.fit(feature_vector)

# Get the membership matrix
u = fcm_instance.u

# Determine the cluster for each pixel
idx = np.argmax(u, axis=1)
segmented_image = idx.reshape(I.shape)

tumor_region = I[tumor_mask]
tumor_features = tumor_region.flatten()

# Extract features from the non-tumor region
non_tumor_mask = ~tumor_mask
non_tumor_region = I[non_tumor_mask]
non_tumor_features = non_tumor_region.flatten()

# Ensure the feature vectors have the same length
max_length = max(len(tumor_features), len(non_tumor_features))
tumor_features = np.pad(tumor_features, (0, max_length - len(tumor_features)), mode='constant')
non_tumor_features = np.pad(non_tumor_features, (0, max_length - len(non_tumor_features)), mode='constant')

# Combine features and labels
X = np.vstack((tumor_features, non_tumor_features))
y = np.concatenate((np.ones_like(tumor_features), np.zeros_like(non_tumor_features)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Apply tumor mask to original image
highlighted_image = np.copy(I)
highlighted_image[tumor_mask] = 1  # Highlight tumor region

# Display the highlighted image
plt.imshow(highlighted_image, cmap='Oranges')
plt.title('Highlighted Tumor Region')
plt.axis('off')
plt.show()