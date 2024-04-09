import fuzzycmeans as fcm
from skimage import io, color
import numpy as np
from skimage.morphology import disk, opening
import cv2
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image
filename = 'h.jpg'  # Provide the path to your image
I = io.imread(filename)

# Check if the image is grayscale
if len(I.shape) == 3 and I.shape[2] == 3:
    # Convert RGB image to grayscale
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

# Create a mask for the tumor region
tumor_mask = segmented_image == 1  # Assuming tumor is cluster 1

# Apply morphological operations to further refine the tumor region
se = disk(5)  # Define a disk-shaped structuring element
tumor_mask = opening(tumor_mask, se)  # Perform morphological opening

# Extract features from the tumor region for classification
tumor_features = []
for filename in os.listdir(r'C:\Users\Ajinkya\Desktop\Final Year project\data_migration\yes'):
    img_path = os.path.join(r'C:\Users\Ajinkya\Desktop\Final Year project\data_migration\yes', filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.medianBlur(img, 3)  # Apply median filter to remove noise
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Apply Otsu thresholding
    tumor_features.append(img.flatten())

# Similarly, extract features from the non-tumor region for classification
non_tumor_features = []
for filename in os.listdir(r'C:\Users\Ajinkya\Desktop\Final Year project\data_migration\no'):
    img_path = os.path.join(r'C:\Users\Ajinkya\Desktop\Final Year project\data_migration\no', filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.medianBlur(img, 3)  # Apply median filter to remove noise
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Apply Otsu thresholding
    non_tumor_features.append(img.flatten())

# Combine features and labels
X = np.vstack((tumor_features, non_tumor_features))
y = np.concatenate((np.ones(len(tumor_features)), np.zeros(len(non_tumor_features))))

# Perform PCA to reduce dimensionality to 3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Plot 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot tumor features in blue
ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2], c='b', label='Tumor')

# Plot non-tumor features in red
ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2], c='r', label='Non-Tumor')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D Scatter Plot of Tumor and Non-Tumor Features')
ax.legend()

plt.show()
