import fuzzycmeans as fcm
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image
filename = 'brain_mri.jpg'
I = io.imread(filename)

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

# Compute the centroids
cntr = np.dot(u.T, feature_vector) / np.sum(u, axis=0)[:, None]

# Plot the cluster centers
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(cntr[:, 0], np.zeros_like(cntr[:, 0]), c='r', marker='o', s=100)
ax.set_xlabel('Intensity')
ax.set_title('Cluster Centers')
plt.show()