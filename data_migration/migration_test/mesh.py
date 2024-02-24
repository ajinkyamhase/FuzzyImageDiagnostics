import fuzzycmeans as fcm
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image
filename = 'brain_mri.jpg'
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

# Reshape membership matrix to match image shape
u_img = u.reshape(I.shape[0], I.shape[1], num_clusters)

# Create meshgrid
x = np.arange(I.shape[1])
y = np.arange(I.shape[0])
x, y = np.meshgrid(x, y)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot colored mesh
for i in range(num_clusters):
    ax.plot_surface(x, y, u_img[:, :, i], cmap='viridis', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
ax.set_title('3D Colored Mesh Plot of Cluster Memberships')
plt.show()
