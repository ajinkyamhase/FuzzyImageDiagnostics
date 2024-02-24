import fuzzycmeans as fcm
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening
# Prompt the user to select an image file
from tkinter import filedialog
from tkinter import Tk

Tk().withdraw()

filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
# Load the image
#filename = 'brain_mri.jpg'
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

# Compute the centroids
cntr = np.dot(u.T, feature_vector) / np.sum(u, axis=0)[:, None]

# Determine the cluster for each pixel
idx = np.argmax(u, axis=1)
segmented_image = idx.reshape(I.shape)

# Display the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')
plt.show()

# Create a mask for the abnormal region (tumor)
threshold = 0.5  # You may need to adjust this threshold
abnormal_mask = segmented_image == 1  # Assuming tumor is cluster 1

# Apply morphological operations to further refine the abnormal region
se = disk(5)  # Define a disk-shaped structuring element
abnormal_mask = opening(abnormal_mask, se)  # Perform morphological opening

# Overlay the abnormal region on the original image
rgb_image = color.gray2rgb(I)  # Convert to an RGB image
rgb_image[abnormal_mask] = [1, 0, 0]  # Overlay in red

# Display the original image with the abnormalities highlighted
plt.imshow(rgb_image)
plt.title('Abnormalities Highlighted')
plt.show()

# Optional: Save the segmented image or the overlay image
# io.imsave('segmented_image.jpg', segmented_image)
# io.imsave('abnormalities_highlighted.jpg', rgb_image)

# Optional: Calculate statistics or region properties for further analysis
