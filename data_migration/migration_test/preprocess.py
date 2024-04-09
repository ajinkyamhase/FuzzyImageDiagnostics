import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from skimage import io, color
from sklearn.mixture import GaussianMixture
import ipyvolume as ipv

# Function to preprocess image using fuzzy clustering
def preprocess_image(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented_image

# Function to load images from a folder
def load_images(folder):
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
    return images

# Load and preprocess image
filename = 'h.jpg'  # Provide the path to your image
I = io.imread(filename)
if len(I.shape) == 3 and I.shape[2] == 3:
    I = color.rgb2gray(I)
tumor_mask = preprocess_image(I)

# Load brain image
brain = io.imread('brain_image.jpg')  # Load your brain image here

# Apply tumor mask to brain image
highlighted_brain = np.copy(brain)
highlighted_brain[tumor_mask == 0] = 0  # Set non-tumor region to black

# Create a figure
fig = ipv.figure()

# Plot brain
vol = ipv.volshow(brain, lighting=True, level=[0.2, 0.7, 0.9], opacity=0.03, data_min=0, data_max=255)

# Plot highlighted tumor region
highlighted_tumor = np.zeros_like(brain)
highlighted_tumor[tumor_mask > 0] = 255
highlighted_tumor = highlighted_tumor.astype(np.uint8)
highlighted_tumor = np.moveaxis(highlighted_tumor, [0, 1, 2], [2, 0, 1])
ipv.volshow(highlighted_tumor, level=0.5, opacity=0.2, extent=[[0, 255], [0, 255], [0, 255]])

# Set view angle
ipv.view(azimuth=45, elevation=30)

# Show the plot
ipv.show()
