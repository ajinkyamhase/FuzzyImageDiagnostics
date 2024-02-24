from skimage import io, color
import fuzzycmeans as fcm
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening

# Prompt the user to select an image file
from tkinter import filedialog
from tkinter import Tk

Tk().withdraw()

filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])

if not filename:
    print('No image selected. Exiting.')
else:
    # Load the image
    I = io.imread(filename)

    # Check if the image is grayscale
    if len(I.shape) == 3 and I.shape[2] == 3:
        # Convert RGB image to grayscale
        I_gray = color.rgb2gray(I)
    else:
        # If it's already grayscale, keep it as is
        I_gray = I.astype(np.float64)

    # Define the number of clusters (2 for brain and tumor)
    num_clusters = 2

    # Fuzzy C-Means clustering
    cntr, u, u0, d, jm, p, fpc, obj_func = fcm.fcm(I_gray.flatten(), num_clusters, 2, 0.001, 1000)

    # Determine the cluster for each pixel
    idx = np.argmax(u, axis=1)
    segmented_image = idx.reshape(I_gray.shape)

    # Display the original and segmented images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(I_gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(segmented_image, cmap='gray')
    axes[1].set_title('Segmented Image')
    plt.show()

    # Create a mask for the abnormal region (tumor)
    threshold = 0.5  # You may need to adjust this threshold
    abnormal_mask = segmented_image == 1  # Assuming tumor is cluster 1

    # Optionally, apply morphological operations to further refine the abnormal region
    se = disk(5)  # Define a disk-shaped structuring element
    abnormal_mask = opening(abnormal_mask, se)  # Perform morphological opening

    # Overlay the abnormal region on the original image
    rgb_image = color.gray2rgb(I_gray)  # Convert to an RGB image
    rgb_image[abnormal_mask] = [1, 0, 0]  # Overlay in red

    # Display the original image with the abnormalities highlighted
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(I_gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(rgb_image)
    axes[1].set_title('Abnormalities Highlighted')
    plt.show()
