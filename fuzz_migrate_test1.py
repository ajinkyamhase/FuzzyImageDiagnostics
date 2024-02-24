from skimage import io, color
from skimage.segmentation import mark_boundaries
import skfuzzy
from skfuzzy import fcm
#from skfuzzy import cmeans, ggplot
# Prompt the user to select an image file
from tkinter import filedialog
from tkinter import Tk
import numpy as np

Tk().withdraw()

filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])

if not filename:
    print('No image selected. Exiting.')
else:
    # Load the image
    I = io.imread(filename)
    I = color.rgb2gray(I)  # Convert to grayscale
    I = I.astype(np.float64)  # Convert to double data type

    # Define the number of clusters (2 for brain and tumor)
    num_clusters = 2

    # Fuzzy C-Means clustering
    center, U, obj_fcn = fcm(I.flatten(), num_clusters)

    # Determine the cluster for each pixel
    idx = np.argmax(U, axis=1)
    segmented_image = idx.reshape(I.shape)

    # Display the original and segmented images
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(I, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(segmented_image, cmap='gray')
    axes[1].set_title('Segmented Image')
    plt.show()

    # Create a mask for the abnormal region (tumor)
    threshold = 0.5  # You may need to adjust this threshold
    abnormal_mask = segmented_image == 2  # Assuming tumor is cluster 2

    # Optionally, apply morphological operations to further refine the abnormal region
    from skimage.morphology import disk, opening

    se = disk(5)  # Define a disk-shaped structuring element
    abnormal_mask = opening(abnormal_mask, se)  # Perform morphological opening

    # Overlay the abnormal region on the original image
    rgb_image = color.gray2rgb(I)  # Convert to an RGB image
    rgb_image[abnormal_mask] = [1, 0, 0]  # Overlay in red

    # Display the original image with the abnormalities highlighted
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(I, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(rgb_image)
    axes[1].set_title('Abnormalities Highlighted')
    plt.show()

    # Optional: Save the segmented image or the overlay image
    # io.imsave('segmented_image.jpg', segmented_image)
    # io.imsave('abnormalities_highlighted.jpg', rgb_image)

    # Optional: Calculate statistics or region properties for further analysis