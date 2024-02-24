import os
import cv2
from skimage.color import rgb2gray
from sklearn.cluster import FuzzyCMeans

# Prompt user to select image
filename = input("Enter image filename (or leave blank for interactive selection): ")

# Handle interactive image selection if no filename provided
if not filename:
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])

if not filename:
    print("No image selected. Exiting.")
    exit()

# Load image
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

# Convert to grayscale if needed (adjust for color spaces)
if len(image.shape) == 3:
    image = rgb2gray(image)

# Reshape image to 2D array for clustering
image_flat = image.reshape(-1, 1)

# Define number of clusters (adjust as needed)
num_clusters = 2

# Fuzzy C-Means clustering
fcm = FuzzyCMeans(n_clusters=num_clusters, max_iter=1000, random_state=42)  # Set random_state for reproducibility
fcm.fit(image_flat)
centers = fcm.cluster_centers_
U = fcm.u

# Determine cluster for each pixel
cluster_labels = np.argmax(U, axis=1).reshape(image.shape[:2])

# Display original and segmented images
cv2.imshow("Original Image", image)
cv2.imshow("Segmented Image", cluster_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional post-processing steps (e.g., morphological operations, noise removal)

# Create abnormal region mask (adjust threshold and cluster index)
threshold = 0.5
abnormal_mask = cluster_labels == 1  # Assuming tumor is cluster 1

# Apply morphological operations (adjust parameters as needed)
kernel = np.ones((5, 5), np.uint8)  # Disk-shaped structuring element
abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_OPEN, kernel)  # Morphological opening

# Overlay abnormal region on original image
overlay_image = image.copy()
overlay_image[:, :, 2] = abnormal_mask * 255  # Overlay in red

# Display original image with highlighted abnormalities
cv2.imshow("Original Image with Abnormalities Highlighted", overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save segmented image or overlay image
# cv2.imwrite("segmented_image.png", cluster_labels)
# cv2.imwrite("abnormalities_highlighted.png", overlay_image)

# Optional: Calculate statistics or region properties for further analysis

print("Segmentation completed successfully!")
