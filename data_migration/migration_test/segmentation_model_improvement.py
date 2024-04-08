import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Define the function for preprocessing the image
def preprocess_image(image):
    # Apply Gaussian filter for noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply thresholding to segment the image
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return segmented_image

# Load the images from 'yes' and 'no' folders
def load_images(folder):
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
    return images

def extract_features(images, label):
    features = []
    for image_path in images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        segmented_image = preprocess_image(image)
        if segmented_image is not None:
            features.append((segmented_image.flatten(), label))
    return features

# Load images and extract features
yes_images = load_images('C:/Users/Ajinkya/Desktop/Final Year project/data_migration/yes')
no_images = load_images('C:/Users/Ajinkya/Desktop/Final Year project/data_migration/no')

# Extract features and labels
tumor_features = extract_features(yes_images, 1)
non_tumor_features = extract_features(no_images, 0)

# Ensure all features have the same shape
min_shape = min([len(f[0]) for f in tumor_features + non_tumor_features])
tumor_features = [(f[0][:min_shape], f[1]) for f in tumor_features]
non_tumor_features = [(f[0][:min_shape], f[1]) for f in non_tumor_features]

# Combine features and shuffle
all_features = tumor_features + non_tumor_features
np.random.shuffle(all_features)

# Separate features and labels
X = np.array([f[0] for f in all_features])
y = np.array([f[1] for f in all_features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier using the extracted features
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained classifier
joblib.dump(clf, 'brain_tumor_classifier.joblib')
