import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening
from skimage import io, color
import joblib
import time

import fuzzycmeans as fcm#main


class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tumor Detection App")

        # Initialize variables
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.svm_model = None
        self.model_filename = "svm_model.pkl"

        # Create GUI components
        self.label = tk.Label(root, text="Tumor Detection App", font=('Helvetica', 18))
        self.label.pack(pady=10)

        self.load_yes_button = tk.Button(root, text="Load Yes Data", command=self.load_yes_data)
        self.load_yes_button.pack(pady=5)

        self.load_no_button = tk.Button(root, text="Load No Data", command=self.load_no_data)
        self.load_no_button.pack(pady=5)

        self.train_button = tk.Button(root, text="Train Classifier", command=self.train_classifier)
        self.train_button.pack(pady=5)

        self.accuracy_button = tk.Button(root, text="Calculate Accuracy", command=self.calculate_accuracy)
        self.accuracy_button.pack(pady=5)

        self.save_model_button = tk.Button(root, text="Save Model", command=self.save_model)
        self.save_model_button.pack(pady=5)

        self.load_model_button = tk.Button(root, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack(pady=5)

    def load_yes_data(self):
        data_dir = filedialog.askdirectory(initialdir="/", title="Select Yes Data Directory")
        if data_dir:
            print("Loading Yes Data from:", data_dir)
            # Load labeled data from the "yes" folder
            X_yes, y_yes = self.load_labeled_data(data_dir, label=1)
            self.update_training_data(X_yes, y_yes)

    def load_no_data(self):
        data_dir = filedialog.askdirectory(initialdir="/", title="Select No Data Directory")
        if data_dir:
            print("Loading No Data from:", data_dir)
            # Load labeled data from the "no" folder
            X_no, y_no = self.load_labeled_data(data_dir, label=0)
            self.update_training_data(X_no, y_no)

    def load_labeled_data(self, data_dir, label):
        features = []
        labels = []
        for filename in os.listdir(data_dir):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            features.append(img.flatten())
            labels.append(label)
        return np.array(features), np.array(labels)

    def update_training_data(self, X_data, y_data):
        if self.X_train is None:
            self.X_train = X_data
            self.y_train = y_data
        else:
            self.X_train = np.concatenate((self.X_train, X_data), axis=0)
            self.y_train = np.concatenate((self.y_train, y_data), axis=0)

    def train_classifier(self):
        if self.X_train is not None and self.y_train is not None:
            print("Training Classifier...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=0.2, random_state=42)
            self.svm_model = SVC(kernel='linear')
            self.svm_model.fit(self.X_train, self.y_train)
            print("Classifier Trained Successfully!")
        else:
            print("No Training Data Available.")

    def calculate_accuracy(self):
        if self.svm_model is not None and self.X_test is not None and self.y_test is not None:
            y_pred = self.svm_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print("Accuracy:", accuracy)
            self.display_highlighted_image()
        else:
            print("Classifier or Test Data not available.")

    def display_highlighted_image(self):
        if self.X_test is not None:
            # Load the original image
            filename = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                                  filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"),))
            if filename:
                I = io.imread(filename)
                if len(I.shape) == 3 and I.shape[2] == 3:
                    I = color.rgb2gray(I)

                # Fuzzy C-Means clustering
                num_clusters = 2
                num_pixels = I.size
                feature_vector = I.reshape(num_pixels, 1)
                fcm_instance = fcm.FCM(n_clusters=num_clusters)
                fcm_instance.fit(feature_vector)
                u = fcm_instance.u
                idx = np.argmax(u, axis=1)
                segmented_image = idx.reshape(I.shape)
                tumor_mask = segmented_image == 1
                se = disk(5)
                tumor_mask = opening(tumor_mask, se)

                # Apply tumor mask to original image
                highlighted_image = np.copy(I)
                highlighted_image[tumor_mask] = 1

                # Display the highlighted image
                plt.imshow(highlighted_image, cmap='Oranges')
                plt.title('Highlighted Tumor Region')
                plt.axis('off')
                plt.show()

    def save_model(self):
        if self.svm_model:
            joblib.dump(self.svm_model, self.model_filename)
            print("Model saved successfully.")

    def load_model(self):
        if os.path.exists(self.model_filename):
            self.svm_model = joblib.load(self.model_filename)
            print("Model loaded successfully.")
        else:
            print("Model file not found.")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectionApp(root)
    app.run()
