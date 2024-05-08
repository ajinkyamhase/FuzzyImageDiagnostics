import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib
from skimage.morphology import disk, opening
import fuzzycmeans as fcm
from PIL import Image

class TumorDetectionApp:
    def __init__(self):
        # Initialize variables
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.svm_model = None
        self.model_filename = "svm_model.pkl"

    def load_labeled_data(self, uploaded_files, label):
        features = []
        labels = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            img = img.convert('L')
            img = img.resize((100, 100))
            img = np.array(img)
            features.append(img.flatten())
            labels.append(label)
        return np.array(features), np.array(labels)

    def update_training_data(self, X_data, y_data):
        if self.X_train is None:
            self.X_train = X_data
            self.y_train = y_data
        else:
            self.X_train = np.concatenate((self.X_train, X_data))
            self.y_train = np.concatenate((self.y_train, y_data))

    def train_classifier(self):
        if self.X_train is not None and self.y_train is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42)
            self.svm_model = SVC(kernel='linear')
            self.svm_model.fit(self.X_train, self.y_train)
            st.success("Classifier trained successfully!")
        else:
            st.success("Classifier not trained")

    def calculate_accuracy(self):
        if self.svm_model is not None and self.X_test is not None and self.y_test is not None:
            y_pred = self.svm_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            st.write(f"Accuracy: {accuracy * 100:.2f}%")

    def display_highlighted_image(self, uploaded_file):
        if uploaded_file is not None:
            img_data = uploaded_file.getvalue()
            image = cv2.imdecode(np.fromstring(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)

            # Apply Fuzzy C-Means clustering
            num_clusters = 2
            num_pixels = image.size
            feature_vector = image.reshape(num_pixels, 1)
            fcm_instance = fcm.FCM(n_clusters=num_clusters)
            fcm_instance.fit(feature_vector)
            u = fcm_instance.u
            idx = np.argmax(u, axis=1)
            segmented_image = idx.reshape(image.shape)
            tumor_mask = segmented_image == 1
            se = disk(5)
            tumor_mask = opening(tumor_mask, se)

            highlighted_image = np.copy(image)
            highlighted_image[tumor_mask] = 255

            st.image(highlighted_image, caption='Highlighted Tumor Region', use_column_width=True)

    def save_model(self):
        if self.svm_model:
            joblib.dump(self.svm_model, self.model_filename)
            st.success("Model saved successfully.")

    def load_model(self):
        if os.path.exists(self.model_filename):
            self.svm_model = joblib.load(self.model_filename)
            st.success("Model loaded successfully.")

    def run(self):
        st.title("Tumor Detection App")
        menu_choice = st.sidebar.radio("Menu", ["Load Yes Data", "Load No Data", "Train Classifier",
                                                "Display Highlighted Image", "Save Model", "Load Model"])

        if menu_choice == "Load Yes Data" or menu_choice == "Load No Data":
            uploaded_files = st.file_uploader(f"Upload {menu_choice.split()[1]} Data",
                                              accept_multiple_files=True, type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_files:
                label = 1 if menu_choice == "Load Yes Data" else 0
                X_data, y_data = self.load_labeled_data(uploaded_files, label)
                self.update_training_data(X_data, y_data)
                st.success(f"{len(uploaded_files)} images loaded successfully for '{menu_choice.split()[1]}' data.")

        elif menu_choice == "Train Classifier":
            self.train_classifier()

        elif menu_choice == "Display Highlighted Image":
            uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png", "bmp"])
            self.display_highlighted_image(uploaded_file)

        elif menu_choice == "Save Model":
            self.save_model()

        elif menu_choice == "Load Model":
            self.load_model()

if __name__ == "__main__":
    app = TumorDetectionApp()
    app.run()
