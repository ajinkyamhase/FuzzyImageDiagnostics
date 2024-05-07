import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib
from skimage.morphology import disk, opening
import fuzzycmeans as fcm

class TumorDetectionApp:
    def __init__(self):
        # Initialize variables
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.svm_model = None
        self.model_filename = "svm_model.pkl"

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
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=0.2, random_state=42)
            self.svm_model = SVC(kernel='linear')
            self.svm_model.fit(self.X_train, self.y_train)

    def calculate_accuracy(self):
        if self.svm_model is not None and self.X_test is not None and self.y_test is not None:
            y_pred = self.svm_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            return accuracy

    def display_highlighted_image(self):
        # Load the original image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file is not None:
            I = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

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
            highlighted_image[tumor_mask] = 255

            # Display the highlighted image
            st.image(highlighted_image, caption='Highlighted Tumor Region', use_column_width=True)

    def save_model(self):
        if self.svm_model:
            joblib.dump(self.svm_model, self.model_filename)

    def load_model(self):
        if os.path.exists(self.model_filename):
            self.svm_model = joblib.load(self.model_filename)

    def run(self):
        st.title("Tumor Detection App")

        # Create sidebar
        st.sidebar.header("Menu")
        menu_choice = st.sidebar.radio("Select an option", ("Load Yes Data", "Load No Data", "Train Classifier",
                                                            "Display Highlighted Image",
                                                            "Save Model", "Load Model"))

        if menu_choice == "Load Yes Data":
            data_dir = st.sidebar.text_input("Enter the path to the directory containing 'yes' data:", "")
            if data_dir:
                X_yes, y_yes = self.load_labeled_data(data_dir, label=1)
                self.update_training_data(X_yes, y_yes)
                st.sidebar.success("Yes Data Loaded Successfully!")

        elif menu_choice == "Load No Data":
            data_dir = st.sidebar.text_input("Enter the path to the directory containing 'no' data:", "")
            if data_dir:
                X_no, y_no = self.load_labeled_data(data_dir, label=0)
                self.update_training_data(X_no, y_no)
                st.sidebar.success("No Data Loaded Successfully!")

        elif menu_choice == "Train Classifier":
            self.train_classifier()
            st.sidebar.success("Classifier Trained Successfully!")

        elif menu_choice == "Display Highlighted Image":
            self.display_highlighted_image()

        elif menu_choice == "Save Model":
            self.save_model()
            st.sidebar.success("Model saved successfully.")

        elif menu_choice == "Load Model":
            self.load_model()
            st.sidebar.success("Model loaded successfully.")

if __name__ == "__main__":
    app = TumorDetectionApp()
    app.run()
