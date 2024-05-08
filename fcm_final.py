import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib
from skimage.morphology import disk, opening
from PIL import Image
import fuzzycmeans as fcm
import skfuzzy as fuzz
def display_highlighted_image(self, uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert('L')
    img = img.resize((256, 256))  # Resize for consistency
    img = np.array(img)

    # Prepare image data for clustering
    num_clusters = 2
    data = img.reshape(-1, 1)  # Reshape image for clustering

    # Perform Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, num_clusters, 2, error=0.005, maxiter=1000, init=None)

    # Find the cluster index. Assuming the tumor is darker (adjust based on your images)
    tumor_cluster_index = np.argmin(cntr.sum(axis=1))
    new_data = np.argmax(u, axis=0)
    segmented_image = new_data.reshape(img.shape)

    # Visualize the original segmented image
    segmented_img_pil = Image.fromarray((segmented_image * 255).astype(np.uint8))
    st.image(segmented_img_pil, caption='Original Segmented Image', use_column_width=True)

    # Create a mask for the tumor
    tumor_mask = segmented_image == tumor_cluster_index
    se = disk(2)  # Smaller disk size
    tumor_mask = opening(tumor_mask, se)

    # Highlight the tumor in the image
    highlighted_img = np.copy(img)
    highlighted_img[tumor_mask] = 255  # Highlight the tumor region with white color

    # Convert to PIL image for display in Streamlit
    highlighted_img = Image.fromarray(highlighted_img)
    st.image(highlighted_img, caption='Highlighted Tumor Region', use_column_width=True)
class TumorDetectionApp:
    def __init__(self):
        # Initialize session state for data and model
        if 'X_train' not in st.session_state:
            st.session_state['X_train'] = None
        if 'y_train' not in st.session_state:
            st.session_state['y_train'] = None
        if 'X_test' not in st.session_state:
            st.session_state['X_test'] = None
        if 'y_test' not in st.session_state:
            st.session_state['y_test'] = None
        if 'svm_model' not in st.session_state:
            st.session_state['svm_model'] = None
        self.model_filename = "svm_model.pkl"

    def load_labeled_data(self, uploaded_files, label):
        features = []
        labels = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file).convert('L').resize((100, 100))
            img = np.array(img)
            features.append(img.flatten())
            labels.append(label)
        return np.array(features), np.array(labels)

    def update_training_data(self, X_data, y_data):
        if st.session_state['X_train'] is None:
            st.session_state['X_train'] = X_data
            st.session_state['y_train'] = y_data
        else:
            st.session_state['X_train'] = np.concatenate((st.session_state['X_train'], X_data))
            st.session_state['y_train'] = np.concatenate((st.session_state['y_train'], y_data))

    def train_classifier(self):
        if st.session_state['X_train'] is not None and st.session_state['y_train'] is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state['X_train'], st.session_state['y_train'], test_size=0.2, random_state=42)
            st.session_state['X_test'], st.session_state['y_test'] = X_test, y_test
            st.session_state['svm_model'] = SVC(kernel='linear')
            st.session_state['svm_model'].fit(X_train, y_train)
            st.success("Classifier trained successfully!")
        else:
            st.error("No data available for training.")

    def calculate_accuracy(self):
        if st.session_state['svm_model'] and st.session_state['X_test'] is not None and st.session_state['y_test'] is not None:
            y_pred = st.session_state['svm_model'].predict(st.session_state['X_test'])
            accuracy = accuracy_score(st.session_state['y_test'], y_pred)
            st.write(f"Accuracy: {accuracy * 100:.2f}%")
        else:
            st.error("No trained model available or missing test data.")

    def save_model(self):
        if st.session_state['svm_model']:
            joblib.dump(st.session_state['svm_model'], self.model_filename)
            st.success("Model saved successfully.")

    def load_model(self):
        if os.path.exists(self.model_filename):
            st.session_state['svm_model'] = joblib.load(self.model_filename)
            st.success("Model loaded successfully.")

    def classify_and_highlight_image(self, uploaded_file):
        if uploaded_file is not None:
            # Load and preprocess the image
            img = Image.open(uploaded_file).convert('L').resize((100, 100))
            img_array = np.array(img).flatten()  # Flatten the image to a 1D array

            # Classify the image
            if st.session_state['svm_model']:
                prediction = st.session_state['svm_model'].predict([img_array])
                st.write(f'Prediction (1: Tumor, 0: No Tumor): {prediction[0]}')

                # If tumor is predicted, proceed to display highlighted image
                if prediction == 1:
                    self.display_highlighted_image(uploaded_file)
                else:
                    st.image(img, caption='No tumor detected', use_column_width=True)
            else:
                st.error("No trained model available. Please load or train the model.")

    def display_highlighted_image(self, uploaded_file):
        img = Image.open(uploaded_file)
        img = img.convert('L')
        img = img.resize((256, 256))  # Resize for consistency
        img = np.array(img)

        # Prepare image data for clustering
        num_clusters = 2
        data = img.reshape(-1, 1)  # Reshape image for clustering

        # Perform Fuzzy C-Means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, num_clusters, 2, error=0.005, maxiter=1000, init=None)

        # Find the cluster index. Assuming the tumor is darker (adjust based on your images)
        tumor_cluster_index = np.argmin(cntr.sum(axis=1))
        new_data = np.argmax(u, axis=0)
        segmented_image = new_data.reshape(img.shape)

        # Visualize the original segmented image
        segmented_img_pil = Image.fromarray((segmented_image * 255).astype(np.uint8))
        st.image(segmented_img_pil, caption='Original Segmented Image', use_column_width=True)

        # Create a mask for the tumor
        tumor_mask = segmented_image == tumor_cluster_index
        se = disk(2)  # Smaller disk size
        tumor_mask = opening(tumor_mask, se)

        # Highlight the tumor in the image
        highlighted_img = np.copy(img)
        highlighted_img[tumor_mask] = 255  # Highlight the tumor region with white color

        # Convert to PIL image for display in Streamlit
        highlighted_img = Image.fromarray(highlighted_img)
        st.image(highlighted_img, caption='Highlighted Tumor Region', use_column_width=True)

    def run(self):
        st.title("Tumor Detection App")
        menu_choice = st.sidebar.radio("Menu", ["Load Yes Data", "Load No Data", "Train Classifier",
                                                "Calculate Accuracy", "Save Model", "Load Model", "Classify and Highlight Image"])

        if menu_choice in ["Load Yes Data", "Load No Data"]:
            data_type = "Yes" if menu_choice == "Load Yes Data" else "No"
            uploaded_files = st.file_uploader(f"Upload {data_type} Images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_files:
                label = 1 if menu_choice == "Load Yes Data" else 0
                X_data, y_data = self.load_labeled_data(uploaded_files, label)
                self.update_training_data(X_data, y_data)
                st.success(f"Loaded {len(uploaded_files)} images for '{data_type}' data.")

        elif menu_choice == "Train Classifier":
            self.train_classifier()

        elif menu_choice == "Calculate Accuracy":
            self.calculate_accuracy()

        elif menu_choice == "Save Model":
            self.save_model()

        elif menu_choice == "Load Model":
            self.load_model()

        elif menu_choice == "Classify and Highlight Image":
            uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_file:
                self.classify_and_highlight_image(uploaded_file)

if __name__ == "__main__":
    app = TumorDetectionApp()
    app.run()
