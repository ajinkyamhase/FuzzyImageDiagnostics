import os
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import joblib
from skimage.morphology import disk, opening
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import skfuzzy as fuzz
import matplotlib.pyplot as plt


class FuzzyMinMaxClassifier:
    def __init__(self, gamma=1.0, theta=0.1):
        self.gamma = gamma
        self.theta = theta
        self.hyperboxes = []

    def fit(self, X, y):
        for point, label in zip(X, y):
            self.add_hyperbox(point, label)

    def predict(self, X):
        predictions = []
        for point in X:
            predictions.append(self.classify_point(point))
        return predictions

    def add_hyperbox(self, point, label):
        expanded = False
        for box in self.hyperboxes:
            if box['class'] == label and all(box['min'] - self.theta <= point) and all(point <= box['max'] + self.theta):
                box['min'] = np.minimum(box['min'], point)
                box['max'] = np.maximum(box['max'], point)
                expanded = True
                break
        if not expanded:
            self.hyperboxes.append({'min': point, 'max': point, 'class': label})

    def classify_point(self, point):
        matches = [box['class'] for box in self.hyperboxes if np.all(box['min'] <= point) and np.all(point <= box['max'])]
        return max(matches, key=matches.count) if matches else 0

class TumorDetectionApp:
    def __init__(self):
        self.model_filename = "fuzzy_min_max_model.pkl"
        self.accuracy_filename = "accuracy_history.pkl"
        self.load_model()
        #if 'accuracy_history' not in st.session_state:
         #   st.session_state['accuracy_history'] = []
    def load_model(self):
        try:
            self.classifier = joblib.load(self.model_filename)
            st.success("Model loaded successfully.")
        except FileNotFoundError:
            self.classifier = FuzzyMinMaxClassifier()
            st.warning("No pre-trained model found. A new model will be created.")
        self.load_accuracy_history()

    def save_model(self):
        joblib.dump(self.classifier, self.model_filename)
        joblib.dump(st.session_state['accuracy_history'], self.accuracy_filename)
        st.success("Model and accuracy data saved successfully.")

    def load_accuracy_history(self):
        if os.path.exists(self.accuracy_filename):
            st.session_state['accuracy_history'] = joblib.load(self.accuracy_filename)
        else:
            st.session_state['accuracy_history'] = []

    def load_labeled_data(self, uploaded_files, label):
        features = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file).convert('L').resize((100, 100))
            img = np.array(img)
            features.append(img.flatten())
        labels = [label] * len(features)
        return np.array(features), np.array(labels)

    def update_training_data(self, X_data, y_data):
        if 'X_train' not in st.session_state:
            st.session_state['X_train'] = X_data
            st.session_state['y_train'] = y_data
        else:
            st.session_state['X_train'] = np.concatenate((st.session_state['X_train'], X_data), axis=0)
            st.session_state['y_train'] = np.concatenate((st.session_state['y_train'], y_data), axis=0)

    def train_classifier(self):
        if st.session_state['X_train'] is not None and st.session_state['y_train'] is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state['X_train'], st.session_state['y_train'], test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            st.session_state['X_test'], st.session_state['y_test'] = X_test, y_test
            self.save_model()
            st.success("Classifier trained successfully!")
        else:
            st.error("No data available for training.")

    def calculate_accuracy(self):
        if 'X_test' in st.session_state and 'y_test' in st.session_state:
            y_pred = self.classifier.predict(st.session_state['X_test'])
            accuracy = accuracy_score(st.session_state['y_test'], y_pred)
            st.write(f"Accuracy: {accuracy * 100:.2f}%")
            st.session_state['accuracy_history'].append(accuracy)
            self.save_model()
            self.plot_accuracy_history()
            return accuracy*100
        else:
            st.error("No trained model available or missing test data.")

    def plot_accuracy_history(self):
        plt.figure()
        plt.plot(st.session_state['accuracy_history'], marker='o', linestyle='-', color='r')
        plt.title('Accuracy Over Time')
        plt.xlabel('Training Sessions')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        st.pyplot(plt)

    def classify_and_highlight_image(self, uploaded_file):
        img = Image.open(uploaded_file).convert('L').resize((100, 100))
        img_array = np.array(img).flatten()
        prediction = self.classifier.predict([img_array])[0]
        st.write(f'Prediction (1: Tumor, 0: No Tumor): {prediction}')
        if prediction == 1:
            self.display_highlighted_image(uploaded_file)

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
    '''def highlight_image(self, uploaded_file):
        img = Image.open(uploaded_file).convert('L')
        img_array = np.array(img)
        thresh = threshold_otsu(img_array)
        bw = opening(img_array > thresh, disk(2))
        cleared = clear_border(bw)
        label_image = label(cleared)
        regions = regionprops(label_image)
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            img_array[minr:maxr, minc:maxc] = 255
        highlighted_img = Image.fromarray(img_array)
        st.image(highlighted_img, caption='Highlighted Tumor Region', use_column_width=True)'''

    def run(self):
        st.title("Tumor Detection App")
        menu_choice = st.sidebar.radio("Menu", ["Load Yes Data", "Load No Data", "Train Classifier",
                                                "Calculate Accuracy", "Classify and Highlight Image"])
        if menu_choice in ["Load Yes Data", "Load No Data"]:
            data_type = "Yes" if menu_choice == "Load Yes Data" else "No"
            uploaded_files = st.file_uploader(f"Upload {data_type} Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
            if uploaded_files:
                label = 1 if menu_choice == "Load Yes Data" else 0
                X_data, y_data = self.load_labeled_data(uploaded_files, label)
                self.update_training_data(X_data, y_data)
                st.success(f"Loaded {len(uploaded_files)} images for '{data_type}' data.")

        elif menu_choice == "Train Classifier":
            self.train_classifier()

        elif menu_choice == "Calculate Accuracy":
            self.calculate_accuracy()

        elif menu_choice == "Classify and Highlight Image":
            uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                self.classify_and_highlight_image(uploaded_file)

if __name__ == "__main__":
    app = TumorDetectionApp()
    app.run()
