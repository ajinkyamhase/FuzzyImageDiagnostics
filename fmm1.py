import os
import numpy as np
from PIL import Image
import streamlit as st
from skimage.morphology import disk, opening
import skfuzzy as fuzz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Fuzzy Min-Max Classifier Implementation
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
        return max(set(matches), key=matches.count) if matches else 0

class TumorDetectionApp:
    def __init__(self):
        self.model_filename = "fuzzy_min_max_model.pkl"
        self.load_model()

    def load_model(self):
        if 'classifier' not in st.session_state or not os.path.exists(self.model_filename):
            st.session_state['classifier'] = FuzzyMinMaxClassifier()
            st.info("New model created. Train to improve.")
        else:
            st.session_state['classifier'] = joblib.load(self.model_filename)
            st.success("Model loaded successfully.")

    def save_model(self):
        joblib.dump(st.session_state['classifier'], self.model_filename)
        st.success("Model saved successfully.")

    def load_and_preprocess_image(self, uploaded_file):
        img = Image.open(uploaded_file).convert('L').resize((100, 100))
        return np.array(img).flatten() / 255.0  # Normalize to [0,1]

    def update_and_train(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        st.session_state['classifier'].fit(X_train, y_train)
        self.save_model()
        accuracy = self.evaluate_model(X_test, y_test)
        st.success(f"Model trained. Accuracy: {accuracy:.2f}%")

    def evaluate_model(self, X_test, y_test):
        y_pred = st.session_state['classifier'].predict(X_test)
        return 100 * accuracy_score(y_test, y_pred)

    def classify_and_visualize(self, uploaded_file):
        img_array = self.load_and_preprocess_image(uploaded_file)
        prediction = st.session_state['classifier'].predict([img_array])[0]
        st.write(f'Prediction: {"Tumor" if prediction == 1 else "No Tumor"}')
        if prediction == 1:
            self.visualize_tumor(uploaded_file)

    def visualize_tumor(self, uploaded_file):
        img = Image.open(uploaded_file).convert('L')
        img_array = np.array(img.resize((256, 256)))

        num_clusters = 2
        data = img_array.reshape(-1, 1)

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data.T, num_clusters, 2, error=0.005, maxiter=1000)

        cluster_membership = np.argmax(u, axis=0)
        segmented_image = cluster_membership.reshape(img_array.shape)

        tumor_cluster_index = np.argmin(cntr.sum(axis=1))
        tumor_mask = segmented_image == tumor_cluster_index
        tumor_mask = opening(tumor_mask, disk(2))

        highlighted_img = np.copy(img_array)
        highlighted_img[tumor_mask] = 255

        st.image(highlighted_img, caption='Highlighted Tumor Region', use_column_width=True)

    def run(self):
        st.title("Tumor Detection Application")
        menu = st.sidebar.radio("Menu", ["Load Data", "Train", "Classify Image"])

        if menu == "Load Data":
            uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True)
            if uploaded_files:
                label = st.sidebar.radio("Label for uploaded images", options=[0, 1], index=0)
                features = np.array([self.load_and_preprocess_image(f) for f in uploaded_files])
                labels = np.array([label] * len(features))
                if st.sidebar.button("Update & Train"):
                    self.update_and_train(features, labels)

        elif menu == "Classify Image":
            uploaded_file = st.file_uploader("Upload an image for classification")
            if uploaded_file:
                self.classify_and_visualize(uploaded_file)

if __name__ == "__main__":
    app = TumorDetectionApp()
    app.run()
