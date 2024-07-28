import streamlit as st
import os
import json

from sympy import python
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the annotations


def load_annotations(annotations_folder):
    annotations = {}
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.json'):
            with open(os.path.join(annotations_folder, filename)) as f:
                try:
                    data = json.load(f)
                    annotations[filename] = data
                except json.JSONDecodeError as e:
                    st.error(f"Error decoding JSON for file {filename}: {e}")
    return annotations

# Parse a single annotation file


def parse_annotation(annotation):
    objects = annotation.get('objects', [])
    boxes = []
    labels = []
    for obj in objects:
        if obj.get('geometryType') == 'rectangle':
            label = obj.get('classTitle')
            exterior = obj.get('points', {}).get('exterior', [])
            if len(exterior) == 2:
                xmin, ymin = exterior[0]
                xmax, ymax = exterior[1]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
            else:
                st.error(f"Invalid exterior points in annotation: {exterior}")
        else:
            st.error(
                f"Invalid geometry type in annotation: {obj.get('geometryType')}")
    return boxes, labels

# Extract histogram features from an image region


def extract_histogram(image, mask=None):
    hist = cv2.calcHist([image], [0, 1, 2], mask, [
                        8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Prepare features for training or testing


def prepare_features(images, boxes_list):
    features = []
    for img, boxes in zip(images, boxes_list):
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmin >= 0 and ymin >= 0 and xmax <= img.shape[1] and ymax <= img.shape[0]:
                roi = img[ymin:ymax, xmin:xmax]
                hist = extract_histogram(roi)
                features.append(hist)
            else:
                st.error(f"Invalid bounding box: {box}")
    return features

# Prepare the dataset by loading images and annotations


def prepare_dataset(images_folder, annotations_folder):
    images = []
    boxes_list = []
    labels_list = []
    annotations = load_annotations(annotations_folder)

    for img_filename in os.listdir(images_folder):
        if img_filename.endswith('.jpeg') or img_filename.endswith('.jpg') or img_filename.endswith('.png'):
            img_path = os.path.join(images_folder, img_filename)
            annotation_filename = img_filename.replace('.jpeg', '.json').replace(
                '.jpg', '.json').replace('.png', '.json')
            if annotation_filename in annotations:
                annotation = annotations[annotation_filename]
                boxes, labels = parse_annotation(annotation)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
                    boxes_list.append(boxes)
                    labels_list.append(labels)
                else:
                    st.error(f"Error loading image: {img_path}")
            else:
                st.error(
                    f"Annotation file not found for image: {img_filename}")

    return images, boxes_list, labels_list

# Visualize detections on an image


def visualize_detections(image, detections):
    for box, label in detections:
        xmin, ymin, xmax, ymax = box
        color = (0, 255, 0) if label == 0 else (
            0, 0, 255)  # Green for crop, Red for weed
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, 'crop' if label == 0 else 'weed',
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

# Detect objects in an image


def detect_objects(image, boxes, model):
    results = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        roi = image[ymin:ymax, xmin:xmax]
        hist = extract_histogram(roi)
        hist = hist.reshape(1, -1)
        label = model.predict(hist)
        results.append((box, label[0]))
    return results


# Paths to the images and annotations folders
annotations_folder = 'annotations'
images_folder = 'images'

# Load the dataset
images, boxes_list, labels_list = prepare_dataset(
    images_folder, annotations_folder)

st.title("Weed and Crop Detection")

# Define the number of images for training and testing
num_train = 1000
num_test = 300

# Initialize session state for the model and training status
if 'model' not in st.session_state:
    st.session_state.model = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

# Description of the process
st.markdown("""
### Detection Process
This application detects weeds and crops in images using a machine learning model. The steps involved are:
1. **Data Preparation:** Images and their corresponding annotations are loaded.
2. **Feature Extraction:** Histogram features are extracted from the regions of interest (ROIs) in the images.
3. **Model Training:** A K-Nearest Neighbors classifier is trained using the extracted features and corresponding labels.
4. **Detection:** The trained model is used to predict whether the regions in the test images contain crops or weeds.
""")

# Display example images
st.markdown("### Example Images")
col1, col2 = st.columns(2)
with col1:
    st.image("detectionex/weed.png",
             caption="Weed Detection Example", use_column_width=True)
with col2:
    st.image("detectionex/crop.jpeg",
             caption="Crop Detection Example", use_column_width=True)

# Button to train the model
if not st.session_state.is_trained:
    if st.button('Train Model'):
        if len(images) >= num_train:
            train_images = images[:num_train]
            train_boxes = boxes_list[:num_train]
            train_labels = labels_list[:num_train]

            st.write(f"Using {num_train} images for training.")

            train_features = prepare_features(train_images, train_boxes)
            train_labels_flat = [
                0 if label == 'crop' else 1 for labels in train_labels for label in labels]
            X_train = np.array(train_features)
            y_train = np.array(train_labels_flat)

            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X_train, y_train)

            st.session_state.model = model
            st.session_state.is_trained = True
            st.success("Model trained successfully.")
        else:
            st.error("Not enough images for training. Please check the dataset.")

# Button to test the model and display detections
if st.session_state.is_trained:
    if st.button('Test Images'):
        if len(images) >= num_train + num_test:
            test_images = images[-num_test:]
            test_boxes = boxes_list[-num_test:]
            test_labels = labels_list[-num_test:]

            test_features = prepare_features(test_images, test_boxes)
            test_labels_flat = [
                0 if label == 'crop' else 1 for labels in test_labels for label in labels]
            X_test = np.array(test_features)
            y_test = np.array(test_labels_flat)

            accuracy = st.session_state.model.score(X_test, y_test)
            st.write(f"Accuracy: {accuracy * 100:.2f}%")

            # Display test images with detections in rows of 4
            for i in range(0, len(test_images[:20]), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(test_images[:20]):
                        image = test_images[i + j]
                        detections = detect_objects(
                            image, test_boxes[i + j], st.session_state.model)
                        image_with_detections = visualize_detections(
                            image.copy(), detections)
                        label = 'Crop' if any(
                            d[1] == 0 for d in detections) else 'Weed'
                        cols[j].image(cv2.cvtColor(
                            image_with_detections, cv2.COLOR_BGR2RGB), caption=label, use_column_width=True)
        else:
            st.error(
                "Not enough data to evaluate the model. Please check the dataset.")
