import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import combinations
from SVM import SVM
import cv2
import os

class OneVsOneSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.classifiers = {}  # Dictionary to store binary classifiers
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        # Create binary classifiers for each pair of classes
        for class1, class2 in combinations(self.classes, 2):
            # Get indices for the current pair of classes
            mask = np.isin(y, [class1, class2])
            X_subset = X[mask]
            y_subset = y[mask]
            
            # Convert to binary labels (-1, 1)
            y_binary = np.where(y_subset == class1, -1, 1)
            
            # Train binary SVM for this pair
            svm = SVM(C=self.C)
            svm.fit(X_subset, y_binary)
            
            # Store the classifier
            self.classifiers[(class1, class2)] = svm
    
    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))
        
        # Get predictions from each binary classifier
        for (class1, class2), svm in self.classifiers.items():
            predictions = svm.predict(X)
            
            # Convert predictions to votes
            class1_idx = np.where(self.classes == class1)[0][0]
            class2_idx = np.where(self.classes == class2)[0][0]
            
            for i, pred in enumerate(predictions):
                if pred < 0:
                    votes[i, class1_idx] += 1
                else:
                    votes[i, class2_idx] += 1
        
        # Return the class with the most votes for each sample
        return self.classes[np.argmax(votes, axis=1)]

def load_and_preprocess_data():
    # Define paths
    train_dir = "Data/Training"
    
    # Initialize lists to store data
    images = []
    labels = []
    
    # Define class mapping
    class_mapping = {
        'no_tumor': 0,
        'glioma_tumor': 1,
        'meningioma_tumor': 2,
        'pituitary_tumor': 3
    }
    
    # Load training data
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            class_label = class_mapping[class_name.replace('augmented_', '')]
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # Read and preprocess image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Apply Gaussian blur to reduce noise
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    
                    # Apply histogram equalization to enhance contrast
                    img = cv2.equalizeHist(img)
                    
                    # Resize to a fixed size
                    img = cv2.resize(img, (28, 28))
                    
                    # Apply edge detection
                    edges = cv2.Canny(img, 50, 150)
                    
                    # Combine original and edge features
                    img = np.concatenate([img.flatten(), edges.flatten()])
                    
                    # Normalize pixel values
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(class_label)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Training OVO-SVM classifier...")
    # Create and train the OVO-SVM classifier
    ovo_svm = OneVsOneSVM(C=50.0)
    ovo_svm.fit(X_train, y_train)
    
    print("Making predictions...")
    # Make predictions
    y_pred = ovo_svm.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")