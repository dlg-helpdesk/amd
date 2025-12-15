import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from feature_extractor import extract_features
import pickle

# ---------------------------
# Settings / Parameters
# ---------------------------
DATASET_DIR = "dataset"          # Main dataset folder containing batch subfolders
INCLUDE_SILENT = True            # Include silent recordings if present
MODEL_OUTPUT = "model/audio_model.pkl"
ENCODER_OUTPUT = "model/label_encoder.pkl"


def load_dataset_from_batch(batch_dir, include_silent=False):
    X, y = [], []

    # Human recordings
    human_dir = os.path.join(batch_dir, 'human')
    if os.path.exists(human_dir):
        for file in os.listdir(human_dir):
            if file.endswith('.wav'):
                X.append(extract_features(os.path.join(human_dir, file)))
                y.append('human')

    # Machine recordings
    machine_dir = os.path.join(batch_dir, 'machine')
    if os.path.exists(machine_dir):
        for file in os.listdir(machine_dir):
            if file.endswith('.wav'):
                X.append(extract_features(os.path.join(machine_dir, file)))
                y.append('machine')

    # Silent recordings (optional)
    if include_silent:
        silent_dir = os.path.join(batch_dir, 'silent')
        if os.path.exists(silent_dir):
            for file in os.listdir(silent_dir):
                if file.endswith('.wav'):
                    X.append(extract_features(os.path.join(silent_dir, file)))
                    y.append('silent')

    return X, y

# ---------------------------
# Training function
# ---------------------------
def main():
    X_total, y_total = [], []

    # Loop through all batch folders
    for batch_name in os.listdir(DATASET_DIR):
        batch_path = os.path.join(DATASET_DIR, batch_name)
        if os.path.isdir(batch_path):
            print(f"Loading batch: {batch_name}")
            X_batch, y_batch = load_dataset_from_batch(batch_path, INCLUDE_SILENT)
            X_total.extend(X_batch)
            y_total.extend(y_batch)

    # Convert to numpy arrays
    X_total = np.array(X_total)
    y_total = np.array(y_total)

    print(f"Total samples loaded: {len(y_total)}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_total)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    acc = clf.score(X_test, y_test)
    print(f"Validation Accuracy: {acc*100:.2f}%")

    # Save model and encoder
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(clf, f)
    with open(ENCODER_OUTPUT, 'wb') as f:
        pickle.dump(le, f)

    print("Training complete. Model saved to:", MODEL_OUTPUT)
    print("Label encoder saved to:", ENCODER_OUTPUT)

# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    main()
