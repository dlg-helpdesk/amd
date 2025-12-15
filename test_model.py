import os
import pickle
from collections import defaultdict
from feature_extractor import extract_features

# ---------------------------
# Settings / Parameters
# ---------------------------
MODEL_PATH = "model/audio_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
TEST_DATASET_DIR = "test_dataset"  # Folder with subfolders: human, machine, silent


# ---------------------------
# Main tester
# ---------------------------
def main():
    # Load model and label encoder
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    # Initialize counters
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_files = 0

    # Loop through each subfolder
    for class_folder in os.listdir(TEST_DATASET_DIR):
        folder_path = os.path.join(TEST_DATASET_DIR, class_folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path).reshape(1, -1)

            # Prediction
            pred_encoded = model.predict(features)[0]
            pred_label = le.inverse_transform([pred_encoded])[0]

            # Update counters
            stats[class_folder]['total'] += 1
            total_files += 1
            if pred_label.lower() == class_folder.lower():
                stats[class_folder]['correct'] += 1
                total_correct += 1

    # Print per-class accuracy
    print("Per-class accuracy:")
    for cls, vals in stats.items():
        correct = vals['correct']
        total = vals['total']
        acc = (correct / total * 100) if total > 0 else 0
        print(f"{cls.capitalize()}: {correct} correct / {total} total --> {acc:.2f}%")

    # Overall accuracy
    overall_acc = (total_correct / total_files * 100) if total_files > 0 else 0
    print(f"\nOverall accuracy: {total_correct} correct / {total_files} total --> {overall_acc:.2f}%")

# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    main()
