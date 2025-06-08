import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# from sklearn.neural_network import MLPClassifier # Example for ANN

# Assuming embedding_utils and other necessary modules are accessible
# from .embedding_utils import get_embedding # Or load pre-generated embeddings
# from .data_ingestion import load_resumes_from_directory, parse_resume_text # For getting text
# from .resume_parser import parse_resume_text # If using raw text

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_match_classifier.joblib")

def create_features(text1_embedding, text2_embedding, other_features=None):
    """
    Creates feature vector from two embeddings and optional other features.
    Example: concatenate embeddings.
    """
    # Simple concatenation
    features = np.concatenate((text1_embedding, text2_embedding))

    # Example: element-wise product
    # product = np.multiply(text1_embedding, text2_embedding)
    # features = np.concatenate((text1_embedding, text2_embedding, product))

    if other_features: # e.g., skill_overlap_score, location_match (as numbers)
        features = np.concatenate((features, np.array(other_features)))
    return features

def load_labeled_data(filepath="labeled_matches.csv"):
    """
    Loads labeled data.
    CSV should have columns like: resume_text_or_id, job_desc_or_id, label (0 or 1),
                                [optional: resume_embedding_path, job_embedding_path]
    This is a placeholder - data loading will depend heavily on actual data format.
    """
    if not os.path.exists(filepath):
        print(f"Error: Labeled data file {filepath} not found. Cannot train model.")
        # For now, create a dummy one for demonstration if it doesn't exist
        print("Creating a dummy labeled_matches.csv for demonstration purposes.")
        dummy_data = {
            'resume_id': ['resume1', 'resume2', 'resume3', 'resume4', 'resume1', 'resume2'],
            'job_id': ['job1', 'job1', 'job2', 'job2', 'job3', 'job3'],
            # Dummy embeddings (these would typically be paths or actual embedding vectors)
            # For simplicity, let's assume we have a way to get these based on IDs
            # Or that the CSV contains paths to pre-computed embeddings
            'label': [1, 0, 0, 1, 0, 1] # 1 for match, 0 for no-match
        }
        # Add dummy feature columns that would normally come from embeddings or other calcs
        # Embedding dim for all-MiniLM-L6-v2 is 384. Features = 2 * 384 = 768
        dummy_data['feature1'] = np.random.rand(len(dummy_data['label']))
        dummy_data['feature2'] = np.random.rand(len(dummy_data['label']))
        # ... up to feature768 (or however many features)

        # In a real scenario, you'd have features derived from embeddings.
        # For this dummy script, we'll just use a few random features.
        # Let's assume X will be created from these dummy features later.

        df = pd.DataFrame(dummy_data)
        df.to_csv(filepath, index=False)
        print(f"Dummy {filepath} created. Please replace with actual labeled data.")
        return df

    return pd.read_csv(filepath)

def train_classifier():
    print("Starting model training process...")
    os.makedirs(MODEL_DIR, exist_ok=True) # Ensure MODEL_DIR relative to script execution

    # 1. Load Labeled Data
    # This data needs to be prepared: (resume_embedding, job_embedding, label)
    # For now, let's assume a CSV with pre-calculated features or paths to embeddings
    # The script is inside resume_matcher, so labeled_matches.csv should be in parent dir (project root)
    labeled_df = load_labeled_data(filepath=os.path.join("..", "labeled_matches.csv"))

    if labeled_df.empty:
        print("No labeled data loaded. Aborting training.")
        return

    # 2. Feature Engineering
    # This is the most critical part. For this placeholder, we'll assume
    # 'labeled_df' contains columns that can be directly used as features,
    # or columns from which features can be easily derived.
    # Example: if embeddings are pre-loaded or computed on the fly based on IDs.

    # Let's assume 'labeled_df' has columns 'resume_id', 'job_id', 'label'
    # And we have a way to get embeddings for these IDs.
    # For this placeholder, let's simulate feature creation.
    # In reality, you'd use generate_resume_embeddings, generate_job_description_embedding,
    # then create_features.

    # Placeholder: Assume features X and labels y are somehow derived from labeled_df
    # For example, if 'feature1', 'feature2' were in the CSV:
    # feature_cols = [col for col in labeled_df.columns if 'feature' in col]
    # if not feature_cols:
    #     print("No feature columns found in labeled_df. Cannot train.")
    #     # Create dummy features if none are found, for the script to run
    #     num_samples = len(labeled_df)
    #     X = np.random.rand(num_samples, 10) # Dummy: N samples, 10 features
    # else:
    #    X = labeled_df[feature_cols].values

    # Simpler dummy feature creation for this placeholder script:
    num_samples = len(labeled_df)
    if num_samples < 2: # Need at least 2 samples for train_test_split
        print("Not enough samples in labeled data to train. Aborting.")
        return

    X = np.random.rand(num_samples, 10) # Dummy: N samples, 10 features. Replace with actual features.
    y = labeled_df['label'].values

    if len(X) != len(y):
        print("Mismatch between feature count and label count. Aborting.")
        return

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if np.sum(y) > 1 and np.sum(y) < len(y) -1 and len(np.unique(y)) > 1 else None)


    # 4. Train Model (SVM example)
    print("Training SVM classifier...")
    # classifier = SVC(kernel='linear', probability=True, random_state=42) # Linear SVM
    classifier = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced') # RBF SVM, handles imbalanced classes
    # To use ANN:
    # classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

    try:
        classifier.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # 5. Evaluate Model
    print("\nModel Evaluation:")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 6. Save Trained Model
    # MODEL_PATH is "models/svm_match_classifier.joblib"
    # This path is relative to where the script is run.
    # If script is run from project root: resume_matcher/models/svm_match_classifier.joblib
    # If script is run from resume_matcher dir: models/svm_match_classifier.joblib
    # The MODEL_DIR = "models" and MODEL_PATH assumes script is run from resume_matcher/

    # Correct path to save model inside resume_matcher/models/
    actual_model_save_path = MODEL_PATH
    if not os.path.dirname(actual_model_save_path): # if MODEL_PATH is just a filename
        actual_model_save_path = os.path.join(MODEL_DIR, MODEL_PATH)

    try:
        joblib.dump(classifier, actual_model_save_path)
        print(f"Trained model saved to: {actual_model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == '__main__':
    # This script would be run manually or as part of a pipeline.
    # Example: python resume_matcher/train_model.py (run from project root)
    # To make it runnable from project root (python resume_matcher/train_model.py)
    # or from resume_matcher directory (python train_model.py),
    # paths need to be handled carefully.

    # MODEL_DIR should be relative to this script file.
    script_dir = os.path.dirname(__file__)
    MODEL_DIR = os.path.join(script_dir, "models") # Ensures models is subdir of script location
    MODEL_PATH = os.path.join(MODEL_DIR, "svm_match_classifier.joblib")


    # Path to labeled_matches.csv (assuming it's in project root, i.e., parent of script_dir)
    project_root = os.path.dirname(script_dir)
    labeled_csv_path = os.path.join(project_root, "labeled_matches.csv")

    if not os.path.exists(labeled_csv_path):
        print(f"Creating a dummy {labeled_csv_path} for the training script to run.")
        dummy_df_main = pd.DataFrame({
            'resume_id': [f'res{i}' for i in range(20)],
            'job_id': [f'job{j}' for i in range(10) for j in range(2)], # 20 entries
            'label': np.random.randint(0, 2, 20)
        })
        dummy_df_main.to_csv(labeled_csv_path, index=False)
        print(f"Dummy {labeled_csv_path} created. Please replace with actual data.")

    # Adjust load_labeled_data to use this path
    def load_labeled_data_modified(filepath=labeled_csv_path):
        if not os.path.exists(filepath):
            print(f"Error: Labeled data file {filepath} not found. Cannot train model.")
            # Fallback dummy creation logic from original load_labeled_data
            # This part is complex to replicate here and might be redundant if outer check works.
            return pd.DataFrame() # Return empty if not found after check
        return pd.read_csv(filepath)

    # Overwrite global load_labeled_data for this run if needed, or pass path to train_classifier
    # For simplicity, train_classifier will be modified to accept the path.

    # Redefine train_classifier slightly to accept data path
    def train_classifier_modified(data_path, model_save_path):
        print("Starting model training process (modified)...")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        labeled_df = load_labeled_data_modified(data_path)
        if labeled_df.empty or 'label' not in labeled_df.columns:
            print("No labeled data with 'label' column loaded. Aborting training.")
            return

        num_samples = len(labeled_df)
        if num_samples < 2:
            print("Not enough samples in labeled data to train. Aborting.")
            return

        X = np.random.rand(num_samples, 10)
        y = labeled_df['label'].values

        if len(X) != len(y):
            print("Mismatch between feature count and label count. Aborting.")
            return

        # Ensure at least two classes for stratify if y has mixed values
        unique_labels, counts = np.unique(y, return_counts=True)
        can_stratify = len(unique_labels) > 1 and all(c >= 2 for c in counts)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None)

        print("Training SVM classifier...")
        classifier = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')

        try:
            classifier.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during model training: {e}")
            return

        print("\nModel Evaluation:")
        try:
            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))
        except Exception as e:
            print(f"Error during model evaluation: {e}")


        try:
            joblib.dump(classifier, model_save_path)
            print(f"Trained model saved to: {model_save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    train_classifier_modified(data_path=labeled_csv_path, model_save_path=MODEL_PATH)
    print("\nTo use the trained model, load it in matcher.py and integrate its predictions.")
