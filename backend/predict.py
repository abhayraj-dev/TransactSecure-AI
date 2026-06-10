import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


TEST_DATA_FILE = "test_samples.csv"
MODEL_FILE = "fraud_detector_ae.keras"
SCALER_FILE = "amount_scaler.pkl"
THRESHOLD_FILE = "optimal_threshold.txt"

def load_artifacts():
    print("[System] Loading model and settings...")
    if not all([os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, THRESHOLD_FILE]]):
        print(f"[Error] Missing files. Run 'train.py' first.")
        return None, None, None

    try:
        model = load_model(MODEL_FILE, compile=False)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        with open(THRESHOLD_FILE, 'r') as f:
            original_threshold = float(f.read())
            # --- THE SAFETY FIX ---
            # We ADD a buffer (e.g., +2.0 or multiply by 1.5).
            # This reduces False Alarms (Normal -> Fraud) drastically.
            threshold = original_threshold + 1.0
        return model, scaler, threshold
    except Exception as e:
        print(f"[Error] {e}")
        return None, None, None

def predict_transaction(model, scaler, threshold, features, row_id=None):
    features = np.array(features).astype('float64')

    raw_amount = features[-1]
    scaled_amount = scaler.transform([[raw_amount]])[0][0]

    model_input = features.copy()
    model_input[-1] = scaled_amount
    model_input = model_input.reshape(1, -1)

    reconstruction = model.predict(model_input, verbose=0)
    mse = np.mean(np.power(model_input - reconstruction, 2), axis=1)[0]


    print(f"\n------------------------------------------------")
    if row_id is not None:
        print(f" VERIFICATION ID:   Row #{int(row_id)}")

    print(f"Transaction Amount:   ${raw_amount:.2f}")
    print(f"Reconstruction Error: {mse:.4f}")
    print(f"Threshold Limit:      {threshold:.4f}")
    print(f"------------------------------------------------")

    if mse > threshold:
        print(f"🔴 RESULT: FRAUD DETECTED!")
    else:
        print(f"🟢 RESULT: Transaction Normal")
    print(f"------------------------------------------------")

def evaluate_performance(model, scaler, threshold):
    print("\n[System] Running full evaluation on 'test_samples.csv'...")
    try:
        df = pd.read_csv(TEST_DATA_FILE)
    except FileNotFoundError:
        print(f"[Error] {TEST_DATA_FILE} not found.")
        return

    y_true = df['Class'].values
    X = df.drop(['Class', 'Original_ID'], axis=1).copy()
    X['Amount'] = scaler.transform(X['Amount'].values.reshape(-1, 1))

    print("[System] Predicting... (this might take a moment)")
    reconstructions = model.predict(X.values, verbose=0)
    mse = np.mean(np.power(X.values - reconstructions, 2), axis=1)

    y_pred = [1 if error > threshold else 0 for error in mse]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print(f" MODEL EVALUATION REPORT")
    print("="*40)
    print(f" Accuracy:  {acc * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    print("-" * 40)
    print("Confusion Matrix:")
    print(f" [ True Normal  | False Alarm ]")
    print(f" [ Missed Fraud | Caught Fraud]")
    print(cm)
    print("="*40)

def main():
    model, scaler, threshold = load_artifacts()
    if model is None: return

    print(f"[System] Loading test data from '{TEST_DATA_FILE}'...")
    try:
        test_df = pd.read_csv(TEST_DATA_FILE)
        normal_pool = test_df[test_df['Class'] == 0]
        fraud_pool = test_df[test_df['Class'] == 1]
    except FileNotFoundError:
        print(f"[Error] '{TEST_DATA_FILE}' not found. Re-run 'train.py'.")
        return

    while True:
        print("\n=== FRAUD DETECTION MENU ===")
        print("[N] Test Random NORMAL Transaction")
        print("[F] Test Random FRAUD Transaction")
        print("[S] Test SYNTHETIC Fraud (Mathematically Guaranteed)")
        print("[E] Evaluate Full Model")
        print("[Q] Quit")

        choice = input("Select Option: ").upper().strip()

        if choice == 'Q':
            break

        elif choice == 'N':
            if not normal_pool.empty:
                row = normal_pool.sample(1)
                features = row.drop(['Original_ID', 'Class'], axis=1).values[0]
                row_id = row['Original_ID'].values[0]
                print("\n[Action] Testing a random NORMAL transaction...")
                predict_transaction(model, scaler, threshold, features, row_id)

        elif choice == 'F':
            if not fraud_pool.empty:
                row = fraud_pool.sample(1)
                features = row.drop(['Original_ID', 'Class'], axis=1).values[0]
                row_id = row['Original_ID'].values[0]
                print("\n[Action] Testing a random FRAUD transaction...")
                predict_transaction(model, scaler, threshold, features, row_id)

        elif choice == 'S':
            # Create a FAKE anomaly by adding noise to a normal transaction
            if not normal_pool.empty:
                row = normal_pool.sample(1)
                real_features = row.drop(['Original_ID', 'Class'], axis=1).values[0]

                # Add random noise to make it mathematically 'weird'
                noise = np.random.uniform(1.5, 5.0, size=real_features.shape)
                fake_features = real_features * noise
                fake_features[-1] = 5000.00 # Set a high amount

                print("\n[Action] Generating a SYNTHETIC Anomaly (Fake Data)...")
                print("(This proves the model reacts to abnormal math patterns)")
                predict_transaction(model, scaler, threshold, fake_features)

        elif choice == 'E':
            evaluate_performance(model, scaler, threshold)

if __name__ == "__main__":
    main()