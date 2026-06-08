import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import pickle
import os

DATA_FILE = "/kaggle/input/datasets/abhayraj0325/creditcard/creditcard.csv"
MODEL_FILE = "fraud_detector_ae.h5"
SCALER_FILE = "amount_scaler.pkl"
THRESHOLD_FILE = "optimal_threshold.txt"
TEST_DATA_FILE = "test_samples.csv"

tf.random.set_seed(42)
np.random.seed(42)

def train_and_save_model():
    print(f"[Trainer] Loading dataset from '{DATA_FILE}'...")
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: '{DATA_FILE}' not found.")
        return

    #Preserve Original Row IDs
    data['Original_ID'] = data.index

    data = data.drop(['Time'], axis=1)

    #Split Data
    normal_df = data[data['Class'] == 0]
    fraud_df = data[data['Class'] == 1]

    # Split Normal Data (80% Train, 20% Test)
    train_df, test_normal_df = train_test_split(normal_df, test_size=0.2, random_state=42)

    # Save the Mixed Test Set for the Demo
    print("[Trainer] Creating test dataset for predict.py...")
    # We combine the 20% unseen Normal data + ALL Fraud data
    test_dataset = pd.concat([test_normal_df, fraud_df])


    test_dataset.to_csv(TEST_DATA_FILE, index=False)
    print(f"[Trainer] Saved {len(test_dataset)} samples (Normal & Fraud) to '{TEST_DATA_FILE}'")

    # Preprocessing for Training ---
    print("[Trainer] Preprocessing training data...")

    
    scaler = StandardScaler()

    train_df['Amount'] = scaler.fit_transform(train_df['Amount'].values.reshape(-1, 1))

    test_normal_df['Amount'] = scaler.transform(test_normal_df['Amount'].values.reshape(-1, 1))
    fraud_df['Amount'] = scaler.transform(fraud_df['Amount'].values.reshape(-1, 1))

    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)

    # Prepare inputs for Keras (Drop Class and Original_ID)
    # The model only wants the 29 numeric features
    cols_to_drop = ['Class', 'Original_ID']
    X_train = train_df.drop(cols_to_drop, axis=1).values
    X_test_normal = test_normal_df.drop(cols_to_drop, axis=1).values
    X_fraud = fraud_df.drop(cols_to_drop, axis=1).values


    X_test_final = np.concatenate([X_test_normal, X_fraud])
    # Create labels (0 for normal, 1 for fraud)
    y_test_final = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

    # Build & Train Model
    print(f"[Trainer] Training autoencoder on {len(X_train)} normal samples...")
    input_dim = X_train.shape[1] # Should be 29
    encoding_dim = 14

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(encoding_dim, activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='linear')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Early Stopping to prevent overfitting
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopper],
        verbose=1
    )

    autoencoder.save(MODEL_FILE)

    # Calculate Optimal Threshold 
    print("[Trainer] Calculating optimal threshold...")
    reconstructions = autoencoder.predict(X_test_final)
    mse = np.mean(np.power(X_test_final - reconstructions, 2), axis=1)

    precision, recall, thresholds = precision_recall_curve(y_test_final, mse)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    print(f"[Trainer] Optimal Threshold: {optimal_threshold:.6f}")
    with open(THRESHOLD_FILE, 'w') as f:
        f.write(str(optimal_threshold))

    print("[Trainer] Done. We can now run 'predict.py'.")

if __name__ == "__main__":
    if os.path.exists(DATA_FILE):
        train_and_save_model()
    else:
        print(f"Please provide {DATA_FILE}")