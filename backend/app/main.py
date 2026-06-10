from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pickle
import os
import random
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv 


load_dotenv() 

app = FastAPI(title="TransactSecure AI", version="1.0")

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("[CRITICAL] MONGO_URI is missing from the .env file!")

client = None
db = None
alerts_collection = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionData(BaseModel):
    features: List[float]

model = None
scaler = None
threshold = None
test_data = None
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@app.on_event("startup")
def startup_event():
    global model, scaler, threshold, test_data, client, db, alerts_collection
    print("[System] Waking up AI and loading artifacts...")
    
   
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Test connection
        db = client.transactsecure
        alerts_collection = db.fraud_alerts
        print("[System] Database connected successfully.")
    except Exception as e:
        print(f"[WARNING] Database connection failed: {e}")

    
    base_dir = "ml_artifacts"
    try:
        print("[System] Hardcoding AI architecture to bypass Keras bugs...")
        # 1. Manually build the exact same Autoencoder architecture
        input_layer = Input(shape=(29,))
        encoder = Dense(14, activation="tanh")(input_layer)
        encoder = Dense(7, activation="relu")(encoder)
        decoder = Dense(14, activation="tanh")(encoder)
        decoder = Dense(29, activation="linear")(decoder)
        model = Model(inputs=input_layer, outputs=decoder)

        print("[System] Injecting raw mathematical weights...")
        # 2. Inject the weights matrix directly
        weights_path = os.path.join(base_dir, "model_weights.weights.h5")
        model.load_weights(weights_path)
        with open(os.path.join(base_dir, "amount_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(base_dir, "optimal_threshold.txt"), "r") as f:
            original_threshold = float(f.read())
            threshold = original_threshold + 1.0 
        test_data = pd.read_csv(os.path.join(base_dir, "test_samples.csv"))
        print(f"[System] AI Ready. Operational Threshold: {threshold:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {e}")

@app.get("/")
def read_root():
    """Health check for Render deployment"""
    return {"status": "TransactSecure API is running successfully."}

# Live Stream Endpoints
@app.get("/stream")
def get_random_transaction():
    if test_data is None:
        raise HTTPException(status_code=500, detail="Test data not loaded.")
    row = test_data.sample(1).iloc[0]
    features = row[['V' + str(i) for i in range(1, 29)] + ['Amount']].tolist()
    return {"features": features, "actual_class": int(row.get('Class', 0))}

@app.get("/stream/fraud")
def get_guaranteed_fraud():
    try:
        fraud_cases = test_data[test_data['Class'] == 1]
        row = fraud_cases.sample(1).iloc[0]
        features = row[['V' + str(i) for i in range(1, 29)] + ['Amount']].tolist()
        return {"features": features, "actual_class": 1}
    except:
        synthetic_features = [random.uniform(5.0, 15.0) for _ in range(28)]
        synthetic_features.append(4500.00)
        return {"features": synthetic_features, "actual_class": 1}

# Database History
@app.get("/history")
def get_alert_history():
    """Fetches the 20 most recent fraud alerts from MongoDB."""
    if alerts_collection is None:
        return []
    # Fetch records, sort by newest first, ignore the MongoDB object ID
    docs = list(alerts_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
    return docs

# The XAI Analysis Engine
@app.post("/analyze")
def analyze_transaction(data: TransactionData):
    if len(data.features) != 29:
        raise HTTPException(status_code=400, detail="Need exactly 29 features.")
    
    try:
        features_array = np.array(data.features).astype('float64')
        raw_amount = features_array[-1]
        
        scaled_amount = scaler.transform([[raw_amount]])[0][0]
        model_input = features_array.copy()
        model_input[-1] = scaled_amount
        model_input = model_input.reshape(1, -1)
        
        reconstruction = model.predict(model_input, verbose=0)
        mse = float(np.mean(np.power(model_input - reconstruction, 2), axis=1)[0])
        is_fraud = bool(mse > threshold)
        
        feature_errors = np.power(model_input[0] - reconstruction[0], 2)
        error_breakdown = [
            {"feature": name, "error": float(err)} 
            for name, err in zip(feature_names, feature_errors)
        ]
        error_breakdown.sort(key=lambda x: x["error"], reverse=True)
        top_anomalies = error_breakdown[:3]

        # DATABASE INJECTION: If fraud is detected, permanently log it
        if is_fraud and alerts_collection is not None:
            alert_doc = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_amount": round(raw_amount, 2),
                "risk_score": round(mse, 4),
                "top_anomalies": top_anomalies
            }
            alerts_collection.insert_one(alert_doc)
        
        return {
            "status": "success",
            "is_fraud": is_fraud,
            "risk_score": round(mse, 4),
            "threshold": round(threshold, 4),
            "transaction_amount": round(raw_amount, 2),
            "top_anomalies": top_anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))