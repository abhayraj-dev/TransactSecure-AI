from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os
import random


app = FastAPI(title="FraudShield AI", version="1.0")

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
def load_ai_artifacts():
    global model, scaler, threshold, test_data
    print("[System] Waking up AI and loading artifacts...")
    
    
    base_dir = "ml_artifacts"
    
    try:
        model = load_model(os.path.join(base_dir, "fraud_detector_ae.h5"), compile=False)
        
        with open(os.path.join(base_dir, "amount_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
            
        with open(os.path.join(base_dir, "optimal_threshold.txt"), "r") as f:
            original_threshold = float(f.read())
            threshold = original_threshold + 1.0 # The safety buffer
            
        # Load the test dataset for the Live Stream Simulator
        test_data = pd.read_csv(os.path.join(base_dir, "test_samples.csv"))
        
        print(f"[System] AI Ready. Operational Threshold: {threshold:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {e}")


@app.get("/stream")
def get_random_transaction():
    """Acts as a mock live payment gateway, sending one random transaction."""
    if test_data is None:
        raise HTTPException(status_code=500, detail="Test data not loaded.")
    

    row = test_data.sample(1).iloc[0]
    
    features = row[['V' + str(i) for i in range(1, 29)] + ['Amount']].tolist()
    
   
    actual_class = int(row['Class']) if 'Class' in row else 0
    
    return {"features": features, "actual_class": actual_class}

# The XAI Analysis Endpoint
@app.post("/analyze")
def analyze_transaction(data: TransactionData):
    """The core engine: Predicts fraud and explains WHY."""
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
        
        # Explainable AI (XAI) Engine
        # Calculate exactly WHICH features caused the anomaly
        feature_errors = np.power(model_input[0] - reconstruction[0], 2)
        

        error_breakdown = [
            {"feature": name, "error": float(err)} 
            for name, err in zip(feature_names, feature_errors)
        ]
        error_breakdown.sort(key=lambda x: x["error"], reverse=True)
        
        return {
            "status": "success",
            "is_fraud": is_fraud,
            "risk_score": round(mse, 4),
            "threshold": round(threshold, 4),
            "transaction_amount": round(raw_amount, 2),
            "top_anomalies": error_breakdown[:3] # Send only the top 3 culprits
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))