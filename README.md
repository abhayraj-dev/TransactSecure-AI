# 🛡️ TransactSecure AI: Unsupervised Fraud Detection System

TransactSecure AI is a production-ready credit card fraud detection backend. It utilizes an **Unsupervised Autoencoder** neural network to learn the mathematical shape of "normal" transactions and flags anomalies (fraud) based on dynamic reconstruction error thresholds. 

The system is built on **TensorFlow/Keras 3** and served via a high-performance **FastAPI** backend.

## 🔗 Live Deployment
* **Live Web Application:** [https://transactsecure-ai.vercel.app/](https://transactsecure-ai.vercel.app/) *(Frontend UI)*
* **Live API (Render):** [https://transactsecure-ai-backend.onrender.com](https://transactsecure-ai-backend.onrender.com)
* **Interactive API Docs (Swagger UI):** [https://transactsecure-ai-backend.onrender.com/docs](https://transactsecure-ai-backend.onrender.com/docs) *(Use this to test the endpoints directly from your browser)*

---

## 📊 The Dataset
This model is trained on the highly recognized **Credit Card Fraud Detection Dataset** provided by the Machine Learning Group - ULB (Université Libre de Bruxelles).

* **Dataset Link:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Context:** The dataset contains transactions made by European cardholders in September 2013.
* **Challenge:** It is highly imbalanced, with frauds accounting for only **0.172%** of all transactions (492 frauds out of 284,807 transactions).
* **Features:** Due to confidentiality, the input variables are the result of a PCA (Principal Component Analysis) transformation (V1 to V28). The only features not transformed with PCA are `Time` and `Amount`.

---

## 🚀 Key Engineering Features
* **Unsupervised Anomaly Detection:** Trained exclusively on normal transaction data. This allows the model to catch zero-day fraud tactics without relying on heavily imbalanced labeled data.
* **Weights-Only Architecture Injection:** Bypasses notoriously unstable Keras 3 serialization bugs in cloud environments. The AI architecture is hardcoded into the FastAPI server, and strictly injects raw mathematical weights (`.weights.h5`) for 100% stable cloud boots.
* **Dynamic Thresholding:** Automatically calculates the optimal F1-score threshold during training to perfectly balance Recall (catching fraud) and Precision (avoiding false alarms), with an added production safety buffer.

---

## 🛠️ Tech Stack
* **Frontend:** React, Vite, JavaScript (JSX), Tailwind CSS
* **Machine Learning:** TensorFlow 2.19.0, Keras 3, Scikit-Learn, Pandas, Numpy
* **Backend API:** FastAPI, Uvicorn, Python 3.11+
* **Database:** MongoDB (via PyMongo)
* **Deployment:** Vercel (Frontend UI) & Render (Backend API)
---

## 📂 Project Structure

This project is built as a full-stack monorepo, separating the FastAPI machine learning backend from the React frontend.

```text
TransactSecure-AI/
├── backend/                      # FastAPI + Keras 3 Backend (Deployed on Render)
│   ├── app/
│   │   └── main.py               # FastAPI server & hardcoded AI architecture
│   ├── ml_artifacts/             
│   │   ├── model_weights.weights.h5  # The raw neural network math
│   │   ├── amount_scaler.pkl         # StandardScaler for transaction amounts
│   │   └── optimal_threshold.txt     # Dynamic MSE threshold limit
│   ├── requirements.txt          # Pinned production dependencies
|   ├── train.py                  # Autoencoder training script (Kaggle optimized)
|   ├── predict.py                # Local CLI testing & evaluation suite
│   └── .env                      # Environment variables (excluded from Git)
│
├── frontend/                     # React + Vite Frontend (Deployed on Vercel)
│   ├── public/                   # Static public assets
│   ├── src/                      
│   │   ├── assets/               # Application images/icons
│   │   ├── App.jsx               # Main React application component
│   │   ├── index.css             # Global styles
│   │   └── main.jsx              # React DOM entry point
│   ├── package.json              # Frontend dependencies and scripts
│   └── vite.config.js            # Vite bundler configuration
│
└── README.md                     # Master project documentation
```
---

## 💻 Local Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/abhayraj-dev/TransactSecure-AI.git
cd TransactSecure-AI/backend
```
2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
---
## Training the Model (Optional)
The pre-trained weights are already included in /ml_artifacts. If you wish to retrain the model on new data:

1. Download the dataset from Kaggle.

2. Upload train.py and your dataset to a Kaggle Notebook (or local environment with GPU).

3. Ensure the dataset path in train.py points to your creditcard.csv.

4. Run the training script:
```bash
python train.py
```
* Download the resulting model_weights.weights.h5, amount_scaler.pkl, and optimal_threshold.txt and place them in the backend/ml_artifacts/ directory.
---

## 🧪 Local Testing
You can interact with the model locally via the command-line interface to test its accuracy against random data or custom synthetic anomalies.
```bash
python predict.py
```
---
## 🌐 Running the API Server Locally
To start the FastAPI server on your machine:
```bash
cd app
uvicorn main:app --reload
```
The API will be available at http://127.0.0.1:8000. Navigate to http://127.0.0.1:8000/docs to test the API directly using Swagger UI.
---
## 🎨 Running the Frontend Locally
To start the React UI on your machine:
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the Node modules:
```bash
npm install
```
3. Start the Vite development server:
```bash
npm run dev
```
The frontend will typically be available at http://localhost:5173

## 🏗️ Architecture Context & Future Scope

This project is built as a **Proof of Concept (PoC)** to demonstrate the end-to-end integration of unsupervised machine learning models into modern web architectures. While the current iteration utilizes a static CSV dataset for demonstration and evaluation, the FastAPI backend is designed to be easily adapted for real-time data streams.

**Roadmap for Enterprise Scaling:**
* **Real-Time Data Streaming:** Replace the CSV testing pipeline with Apache Kafka or AWS Kinesis to process live transaction streams.
* **Model Retraining Pipeline:** Implement an Airflow DAG to periodically retrain the Autoencoder as new fraud patterns emerge (concept drift).
* **Database Scaling & Feature Store:** While MongoDB is currently utilized for data persistence, an enterprise deployment would introduce an in-memory caching layer (like Redis) as an ultra-low latency Feature Store, allowing the model to fetch user transaction histories in sub-millisecond times during live inference.

