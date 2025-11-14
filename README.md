# 🚀 End-to-End MLOps Real-Time Financial Fraud Detection System

A fully production-ready MLOps pipeline for real-time financial fraud detection featuring:

⦁	DVC for data versioning

⦁	MLflow for experiment tracking & model registry

⦁	FastAPI inference service

⦁	Docker containerization

⦁	Render cloud deployment

⦁	GitHub Actions CI/CD automation

⦁	Prometheus + Grafana for monitoring


# 📸 System Architecture

fraud-detection-mlops/
│
├── data/
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── inference.py
│ └── app/server.py
│
├── models/
├── dvc.yaml
├── Dockerfile
├── prometheus.yml
├── docker-compose.yaml
└── .github/workflows/ci.yml

# 🏗️ Tech Stack

| Layer                | Tools Used          |
| -------------------- | ------------------- |
| **ML Lifecycle**     | DVC, MLflow         |
| **Model Serving**    | FastAPI, Uvicorn    |
| **CI/CD**            | GitHub Actions      |
| **Containerization** | Docker              |
| **Deployment**       | Render Cloud        |
| **Monitoring**       | Prometheus, Grafana |

# ⚙️ Setup Instructions (Local)

# 1️⃣ Clone the repository

⦁	git clone https://github.com/keerthanabollepally/fraud-detection-mlops.git
⦁	cd fraud-detection-mlops

# 2️⃣ Install dependencies


⦁	pip install -r requirements.txt

# 3️⃣ Pull data using DVC
⦁	dvc pull

# 4️⃣ Run preprocessing

⦁	python src/preprocess.py

# 5️⃣ Train the model

⦁	python src/train.py

# 6️⃣ Run FastAPI locally

⦁	uvicorn src.app.server:app --reload
# ☁️ Deployment (Render)

Your API runs live at:

https://<your-render-service>.onrender.com/predict

# 📡 API Usage
🔹 POST /predict
Example Request:

⦁	{"features": [181, 100.50, 5000.00, 4900.00, 20000.00, 20000.00, 0, 4.61, 0, 0, 1, 0]}
Example Response : 
⦁	{ "fraud_probability": 0.0021, "is_fraud": 0}
<img width="1284" height="887" alt="Screenshot 2025-11-13 123725" src="https://github.com/user-attachments/assets/fff66e7f-cf4f-4d22-8878-715402268685" />

<img width="1252" height="270" alt="Screenshot 2025-11-13 123740" src="https://github.com/user-attachments/assets/f33c0c38-e4c2-431f-a4c5-3a922e522e3e" />

# 🔄 CI/CD Pipeline (GitHub Actions)

⦁	Installs dependencies

⦁	Runs tests

⦁	Builds Docker image

⦁	Pushes to GitHub Container Registry

⦁	Deploys automatically
<img width="1882" height="761" alt="Screenshot 2025-11-14 005611" src="https://github.com/user-attachments/assets/c8f63d18-3bc7-4ddd-8197-8f2f72ab9bb9" />


# 📊 Monitoring with Prometheus + Grafana

⦁	Monitored metrics:

⦁	Request count

⦁	Prediction latency

⦁	Fraud probability drift

⦁	Errors per second
<img width="1630" height="905" alt="Screenshot 2025-11-13 230115" src="https://github.com/user-attachments/assets/e4690cf9-bdc7-4555-8f01-80801ad98b58" />

<img width="1898" height="921" alt="Screenshot 2025-11-13 230127" src="https://github.com/user-attachments/assets/35e8bb26-4dc9-4da2-a3ed-4d7815170bc5" />


# ⭐ Key Achievements

<img width="1872" height="924" alt="Screenshot 2025-11-14 010945" src="https://github.com/user-attachments/assets/60f7a262-b866-4862-b6d0-5bc940177600" />

⦁	Production-ready, cloud-hosted ML service

⦁	Fully automated data → model → deploy pipeline

⦁	Real-time monitoring & logs

⦁	Enterprise-level workflow using modern MLOps stack
