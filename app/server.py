# app/server.py
import os, time, joblib, numpy as np
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
model = joblib.load(MODEL_PATH)

REQUEST_COUNT = Counter("model_request_count", "Total number of prediction requests")
REQUEST_LATENCY = Histogram("model_prediction_latency_seconds", "Prediction latency in seconds")
PREDICTION_EXCEPTIONS = Counter("model_prediction_exceptions_total", "Exceptions during prediction")
FRAUD_PROB = Gauge("model_fraud_probability", "Last predicted fraud probability (0-1)")

app = FastAPI(title="Fraud Detection API with Prometheus")

class Input(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: Input):
    REQUEST_COUNT.inc()
    start = time.time()
    try:
        x = np.array(inp.features).reshape(1, -1)
        probs = model.predict_proba(x)[0]
        pred = int(model.predict(x)[0])
        prob = float(probs[1])
        FRAUD_PROB.set(prob)
        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)
        return {"fraud_prediction": pred, "fraud_probability": prob, "latency": latency}
    except Exception as e:
        PREDICTION_EXCEPTIONS.inc()
        raise

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
