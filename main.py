from fastapi import FastAPI, Request
import pandas as pd
import joblib
import os

app = FastAPI()

# Load trained model (you can change this to xgb or another)
MODEL_PATH = os.getenv("MODEL_PATH", "data/models/xgb.model")
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Alea Sports Betting AI Model is live."}

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()

    # Expecting a dict like: {"feature1": value, "feature2": value, ...}
    df = pd.DataFrame([payload])
    prediction = model.predict(df)[0]

    return {"prediction": float(prediction)}
