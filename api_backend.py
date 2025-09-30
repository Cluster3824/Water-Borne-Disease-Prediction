from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from generate_report import generate_report
from alert import send_alert

# Load models and encoders once at startup
le_source = joblib.load('models/le_source.pkl')
le_season = joblib.load('models/le_season.pkl')
le_gender = joblib.load('models/le_gender.pkl')
le_village = joblib.load('models/le_village.pkl')
mlb = joblib.load('models/mlb_symptoms.pkl')
safety_model = joblib.load('models/water_safety_model.pkl')
with open('models/water_safety_features.json') as f:
    safety_features = json.load(f)
scaler = joblib.load('models/feature_scaler.pkl')
# ...other models are loaded inside generate_report as needed...

app = FastAPI()

class InputData(BaseModel):
    turbidity: float
    pH: float
    temperature: float
    EC: float
    DO: float
    residual_chlorine: float
    source_type: str
    season: str
    location_lat: float
    location_lon: float
    village_id: str
    recent_reports_in_area: int
    age: int
    gender: str
    duration: float
    symptoms: str
    person_name: str

@app.get("/")
def root():
    return HTMLResponse("""
    <h2>Smart Community Health Monitoring API</h2>
    <p>Use <code>POST /predict</code> with JSON body to get predictions.</p>
    """)

@app.post('/predict')
def predict(data: InputData):
    input_dict = data.dict()
    print("[DEBUG] Input received for prediction:", input_dict)
    result = generate_report(input_dict)
    send_alert(result)  # Send alert if water is unsafe
    return result
