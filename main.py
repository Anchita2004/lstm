from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import pytesseract
from PIL import Image
import pdfplumber
import re
import os
# Triggering redeploy

# Path to tesseract.exe
if os.name == "nt":  # Only set on Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\anchi\Downloads\tesseract.exe"


# Load model and scaler
model = tf.keras.models.load_model("lstm_model.h5")

scaler = joblib.load("scaler.pkl")

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],  # Update this if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class PatientData(BaseModel):
    Age: float
    RestingBP: float
    Cholesterol: float
    FastingBS: float
    MaxHR: float
    Oldpeak: float
    Sex: str
    ChestPainType: str
    RestingECG: str
    ExerciseAngina: str
    ST_Slope: str

@app.get("/")
def read_root():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    try:
        transformed = scaler.transform(df)
        reshaped = transformed.reshape((1, 1, transformed.shape[1]))
        prediction = model.predict(reshaped)
        return {
            "probability": float(prediction[0][0]),
            "prediction": int(prediction[0][0] > 0.5)
        }
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

@app.post("/explain")
def explain(data: PatientData):
    def assess_risk_explained(row):
        score = 0
        reasons = []

        if (row['Sex'] == 'M' and row['Age'] >= 55) or (row['Sex'] == 'F' and row['Age'] >= 65):
            score += 1
            reasons.append("Age ≥ threshold for sex")

        if row['RestingBP'] >= 140:
            score += 2
            reasons.append("Resting BP ≥140")

        if row['Cholesterol'] >= 240:
            score += 2
            reasons.append("Cholesterol ≥240")
        elif row['Cholesterol'] >= 200:
            score += 1
            reasons.append("Borderline high cholesterol")

        if row['FastingBS'] == 1:
            score += 1
            reasons.append("Fasting BS >120")

        if row['RestingECG'] in ['ST', 'LVH']:
            score += 1
            reasons.append("Abnormal ECG")

        if row['ChestPainType'] == 'TA':
            score += 2
            reasons.append("Typical angina")
        elif row['ChestPainType'] == 'ATA':
            score += 1
            reasons.append("Atypical angina")

        if row['ExerciseAngina'] == 'Y':
            score += 1
            reasons.append("Exercise angina")

        expected_hr = (220 - row['Age']) * 0.8
        if row['MaxHR'] < expected_hr:
            score += 1
            reasons.append("Low MaxHR")

        if row['Oldpeak'] >= 2.0:
            score += 2
            reasons.append("Oldpeak ≥2.0")
        elif row['Oldpeak'] >= 1.0:
            score += 1
            reasons.append("Oldpeak 1.0–1.99")

        if row['ST_Slope'] == 'Down':
            score += 2
            reasons.append("ST Slope down")
        elif row['ST_Slope'] == 'Flat':
            score += 1
            reasons.append("ST Slope flat")

        if score >= 7:
            level = "High Risk"
        elif score >= 4:
            level = "Moderate Risk"
        elif score >= 1:
            level = "Low Risk"
        else:
            level = "Minimal Risk"

        return level, score, reasons

    def get_recommendations_with_explanation(level):
        rec = {"medicine_classes": [], "preventions": [], "treatment_plan": ""}
        if level == "High Risk":
            rec["medicine_classes"] = ["Statins", "ACE-I/ARB", "Beta-blockers"]
            rec["preventions"] = ["Diet", "Exercise", "Quit smoking", "Sleep"]
            rec["treatment_plan"] = "See cardiologist monthly"
        elif level == "Moderate Risk":
            rec["medicine_classes"] = ["Lifestyle changes", "ACE-I if BP high"]
            rec["preventions"] = ["Healthy habits", "Monitor vitals"]
            rec["treatment_plan"] = "Primary care every 3–6 months"
        elif level == "Low Risk":
            rec["medicine_classes"] = ["None"]
            rec["preventions"] = ["Maintain healthy lifestyle"]
            rec["treatment_plan"] = "Annual checkup"
        else:
            rec["medicine_classes"] = ["None"]
            rec["preventions"] = ["Stay active"]
            rec["treatment_plan"] = "Routine checks"
        return rec

    row = data.dict()
    level, score, reasons = assess_risk_explained(row)
    recs = get_recommendations_with_explanation(level)

    return {
        "level": level,
        "score": score,
        "reasons": reasons,
        "medicines": recs["medicine_classes"],
        "preventions": recs["preventions"],
        "plan": recs["treatment_plan"]
    }

# ✅ Helper for parsing report text
def extract_fields_from_text(text):
    fields = {}
    patterns = {
        "Age": r"Age[:\s]*([0-9]{1,3})",
        "RestingBP": r"RestingBP[:\s]*([0-9]+)",
        "Cholesterol": r"Cholesterol[:\s]*([0-9]+)",
        "FastingBS": r"FastingBS[:\s]*([01])",
        "MaxHR": r"MaxHR[:\s]*([0-9]+)",
        "Oldpeak": r"Oldpeak[:\s]*([0-9.]+)",
        "Sex": r"Sex[:\s]*(Male|Female|M|F)",
        "ChestPainType": r"ChestPainType[:\s]*(\w+)",
        "RestingECG": r"RestingECG[:\s]*(\w+)",
        "ExerciseAngina": r"ExerciseAngina[:\s]*(Y|N)",
        "ST_Slope": r"ST[_ ]?Slope[:\s]*(\w+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key in ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"]:
                fields[key] = int(value)
            elif key == "Oldpeak":
                fields[key] = float(value)
            else:
                fields[key] = value.upper() if key in ["Sex", "ExerciseAngina"] else value
        else:
            fields[key] = ""

    return fields

@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):
    contents = await file.read()
    filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    path = f"temp/{filename}"
    os.makedirs("temp", exist_ok=True)
    
    with open(path, "wb") as f:
        f.write(contents)

    extracted_text = ""

    try:
        if filename.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                texts = []
                for page in pdf.pages:
                    txt = page.extract_text()
                    if not txt:
                        # Use OCR if PDF page is image-based
                        img = page.to_image(resolution=300).original
                        txt = pytesseract.image_to_string(img)
                    texts.append(txt or "")
                extracted_text = "\n".join(texts)

            if not extracted_text.strip():
                return JSONResponse(status_code=400, content={"error": "No extractable text found in PDF."})
            
            parsed = extract_fields_from_text(extracted_text)
            return {"parsed_fields": parsed}

        elif filename.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(path)
            extracted_text = pytesseract.image_to_string(img)

            if not extracted_text.strip():
                return JSONResponse(status_code=400, content={"error": "No extractable text in image."})

            parsed = extract_fields_from_text(extracted_text)
            return {"parsed_fields": parsed}

        elif filename.endswith(".csv"):
            df = pd.read_csv(path)
            if df.shape[0] == 0:
                return JSONResponse(status_code=400, content={"error": "Empty CSV file"})

            row = df.iloc[0].to_dict()
            normalized = {k.strip(): v for k, v in row.items()}
            return {"parsed_fields": normalized}

        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
