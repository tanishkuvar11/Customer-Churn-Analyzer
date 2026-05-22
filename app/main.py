from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import shap
import numpy as np

app=FastAPI()

df = pd.read_csv('dataset/telecom_churn.csv')

with open('model/cox_model.pkl', 'rb') as f:
    model=pickle.load(f)
background_data = pd.read_csv('dataset/telecom_churn.csv')[list(model.params_.index)]
explainer = shap.Explainer(model.predict_partial_hazard, background_data, algorithm='exact')

class Customer(BaseModel):
    ContractRenewal: int
    CustServCalls: int
    DayMins: float
    OverageFee: float
    MonthlyCharge: float

@app.post("/predict")
def predict(customer:Customer):
    data=pd.DataFrame([customer.model_dump()])
    survival=model.predict_survival_function(data)
    times = survival.index.tolist()
    probs = survival.iloc[:, 0].tolist()
    
    return {"times": times, "survival_probabilities": probs}

@app.get("/features")
def get_features():
    return {'features': list(model.params_.index)}

@app.get("/feature_metadata")
def get_feature_metadata():
    metadata = {}
    for feature in model.params_.index:
        unique_vals = df[feature].nunique()
        if unique_vals == 2:
            metadata[feature] = {"type": "binary"}
        elif df[feature].dtype == 'int64':
            metadata[feature] = {
                "type": "integer"
            }
        else:
            metadata[feature] = {
                "type": "float"
            }
    return metadata

@app.post("/explain")
def explain(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    shap_values = explainer(data)
    return {
        "features": list(model.params_.index),
        "shap_values": shap_values.values[0].tolist(),
        "base_value": float(shap_values.base_values[0])
    }