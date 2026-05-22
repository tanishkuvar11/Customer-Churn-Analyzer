import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

st.title("Customer Churn Survival Analysis")
st.write("Enter customer details: ")

response = requests.get("http://127.0.0.1:8000/features")
features = response.json()["features"]

metadata = requests.get("http://127.0.0.1:8000/feature_metadata").json()

load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

inputs = {}
for feature, meta in metadata.items():
    if meta["type"] == "binary":
        inputs[feature] = st.selectbox(feature, options=[0, 1], 
                  format_func=lambda x: "Yes" if x == 1 else "No")
    elif meta["type"] == "integer":
        inputs[feature] = st.number_input(feature, value=0, step=1)
    else:
        inputs[feature] = st.number_input(feature, value=0)

if st.button("Predict Survival"):
    response=requests.post(
        "http://127.0.0.1:8000/predict",
        json=inputs
    )
    result=response.json()
    times = result["times"]
    probs = result["survival_probabilities"]
    one_year_prob = probs[min(range(len(times)), key=lambda i: abs(times[i]-52))]
    st.metric(label="1-Year Survival Probability", value=f"{one_year_prob:.1%}")

    fig, ax = plt.subplots()
    ax.plot(result["times"], result["survival_probabilities"])
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Customer Survival Curve")
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax.legend()

    shap_response = requests.post("http://127.0.0.1:8000/explain", json=inputs).json()

    fig2, ax2 = plt.subplots()
    features = shap_response["features"]
    values = shap_response["shap_values"]

    sorted_pairs = sorted(zip(values, features), key=lambda x: abs(x[0]))
    sorted_values, sorted_features = zip(*sorted_pairs)

    colors = ['red' if v > 0 else 'blue' for v in sorted_values]
    ax2.barh(sorted_features, sorted_values, color=colors)
    ax2.set_title("Feature Contributions to Churn Risk")
    ax2.set_xlabel("SHAP Value (log hazard)")
    ax2.axvline(x=0, color='black', linestyle='--')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Survival Curve")
        st.pyplot(fig)

    with col2:
        st.markdown("### Feature Contributions")
        st.pyplot(fig2)

    summary_prompt = f"""
    You are a customer retention analyst. A customer has the following profile:
    {dict(zip(shap_response['features'], inputs))}

    The top risk drivers for this customer ranked by importance:
    {sorted(dict(zip(shap_response['features'], shap_response['shap_values'])).items(), key=lambda x: abs(x[1]), reverse=True)}

    This customer has a {one_year_prob:.1%} probability of NOT churning over the next year.

    Do two things:
    1. In 2 sentences, explain WHY this specific customer is at risk. Reference their actual input values (e.g. "made 6 support calls", "no contract renewal"). Do NOT mention SHAP values or any numbers from the risk drivers list.
    2. Give 2 specific retention actions targeting their top 2 risk drivers. Be concrete — reference the customer's actual values and suggest measurable actions with specific targets.

    Do not use generic phrases like "consider reaching out", "improve customer experience", or "enhance satisfaction".
    Do not mention SHAP, coefficients, or any model internals.
    """

    completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": summary_prompt}])

    st.write(completion.choices[0].message.content)