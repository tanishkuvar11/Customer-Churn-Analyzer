# Customer Churn Survival Analysis

A production-style ML system that predicts *when* a customer will churn — not just *if* — using survival analysis, SHAP explainability, and LLM-generated retention recommendations.

## Why Survival Analysis

Standard binary classifiers answer "will this customer churn?" Survival analysis answers "when will this customer churn?" — a fundamentally more useful question for retention teams.

A customer predicted to churn in 2 weeks needs immediate action. One predicted to churn in 6 months needs a different strategy. Binary classification treats both identically. Survival analysis doesn't.

Censored data — customers still active at observation time — is handled correctly by survival analysis. A binary classifier either discards or misuses this information.

## Key Findings from EDA

- Customers who called support **4+ times** had a churn rate of 46%+ vs ~11% for those with 3 or fewer calls — a sharp threshold, not a gradual trend
- Customers who **did not renew their contract** churned at 42% vs 11% for those who renewed — nearly 4x higher
- **Account age had almost no predictive power** (correlation: 0.017) — churn is driven by behavior, not tenure
- Log-rank tests confirmed all 5 features as statistically significant (p ≈ 0)

## Model Performance

- **Concordance Index: 0.76** — the model correctly ranks the survival times of two random customers 76% of the time (0.5 = random, 1.0 = perfect)

## How It Works

1. User inputs customer profile via Streamlit dashboard
2. Streamlit calls FastAPI `/predict` endpoint
3. Cox PH model returns a full survival curve (probability of retention at every week)
4. FastAPI `/explain` endpoint returns SHAP values for that customer
5. LLM (Llama 3 via Groq) generates plain-English risk interpretation and retention actions

## Tech Stack

| Tool | Purpose |
|---|---|
| Python, Pandas, NumPy | Data processing and feature engineering |
| Lifelines | Kaplan-Meier curves, log-rank tests, Cox PH model |
| SHAP | Individual prediction explainability |
| FastAPI + Uvicorn | Model serving as REST API |
| Streamlit | Interactive frontend dashboard |
| Groq / Llama 3 | LLM-generated retention recommendations |

## Project Structure

customer-churn-predictor/
├── notebook/
│   └── analysis.ipynb       # EDA, KM curves, log-rank tests, Cox model
├── model/
│   └── cox_model.pkl        # Trained Cox PH model
├── data/
│   └── churn.csv            # Telco churn dataset
├── app/
│   └── main.py              # FastAPI backend
├── streamlit/
│   └── dashboard.py         # Streamlit frontend
└── requirements.txt

## How to Run

**1. Install dependencies**
pip install -r requirements.txt

**2. Start the FastAPI backend**
uvicorn app.main:app --reload

**3. Start the Streamlit dashboard** (in a separate terminal)
streamlit run streamlit/dashboard.py

**4. Open** `http://localhost:8501` in your browser

> Add your Groq API key to a `.env` file: `GROQ_API_KEY=your_key_here`

## Scalability

The serving infrastructure is fully reusable. The FastAPI endpoints dynamically read feature names and metadata from the trained model — no hardcoded field names. To adapt to a new churn dataset: retrain the model, save the new pickle file, and redeploy. The API and dashboard require zero code changes.

## Limitations and Future Work

- Dataset is telecom-specific — EDA and feature engineering would need to be redone for a new domain
- SHAP explanation endpoint has ~30s latency due to exact algorithm over full background data — future optimization: approximate SHAP or precomputed explanations
- Future: automated retraining pipeline, batch prediction endpoint, DeepHit neural survival model for comparison