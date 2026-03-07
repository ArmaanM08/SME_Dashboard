import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title='SME Credit Risk Predictor', layout='centered')
st.title('SME Credit Risk Predictor')

MODEL_PATH = 'models/linear_regression_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'synthetic_sme_default_risk_dataset.csv'

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error('Trained model or scaler not found. Please run the notebook to train and save the model to models/ directory.')
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# load dataset for charts
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = None

st.sidebar.header('Input SME Financial Features')

def_input = {}
# Define inputs with reasonable default ranges based on dataset summary
annual_revenue = st.sidebar.number_input('Annual revenue', min_value=0.0, value=5000000.0, step=100000.0)
monthly_transactions = st.sidebar.number_input('Monthly transactions', min_value=0.0, value=800.0, step=10.0)
avg_transaction_value = st.sidebar.number_input('Average transaction value', min_value=0.0, value=3000.0, step=50.0)
gst_compliance_score = st.sidebar.slider('GST compliance score', 0.0, 1.0, 0.7, 0.01)
upi_transaction_ratio = st.sidebar.slider('UPI transaction ratio', 0.0, 1.0, 0.4, 0.01)
cashflow_stability = st.sidebar.slider('Cashflow stability', 0.0, 1.0, 0.6, 0.01)
repayment_history_score = st.sidebar.slider('Repayment history score', 0.0, 1.0, 0.7, 0.01)
previous_loan_defaults = st.sidebar.number_input('Previous loan defaults (count)', min_value=0, value=0, step=1)
years_in_business = st.sidebar.number_input('Years in business', min_value=0, value=8, step=1)
debt_to_revenue_ratio = st.sidebar.number_input('Debt-to-revenue ratio', min_value=0.0, value=0.5, step=0.01)

input_df = pd.DataFrame([{
    'annual_revenue': annual_revenue,
    'monthly_transactions': monthly_transactions,
    'avg_transaction_value': avg_transaction_value,
    'gst_compliance_score': gst_compliance_score,
    'upi_transaction_ratio': upi_transaction_ratio,
    'cashflow_stability': cashflow_stability,
    'repayment_history_score': repayment_history_score,
    'previous_loan_defaults': previous_loan_defaults,
    'years_in_business': years_in_business,
    'debt_to_revenue_ratio': debt_to_revenue_ratio
}])

st.subheader('Input Summary')
st.write(input_df.T)

# Scale and predict
X_scaled = scaler.transform(input_df)
pred = model.predict(X_scaled)[0]

# Risk category thresholds (example): <= p10 Low, <= p60 Medium, else High
# Use dataset percentiles if available
if df is not None:
    p10 = df['default_probability'].quantile(0.10)
    p60 = df['default_probability'].quantile(0.60)
else:
    p10, p60 = 0.0001, 0.001

if pred <= p10:
    risk_cat = 'Low'
elif pred <= p60:
    risk_cat = 'Medium'
else:
    risk_cat = 'High'

st.subheader('Prediction')
st.metric('Predicted default probability', f"{pred:.6f}")
st.metric('Risk category', risk_cat)

st.subheader('Dataset Charts')
if df is not None:
    st.write('Distribution of annual revenue')
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df['annual_revenue'], bins=40, color='skyblue')
    st.pyplot(fig)

    st.write('Default probability by repayment history score (binned)')
    fig2, ax2 = plt.subplots(figsize=(6,3))
    df['rep_bin'] = pd.qcut(df['repayment_history_score'], q=4, duplicates='drop')
    df.groupby('rep_bin')['default_probability'].mean().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)
else:
    st.info('Dataset not available for charts.')

st.write('\n---\n')
st.write('Model and dashboard are simplified educational artifacts. For production use add validation, monitoring, and explainability layers.')
