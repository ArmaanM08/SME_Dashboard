import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration & Styling ---
st.set_page_config(
    page_title="SME Credit Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #64748b;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Data ---
@st.cache_resource
def load_model_assets():
    model_path = os.path.join("models", "linear_regression_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, scaler = load_model_assets()

# --- Helpers ---
def predict_risk(features_dict):
    if not model or not scaler:
        return 0.0, "Unknown"
        
    feature_cols = [
        'annual_revenue', 'monthly_transactions', 'avg_transaction_value',
        'gst_compliance_score', 'upi_transaction_ratio', 'cashflow_stability',
        'repayment_history_score', 'previous_loan_defaults', 'years_in_business',
        'debt_to_revenue_ratio'
    ]
    
    X = [[features_dict.get(c, 0.0) for c in feature_cols]]
    X_scaled = scaler.transform(np.array(X))
    pred = float(model.predict(X_scaled)[0])
    
    # Cap prediction between 0 and 1
    pred = max(0.0, min(1.0, pred))
    
    if pred <= 0.0001:
        risk = "Low"
    elif pred <= 0.001:
        risk = "Medium"
    else:
        risk = "High"
        
    return pred, risk

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 10000, # Scale up for readability if values are very small
        title={'text': "Risk Score (Scaled)", 'font': {'size': 24, 'color': '#1e293b'}},
        gauge={
            'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 1], 'color': "#bbf7d0"},      # Low
                {'range': [1, 10], 'color': "#fef08a"},     # Medium
                {'range': [10, 50], 'color': "#fecaca"}     # High
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score * 10000
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1e293b"})
    return fig

def create_radar_chart(features):
    # Normalize features for radar chart display (0-1 scale)
    # These are rough max estimates based on typical SME data for visualization
    max_vals = {
        'gst_compliance_score': 1.0,
        'upi_transaction_ratio': 1.0,
        'cashflow_stability': 1.0,
        'repayment_history_score': 1.0,
        'debt_to_revenue_ratio': 1.0
    }
    
    categories = ['GST Compliance', 'UPI Ratio', 'Cashflow Stability', 'Repayment History', 'Inverse Debt Ratio']
    
    # Calculate values, capping at 1.0. For debt, lower is better, so we invert it.
    values = [
        min(1.0, features['gst_compliance_score'] / max_vals['gst_compliance_score']),
        min(1.0, features['upi_transaction_ratio'] / max_vals['upi_transaction_ratio']),
        min(1.0, features['cashflow_stability'] / max_vals['cashflow_stability']),
        min(1.0, features['repayment_history_score'] / max_vals['repayment_history_score']),
        max(0.0, 1.0 - (features['debt_to_revenue_ratio'] / max_vals['debt_to_revenue_ratio']))
    ]
    
    # Close the loop
    values.append(values[0])
    categories.append(categories[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=2),
        name='Current Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#e2e8f0'),
            angularaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0')
        ),
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#1e293b"}
    )
    return fig

# --- App Layout ---

# Sidebar for Inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=60)
    st.title("Borrower Profile")
    st.markdown("Adjust parameters to simulate risk.")
    st.markdown("---")
    
    st.subheader("Financials")
    ann_rev = st.number_input("Annual Revenue (₹)", min_value=0, max_value=10000000, value=120000, step=10000)
    monthly_tx = st.slider("Monthly Transactions", min_value=0, max_value=2000, value=200)
    avg_tx_val = st.number_input("Avg Transaction Value (₹)", min_value=0, max_value=100000, value=600, step=100)
    
    st.subheader("Compliance & History")
    gst_score = st.slider("GST Compliance Score", 0.0, 1.0, 0.9)
    repay_score = st.slider("Repayment History Score", 0.0, 1.0, 0.95)
    prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)
    
    st.subheader("Business Metrics")
    upi_ratio = st.slider("UPI Transaction Ratio", 0.0, 1.0, 0.4)
    cashflow = st.slider("Cashflow Stability", 0.0, 1.0, 0.8)
    years_biz = st.slider("Years in Business", 0, 50, 3)
    debt_ratio = st.slider("Debt-to-Revenue Ratio", 0.0, 2.0, 0.2)

# Pack features
current_features = {
    'annual_revenue': ann_rev,
    'monthly_transactions': monthly_tx,
    'avg_transaction_value': avg_tx_val,
    'gst_compliance_score': gst_score,
    'upi_transaction_ratio': upi_ratio,
    'cashflow_stability': cashflow,
    'repayment_history_score': repay_score,
    'previous_loan_defaults': prev_defaults,
    'years_in_business': years_biz,
    'debt_to_revenue_ratio': debt_ratio
}

# Main Panel
st.title("🏢 SME Credit Risk Intelligence")
st.markdown("Real-time default probability evaluation and financial health monitoring.")

# Calculate Prediction
prob, risk_cat = predict_risk(current_features)

# Formatting
if risk_cat == "Low":
    risk_color = "🟢"
elif risk_cat == "Medium":
    risk_color = "🟡"
else:
    risk_color = "🔴"

# Top Metrics Row
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Predicted Probability", value=f"{prob:.5f}")
with col2:
    st.metric(label="Risk Category", value=f"{risk_color} {risk_cat}")
with col3:
    st.metric(label="Analysis Status", value="✅ Live")

st.markdown("<hr style='border: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

# Charts Row
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("### Risk Gauge")
    st.markdown("<p style='color: #64748b; font-size: 0.9rem;'>Visualizing the modeled default risk score.</p>", unsafe_allow_html=True)
    gauge_fig = create_gauge_chart(prob)
    st.plotly_chart(gauge_fig, use_container_width=True)

with col_chart2:
    st.markdown("### Business Health Profile")
    st.markdown("<p style='color: #64748b; font-size: 0.9rem;'>Multi-dimensional view of key business health indicators.</p>", unsafe_allow_html=True)
    radar_fig = create_radar_chart(current_features)
    st.plotly_chart(radar_fig, use_container_width=True)

# Feature Summary Table
with st.expander("View Detailed Feature Inputs"):
    st.markdown("Current snapshot of variables sent to the model inference engine:")
    df_features = pd.DataFrame([current_features]).T.reset_index()
    df_features.columns = ['Feature', 'Value']
    st.dataframe(df_features, use_container_width=True, hide_index=True)
