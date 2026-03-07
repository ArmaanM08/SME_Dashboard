import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Number of SMEs
n_samples = 5000

# -----------------------------
# Generate SME Financial Features
# -----------------------------

annual_revenue = np.random.normal(5000000, 1500000, n_samples)
annual_revenue = np.clip(annual_revenue, 200000, None)

monthly_transactions = np.random.normal(800, 250, n_samples)
monthly_transactions = np.clip(monthly_transactions, 50, None)

avg_transaction_value = np.random.normal(3500, 1200, n_samples)
avg_transaction_value = np.clip(avg_transaction_value, 200, None)

gst_compliance_score = np.random.uniform(0.5, 1.0, n_samples)

upi_transaction_ratio = np.random.uniform(0.1, 0.9, n_samples)

cashflow_stability = np.random.uniform(0.3, 1.0, n_samples)

repayment_history_score = np.random.uniform(0.4, 1.0, n_samples)

previous_loan_defaults = np.random.poisson(0.4, n_samples)

years_in_business = np.random.randint(1, 15, n_samples)

debt_to_revenue_ratio = np.random.uniform(0.1, 1.2, n_samples)

# -----------------------------
# Generate Default Probability (Target Variable)
# -----------------------------

risk_score = (
    -0.0000002 * annual_revenue
    -0.002 * monthly_transactions
    -0.001 * avg_transaction_value
    -1.5 * gst_compliance_score
    -1.2 * cashflow_stability
    -1.3 * repayment_history_score
    +1.8 * previous_loan_defaults
    +1.2 * debt_to_revenue_ratio
    -0.05 * years_in_business
)

# Convert to probability using sigmoid
default_probability = 1 / (1 + np.exp(-risk_score))

# -----------------------------
# Create DataFrame
# -----------------------------

data = pd.DataFrame({
    "annual_revenue": annual_revenue,
    "monthly_transactions": monthly_transactions,
    "avg_transaction_value": avg_transaction_value,
    "gst_compliance_score": gst_compliance_score,
    "upi_transaction_ratio": upi_transaction_ratio,
    "cashflow_stability": cashflow_stability,
    "repayment_history_score": repayment_history_score,
    "previous_loan_defaults": previous_loan_defaults,
    "years_in_business": years_in_business,
    "debt_to_revenue_ratio": debt_to_revenue_ratio,
    "default_probability": default_probability
})

# Save dataset
data.to_csv("synthetic_sme_default_risk_dataset.csv", index=False)

print(data.head())