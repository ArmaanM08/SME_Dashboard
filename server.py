from flask import Flask, request, send_from_directory, jsonify
import os
import json
import joblib
import numpy as np

app = Flask(__name__, static_folder='web', static_url_path='')

# Load model and scaler
ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, 'models', 'linear_regression_model.pkl')
SCALER_PATH = os.path.join(ROOT, 'models', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json() or {}
    feature_cols = [
        'annual_revenue', 'monthly_transactions', 'avg_transaction_value',
        'gst_compliance_score', 'upi_transaction_ratio', 'cashflow_stability',
        'repayment_history_score', 'previous_loan_defaults', 'years_in_business',
        'debt_to_revenue_ratio'
    ]
    X = []
    for c in feature_cols:
        v = payload.get(c, 0)
        X.append(float(v))
    X_arr = np.array([X])
    X_scaled = scaler.transform(X_arr)
    pred = float(model.predict(X_scaled)[0])
    # simple categorization
    if pred <= 0.0001:
        risk = 'Low'
    elif pred <= 0.001:
        risk = 'Medium'
    else:
        risk = 'High'
    return jsonify({'predicted_default_probability': pred, 'risk_category': risk})


# Netlify function compatibility route (local dev)
@app.route('/.netlify/functions/predict', methods=['POST', 'OPTIONS'])
def netlify_function_predict():
    if request.method == 'OPTIONS':
        return ('', 200)
    return predict()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)
