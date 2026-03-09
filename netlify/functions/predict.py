import os
import json
import traceback
from typing import Tuple

import joblib
import numpy as np


def _load_model_and_scaler() -> Tuple[object, object, object]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(root, 'models', 'linear_regression_model.pkl')
    scaler_path = os.path.join(root, 'models', 'scaler.pkl')
    data_path = os.path.join(root, 'synthetic_sme_default_risk_dataset.csv')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # try to read dataset percentiles for thresholds (optional)
    p10 = None
    p60 = None
    try:
        import pandas as pd
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if 'default_probability' in df.columns:
                p10 = float(df['default_probability'].quantile(0.10))
                p60 = float(df['default_probability'].quantile(0.60))
    except Exception:
        pass

    return model, scaler, (p10, p60)


def categorize(pred: float, thresholds: Tuple[float, float]) -> str:
    p10, p60 = thresholds
    if p10 is None or p60 is None:
        # fallback thresholds
        p10, p60 = 0.0001, 0.001
    if pred <= p10:
        return 'Low'
    if pred <= p60:
        return 'Medium'
    return 'High'


MODEL_CACHE = None


def handler(event, context):
    global MODEL_CACHE
    try:
        if MODEL_CACHE is None:
            MODEL_CACHE = _load_model_and_scaler()
        model, scaler, thresholds = MODEL_CACHE

        body = event.get('body') or '{}'
        payload = json.loads(body)

        # expected payload: dict of feature_name -> value
        # Build feature vector in the same order used in training
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
        risk = categorize(pred, thresholds)

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'predicted_default_probability': pred, 'risk_category': risk})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e), 'trace': traceback.format_exc()})
        }
