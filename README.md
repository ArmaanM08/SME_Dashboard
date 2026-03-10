# SME Dashboard — Credit Risk Predictor

Lightweight web UI + prediction endpoint for an SME credit-risk model. Use the static frontend in `web/` and a serverless prediction function (Netlify) or the included Flask server for local development.

**Features**
- Web UI for entering borrower features and getting a default probability and risk category.
- Serverless function wrapper in `netlify/functions/predict.py` for easy Netlify deployment.
- Local Flask server in `server.py` for development and testing.

**Quick Start (local)**
1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the local Flask server (serves `web/` and the predict route):

   ```bash
   python3 server.py
   ```

4. Open the app at `http://localhost:8501` and use the Predict form.

**Run Netlify Functions Locally**
- Install Netlify CLI and run a local dev server (recommended for testing serverless behavior):

  ```bash
  npm install -g netlify-cli
  netlify dev
  ```

**Deploying to Netlify**
- The repository includes `netlify.toml` configured to publish the `web` folder and provide a SPA redirect to `index.html`.
- Ensure your site is set to build from the correct branch and that the functions directory is detected (`netlify/functions`).
- If you get a 404 when calling `/.netlify/functions/predict`, check the Netlify Deploy logs → Functions tab for build/install errors (often caused by missing binary dependencies or failing package installations).

**API: Predict Endpoint**
- Endpoint (Netlify): `POST /.netlify/functions/predict`
- Endpoint (local Flask): `POST /.netlify/functions/predict` or `POST /predict`
- Content-Type: `application/json`
- Expected payload fields (numbers):
  - `annual_revenue`, `monthly_transactions`, `avg_transaction_value`
  - `gst_compliance_score`, `upi_transaction_ratio`, `cashflow_stability`
  - `repayment_history_score`, `previous_loan_defaults`, `years_in_business`, `debt_to_revenue_ratio`

Example request body:

```json
{
  "annual_revenue": 120000,
  "monthly_transactions": 200,
  "avg_transaction_value": 600,
  "gst_compliance_score": 0.9,
  "upi_transaction_ratio": 0.4,
  "cashflow_stability": 0.8,
  "repayment_history_score": 0.95,
  "previous_loan_defaults": 0,
  "years_in_business": 3,
  "debt_to_revenue_ratio": 0.2
}
```

Example success response:

```json
{
  "predicted_default_probability": 0.00045,
  "risk_category": "Medium"
}
```

**Important Files**
- [web/index.html](web/index.html) — static frontend entry
- [web/main.js](web/main.js) — frontend logic and fetch to the predict endpoint
- [netlify/functions/predict.py](netlify/functions/predict.py) — serverless function handler
- [server.py](server.py) — lightweight Flask server for local dev
- [netlify.toml](netlify.toml) — Netlify publish/settings and SPA redirect
- [requirements.txt](requirements.txt) — Python dependencies

**Troubleshooting**
- "Unexpected token '<'" in the browser console: the app attempted to parse an HTML error page (usually a 404) as JSON. Check the network tab for the failing request and verify the endpoint exists.
- 404 for `/.netlify/functions/predict`: open Netlify Deploy logs and the Functions panel. If the function failed to build because of heavy dependencies (numpy/scikit-learn), consider using a lightweight runtime or building with a Docker-based build that includes wheels.
- Locally, if model files are missing, ensure `models/linear_regression_model.pkl` and `models/scaler.pkl` exist. The function falls back to a simple thresholding if dataset percentiles aren't available.

**Next steps / Recommendations**
- Add basic request validation and authentication for the API.
- Add automated tests for the function and a small CI job to ensure the function builds on Netlify.
- Add monitoring/alerts for model performance drift if used in production.

**License & Attribution**
This repository is a small demo for SME credit-risk prediction. Adapt and extend as needed.
