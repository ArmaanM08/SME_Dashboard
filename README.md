# SME Dashboard — Credit Risk Predictor

A modern, interactive Streamlit application for predicting SME credit risk. This dashboard provides real-time default probability evaluation and financial health monitoring using a machine learning model.

## Features
- **Live Predictive UI**: Interactive sliders and inputs that instantly update the predicted risk score.
- **Premium Data Visualizations**: Interactive Plotly Gauge and Radar charts for a comprehensive view of business health.
- **Seamless Deployment**: Ready to be deployed as a Streamlit app to Streamlit Community Cloud.

## Quick Start (local)
1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. The app will automatically open in your default browser at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

Deploying this dashboard is incredibly simple using Streamlit Community Cloud, as the repository is already structured correctly.

1. **Push to GitHub**: Ensure this repository is pushed to a public or private GitHub repository.
2. **Log in to Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. **Deploy App**: Click the **"New app"** button.
4. **Configure Deployment**:
   - **Repository**: Select your GitHub repository for the dashboard.
   - **Branch**: Select the main/master branch.
   - **Main file path**: Enter `app.py`.
5. **Click Deploy!**: Streamlit will automatically read the `requirements.txt` file, install dependencies, and host your interactive application securely.

## Model Notes
- The application uses a scikit-learn `LinearRegression` model (`models/linear_regression_model.pkl`) and a `StandardScaler` (`models/scaler.pkl`) pre-trained on a synthetic dataset representing SME financials.

## License & Attribution
This repository is a demo for SME credit-risk prediction. Adapt and extend as needed.
