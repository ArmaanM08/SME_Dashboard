import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_PATH = 'synthetic_sme_default_risk_dataset.csv'
CHART_DIR = 'charts'
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, path, dpi=200):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    assert os.path.exists(DATA_PATH), f'Data file not found: {DATA_PATH}'
    df = pd.read_csv(DATA_PATH)

    print('Loaded dataset:', DATA_PATH)
    print('Shape:', df.shape)
    print(df.head().to_string(index=False))

    # 1) Missing values and duplicates
    print('\nMissing values per column:')
    print(df.isnull().sum())
    print('\nDuplicate rows:', df.duplicated().sum())
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2) Outlier handling: clip at 1st and 99th percentiles for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'default_probability' in numeric_cols:
        numeric_cols.remove('default_probability')

    for c in numeric_cols:
        lo = df[c].quantile(0.01)
        hi = df[c].quantile(0.99)
        df[c] = df[c].clip(lower=lo, upper=hi)
    print('\nApplied 1%-99% clipping to numeric features')

    # 3) Correlation heatmap
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'shrink': .8})
    ax.set_title('Correlation matrix')
    save_fig(fig, os.path.join(CHART_DIR, 'correlation_heatmap.png'))
    print('Saved correlation heatmap to', os.path.join(CHART_DIR, 'correlation_heatmap.png'))

    # 4) Distribution of annual revenue
    if 'annual_revenue' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['annual_revenue'], bins=40, kde=True, ax=ax, color='skyblue')
        ax.set_title('Distribution of Annual Revenue')
        ax.set_xlabel('Annual Revenue')
        save_fig(fig, os.path.join(CHART_DIR, 'annual_revenue_distribution.png'))
        print('Saved annual revenue distribution to', os.path.join(CHART_DIR, 'annual_revenue_distribution.png'))

    # 5) Scatter: monthly_transactions vs avg_transaction_value colored by default_probability
    if set(['monthly_transactions', 'avg_transaction_value', 'default_probability']).issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(df['monthly_transactions'], df['avg_transaction_value'], c=df['default_probability'], cmap='viridis', alpha=0.8)
        ax.set_xlabel('Monthly transactions')
        ax.set_ylabel('Avg transaction value')
        ax.set_title('Monthly transactions vs Avg transaction value (colored by default probability)')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('default_probability')
        save_fig(fig, os.path.join(CHART_DIR, 'transactions_vs_avgvalue.png'))
        print('Saved transactions vs avg transaction value chart to', os.path.join(CHART_DIR, 'transactions_vs_avgvalue.png'))

    # 6) Boxplots by default probability quartiles
    if 'repayment_history_score' in df.columns and 'cashflow_stability' in df.columns and 'default_probability' in df.columns:
        df['default_q'] = pd.qcut(df['default_probability'], q=4, duplicates='drop', labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='default_q', y='repayment_history_score', data=df, ax=ax)
        ax.set_title('Repayment history score by default probability quartile')
        save_fig(fig, os.path.join(CHART_DIR, 'repayment_history_by_default_q.png'))
        print('Saved repayment history boxplot to', os.path.join(CHART_DIR, 'repayment_history_by_default_q.png'))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='default_q', y='cashflow_stability', data=df, ax=ax)
        ax.set_title('Cashflow stability by default probability quartile')
        save_fig(fig, os.path.join(CHART_DIR, 'cashflow_stability_by_default_q.png'))
        print('Saved cashflow stability boxplot to', os.path.join(CHART_DIR, 'cashflow_stability_by_default_q.png'))

    # 7) SME Segmentation using K-Means
    cluster_features = ['annual_revenue', 'monthly_transactions', 'avg_transaction_value', 'cashflow_stability', 'repayment_history_score', 'debt_to_revenue_ratio', 'upi_transaction_ratio']
    cluster_features = [c for c in cluster_features if c in df.columns]
    X_cluster = df[cluster_features].copy()

    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

    inertias = []
    K_RANGE = range(1, 11)
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_cluster_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(K_RANGE), inertias, marker='o')
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for optimal k')
    save_fig(fig, os.path.join(CHART_DIR, 'kmeans_elbow.png'))
    print('Saved KMeans elbow plot to', os.path.join(CHART_DIR, 'kmeans_elbow.png'))

    # choose k=3
    k_opt = 3
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    df['cluster'] = clusters

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_cluster_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=df['cluster'], cmap='tab10')
    ax.set_title('KMeans clusters (PCA projection)')
    save_fig(fig, os.path.join(CHART_DIR, 'kmeans_clusters_pca.png'))
    print('Saved KMeans PCA cluster projection to', os.path.join(CHART_DIR, 'kmeans_clusters_pca.png'))

    cluster_default = df.groupby('cluster')['default_probability'].mean().sort_values()
    order = cluster_default.index.tolist()
    if len(order) >= 3:
        label_map = {order[0]: 'Low Risk', order[1]: 'Medium Risk', order[2]: 'High Risk'}
    else:
        # fallback
        label_map = {i: f'Segment_{i}' for i in order}
    df['risk_segment'] = df['cluster'].map(label_map)

    df[['cluster', 'risk_segment']].to_csv(os.path.join(OUTPUT_DIR, 'df_with_segments.csv'), index=False)
    print('Saved dataset with segments to', os.path.join(OUTPUT_DIR, 'df_with_segments.csv'))

    # 8) Regression model (Linear Regression)
    feature_cols = ['annual_revenue', 'monthly_transactions', 'avg_transaction_value', 'gst_compliance_score', 'upi_transaction_ratio', 'cashflow_stability', 'repayment_history_score', 'previous_loan_defaults', 'years_in_business', 'debt_to_revenue_ratio']
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df['default_probability'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('\nLinear Regression evaluation:')
    print('MSE:', mse)
    print('R2:', r2)

    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, 'linear_regression_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(lr, model_path)
    joblib.dump(scaler, scaler_path)
    print('Saved Linear Regression model to', model_path)
    print('Saved scaler to', scaler_path)

    # Save a simple diagnostic plot: predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual default_probability')
    ax.set_ylabel('Predicted default_probability')
    ax.set_title('Predicted vs Actual')
    save_fig(fig, os.path.join(CHART_DIR, 'predicted_vs_actual.png'))
    print('Saved predicted vs actual plot to', os.path.join(CHART_DIR, 'predicted_vs_actual.png'))

    print('\nAll done. Charts in', CHART_DIR, '| Models in', MODEL_DIR, '| Outputs in', OUTPUT_DIR)


if __name__ == '__main__':
    main()
