import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def prepare_churn_data(df):
    """
    Prepares dataset for Churn Prediction.
    Defines Churn: If a customer hasn't bought in the last 90 days, they are 'Churned'.
    """
    # Calculate RFM first
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm.rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalAmount': 'Monetary'}, inplace=True)
    
    # Define Target Variable: Churn (1 if Recency > 90 days, else 0)
    rfm['Is_Churn'] = rfm['Recency'].apply(lambda x: 1 if x > 90 else 0)
    
    return rfm

def train_churn_model(rfm_df):
    """
    Trains Ensemble Models (Random Forest & Gradient Boosting) to predict Churn.
    """
    # Features (X) and Target (y)
    # We remove 'Recency' from X because it directly defines Churn (Data Leakage prevention for demo)
    # Ideally, we use past behavior to predict future, but for this lab, using F & M is fine.
    X = rfm_df[['Frequency', 'Monetary']]
    y = rfm_df['Is_Churn']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Random Forest (Bagging - Exp 10)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # 2. Gradient Boosting (Boosting - Exp 11)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    # Return models and metrics
    results = {
        'RF_Model': rf_model,
        'GB_Model': gb_model,
        'RF_Accuracy': rf_acc,
        'GB_Accuracy': gb_acc,
        'Test_Data': (X_test, y_test)
    }
    
    return results

def plot_churn_distribution(rfm_df):
    """
    Visualizes Churn vs Non-Churn customers.
    """
    counts = rfm_df['Is_Churn'].value_counts().reset_index()
    counts.columns = ['Is_Churn', 'Count']
    counts['Label'] = counts['Is_Churn'].map({1: 'Churned (Left)', 0: 'Active (Stayed)'})
    
    fig = px.pie(counts, values='Count', names='Label', 
                 title='Customer Churn Distribution',
                 color='Label',
                 color_discrete_map={'Churned (Left)': 'red', 'Active (Stayed)': 'green'})
    return fig