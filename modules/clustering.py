import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_rfm(df):
    """
    Computes RFM (Recency, Frequency, Monetary) metrics for each customer.
    """
    # Reference date (usually the day after the last transaction in dataset)
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Group by Customer ID and calculate metrics
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'Invoice': 'nunique', # Frequency
        'TotalAmount': 'sum' # Monetary Value
    })
    
    # Rename columns for clarity
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)
    
    return rfm

def perform_clustering(rfm_df, n_clusters=3):
    """
    Applies K-Means clustering on the RFM data.
    Returns the dataframe with a 'Cluster' column.
    """
    # 1. Scaling the data (K-Means is sensitive to variance)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    
    # 2. Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    
    # 3. Assign cluster labels
    rfm_df['Cluster'] = kmeans.labels_
    
    # Map Cluster IDs to string for better visualization logic (Optional naming)
    rfm_df['Cluster_Label'] = rfm_df['Cluster'].astype(str)
    
    return rfm_df

def plot_clusters_3d(rfm_df):
    """
    Creates a 3D Scatter Plot to visualize customer segments.
    """
    fig = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary',
              color='Cluster_Label',
              opacity=0.7,
              size_max=10,
              title='3D Customer Segmentation (RFM Analysis)',
              labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Count)', 'Monetary': 'Monetary ($)'})
    
    return fig