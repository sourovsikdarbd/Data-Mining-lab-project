import pandas as pd
import plotly.express as px

def get_sales_trend(df):
    """
    Analyzes monthly sales trends to identify peak seasons.
    Returns a Plotly line chart.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_trend = df.copy()
    
    # Extract Month and Year for grouping
    df_trend['Month_Year'] = df_trend['InvoiceDate'].dt.to_period('M').astype(str)
    
    # Group by Month and sum the TotalAmount
    monthly_sales = df_trend.groupby('Month_Year')['TotalAmount'].sum().reset_index()
    
    # Create Line Chart
    fig = px.line(monthly_sales, x='Month_Year', y='TotalAmount', 
                  title='Monthly Sales Trend Analysis',
                  markers=True,
                  labels={'TotalAmount': 'Total Sales ($)', 'Month_Year': 'Month'})
    
    return fig

def get_top_countries(df):
    """
    Identifies top 10 countries by total sales volume.
    Returns a Plotly bar chart.
    """
    # Group by Country and sum TotalAmount
    country_sales = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10).reset_index()
    
    # Create Bar Chart
    fig = px.bar(country_sales, x='Country', y='TotalAmount', 
                 title='Top 10 Countries by Revenue',
                 color='TotalAmount',
                 color_continuous_scale='Viridis',
                 labels={'TotalAmount': 'Revenue ($)'})
    
    return fig

def get_hourly_sales(df):
    """
    Analyzes sales distribution by hour of the day to optimize ad timing.
    Returns a Plotly histogram.
    """
    df_hour = df.copy()
    df_hour['Hour'] = df_hour['InvoiceDate'].dt.hour
    
    hourly_counts = df_hour.groupby('Hour')['Invoice'].nunique().reset_index()
    
    fig = px.bar(hourly_counts, x='Hour', y='Invoice', 
                 title='Peak Shopping Hours',
                 labels={'Invoice': 'Number of Orders', 'Hour': 'Hour of Day'})
    return fig


def get_global_map(df):
    """
    Generates an interactive World Map showing sales density.
    """
    country_sales = df.groupby('Country')['TotalAmount'].sum().reset_index()
    
    # Plotly Choropleth Map
    fig = px.choropleth(country_sales, 
                        locations="Country", 
                        locationmode='country names',
                        color="TotalAmount", 
                        hover_name="Country",
                        range_color=[0, 100000], # Range adjust kora jai
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="🌍 Global Sales Heatmap")
    fig.update_layout(geo=dict(showframe=False, showcoastlines=False, projection_type='orthographic'))
    return fig 