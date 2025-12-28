import streamlit as st
import pandas as pd
import plotly.express as px
# Import the new EDA module
from modules.eda import get_global_map, get_sales_trend, get_top_countries, get_hourly_sales
# Import the clustering module
from modules.clustering import compute_rfm, perform_clustering, plot_clusters_3d
# Import the recommender module
from modules.recommender import get_basket_data, generate_rules
# Import prediction module
from modules.prediction import prepare_churn_data, train_churn_model, plot_churn_distribution

# Importing our module
from modules.data_loader import load_dataset, clean_data

# 1. Page configuration (icons and layout in the title bar)
st.set_page_config(page_title="RetailNexus 360", layout="wide")

#2. Title and Intro
st.title(" RetailNexus 360: Customer Intelligence Dashboard")
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    color: #555;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Advanced Analytics for E-Commerce: Segmentation, Recommendation & Prediction</p>', unsafe_allow_html=True)

st.markdown("---")

# 3. Sidebar (Menu)
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to:", 
    ["Project Overview", "Data Analysis (EDA)", "Customer Segmentation", "Market Basket Analysis", "Churn Prediction"])


# app.py এর সাইডবার সেকশনে (st.sidebar...)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Business Insights")



# 4. Data Loading Section
# with st.spinner('Loading Data from System... Please Wait...'):
#     try:
#         raw_df = load_dataset()
#         df = clean_data(raw_df)
#         st.sidebar.success(" Data Loaded Successfully!")
#         st.sidebar.info(f"Total Transactions: {df.shape[0]}")
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         st.stop()
# ==========================================
# ৪.  (DYNAMIC UPLOAD SYSTEM)
# ==========================================

st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio("Select Data Source:", ["Use Demo Dataset", "Upload My Own File"])

df = None 


if data_source == "Use Demo Dataset":
    with st.spinner('Loading Demo Data... Please Wait...'):
        try:
            raw_df = load_dataset() 
            df = clean_data(raw_df)
            st.sidebar.success("✅ Demo Data Loaded!")
        except Exception as e:
            st.error(f"Error loading demo data: {e}")


elif data_source == "Upload My Own File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        with st.spinner('Processing Uploaded File...'):
            try:
                
                if uploaded_file.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)
                
                
                required_columns = ['Invoice', 'Quantity', 'Price', 'Customer ID', 'Country', 'InvoiceDate']
                missing_cols = [col for col in required_columns if col not in raw_df.columns]
                
                if len(missing_cols) > 0:
                    st.error(f"⚠️ Error: Your dataset is missing these required columns: {missing_cols}")
                    st.stop() 
                else:
                    df = clean_data(raw_df) 
                    st.sidebar.success("✅ Your File Loaded Successfully!")
                    
            except Exception as e:
                st.error(f"File Error: {e}")


if df is None:
    st.info("👈 Please select 'Use Demo Dataset' or Upload a file from the Sidebar to start.")
    st.stop() 


st.sidebar.info(f"Total Transactions: {df.shape[0]}")



st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Business Insights")

# Automated Logic to generate text
total_sales = df['TotalAmount'].sum()
avg_order = df['TotalAmount'].mean()
top_country = df.groupby('Country')['TotalAmount'].sum().idxmax()

st.sidebar.markdown(f"""
- **Revenue Health:** The store has generated **${total_sales:,.0f}**.
- **Avg Basket Size:** Customers spend about **${avg_order:.2f}** per order.
- **Top Market:** **{top_country}** is your dominant market. Focus marketing there.
""")

# Dynamic Alert
churn_risk = 25 # (Suppose dummy or calculated)
if churn_risk > 20:
    st.sidebar.error(f"⚠️ Alert: Churn Risk is High! Check 'Churn Prediction' module.")
else:
    st.sidebar.success("✅ Customer retention is healthy.")

# 5. Page Logic
if options == "Project Overview":
    st.header("Project Overview")
    
    # KPI Metrics (Top bar)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['TotalAmount'].sum():,.0f}")
    col2.metric("Total Transactions", f"{df.shape[0]}")
    col3.metric("Unique Customers", f"{df['Customer ID'].nunique()}")
    col4.metric("Total Products", f"{df['Description'].nunique()}")
    
    st.write("### Raw Data Preview")
    st.dataframe(df.head(10))
    
    st.write("### Top Selling Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)


elif options == "Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Visualizing business performance and customer behavior patterns.")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Sales Trend", "Geographic Analysis", "Time Analysis"])
    
    with tab1:
        st.subheader("Monthly Revenue Trend")
        # Call the function from eda.py
        fig_trend = get_sales_trend(df)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("Observation: Identify which months have the highest sales (Seasonal Patterns).")
        
    with tab2:
        st.subheader("🌍 Global Market Penetration")
        st.write("Geospatial analysis of sales distribution across the world.")
        
        # Call the map function
        fig_map = get_global_map(df)
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.info("💡 Insight: The 3D globe shows market density. Darker colors indicate higher revenue regions.")
        
    with tab3:
        st.subheader("Peak Shopping Hours")
        fig_hour = get_hourly_sales(df)
        st.plotly_chart(fig_hour, use_container_width=True)
        st.caption("Observation: Helps in deciding when to push marketing emails.")

elif options == "Customer Segmentation":
    st.header("👥 Customer Segmentation (Clustering)")
    st.write("Using K-Means Algorithm to group customers based on purchasing behavior (RFM).")
    
    # Step 1: Compute RFM
    with st.spinner("Calculating RFM Metrics..."):
        rfm_df = compute_rfm(df)
        
    # User Input for Number of Clusters (Interactive!)
    st.sidebar.markdown("### Clustering Parameters")
    n_clusters = st.sidebar.slider("Select Number of Clusters (k)", 2, 6, 3)
    
    # Step 2: Perform Clustering
    if st.button("Run Clustering Algorithm"):
        clustered_df = perform_clustering(rfm_df, n_clusters=n_clusters)
        
        st.success(f"Algorithm successfully grouped customers into {n_clusters} segments!")
        
        # Tabs for Visualization and Data
        tab1, tab2 = st.tabs(["3D Visualization", "Cluster Data"])
        
        with tab1:
            st.write("### 3D Cluster Visualization")
            st.write("Rotate the graph to see how customers are separated.")
            fig_3d = plot_clusters_3d(clustered_df)
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with tab2:
            st.write("### Customer Segments Data")
            st.dataframe(clustered_df.sort_values('Monetary', ascending=False))
            
            # Show basic stats per cluster
            st.write("### Cluster Statistics (Average)")
            st.table(clustered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())

elif options == "Market Basket Analysis":
    st.header("🛒 Market Basket Analysis (Apriori Algorithm)")
    st.write("Discovering hidden associations between products (e.g., Bread -> Butter).")

    # Sidebar Filters
    st.sidebar.markdown("### Analysis Parameters")
    selected_country = st.sidebar.selectbox("Select Country for Analysis", 
                                            ["France", "Germany", "Spain", "EIRE"]) # UK is too big for live demo
    min_support = st.sidebar.slider("Minimum Support (Popularity)", 0.01, 0.2, 0.05)
    min_confidence = st.sidebar.slider("Minimum Confidence (Likelihood)", 0.1, 1.0, 0.5)

    if st.button("Run Association Rule Mining"):
        with st.spinner(f"Mining patterns for {selected_country}..."):
            
            # 1. Get Basket Data
            basket_sets = get_basket_data(df, country=selected_country)
            st.write(f"Processing **{basket_sets.shape[0]}** transactions from {selected_country}...")
            
            # 2. Generate Rules
            rules = generate_rules(basket_sets, min_support, min_confidence)
            
            if not rules.empty:
                st.success(f"Found {len(rules)} Association Rules!")
                
                # Format the output for better reading
                rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
                rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])
                
                # Show Top Rules
                st.subheader("Top Product Associations")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                             .sort_values('lift', ascending=False).head(10))
                
                # Recommendation Simulator
                st.markdown("---")
                st.subheader("🤖 Product Recommender Tool")
                product_list = rules['antecedents'].unique()
                selected_product = st.selectbox("Select a Product to get Recommendations:", product_list)
                
                if selected_product:
                    recs = rules[rules['antecedents'] == selected_product]
                    if not recs.empty:
                        st.write(f"If a customer buys **{selected_product}**, they are likely to buy:")
                        for _, row in recs.iterrows():
                            st.info(f"👉 **{row['consequents']}** (Confidence: {row['confidence']:.2%})")
                    else:
                        st.warning("No strong recommendations found for this product with current settings.")
            else:
                st.warning("No rules found! Try lowering the 'Minimum Support' in the sidebar.")

elif options == "Churn Prediction":
    st.header("🔮 Customer Churn Prediction (Supervised ML)")
    st.write("Predicting which customers are likely to stop buying.")
    
    # Step 1: Prepare Data
    rfm_df = prepare_churn_data(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", rfm_df.shape[0])
    with col2:
        churn_rate = rfm_df['Is_Churn'].mean() * 100
        st.metric("Current Churn Rate", f"{churn_rate:.2f}%")
        
    # Step 2: Visual Distribution
    st.plotly_chart(plot_churn_distribution(rfm_df), use_container_width=True)
    
    st.markdown("---")
    
    # Step 3: Train Models
    st.subheader("Train Machine Learning Models")
    st.write("Comparing **Random Forest (Bagging)** and **Gradient Boosting (Boosting)**.")
    
    if st.button("Train Models Now"):
        with st.spinner("Training models..."):
            results = train_churn_model(rfm_df)
            
            # Show Metrics
            c1, c2 = st.columns(2)
            c1.info(f"Random Forest Accuracy: **{results['RF_Accuracy']:.2%}**")
            c2.success(f"Gradient Boosting Accuracy: **{results['GB_Accuracy']:.2%}**")
            
            st.write("### Model Explanation")
            st.write("The **Gradient Boosting** model usually performs better as it corrects errors from previous trees.")
            
            # Prediction Tool
            st.markdown("---")
            st.subheader("🔎 Check Customer Risk")
            
            # Select a random customer to test
            customer_id = st.selectbox("Select Customer ID", rfm_df['Customer ID'].unique())
            
            if customer_id:
                cust_data = rfm_df[rfm_df['Customer ID'] == customer_id]
                freq = cust_data['Frequency'].values[0]
                mon = cust_data['Monetary'].values[0]
                
                # Predict
                model = results['GB_Model']
                pred = model.predict([[freq, mon]])[0]
                prob = model.predict_proba([[freq, mon]])[0][1]
                
                st.write(f"**Customer Stats:** Bought {freq} times, Spent ${mon:,.2f}")
                
                if pred == 1:
                    st.error(f"⚠️ High Risk! Probability of Churn: {prob:.2%}")
                else:
                    st.balloons()
                    st.success(f"✅ Safe Customer. Probability of Churn: {prob:.2%}")
    st.markdown("---")
    st.subheader("📄 Export Results")

    # Convert dataframe to CSV for download
    csv = rfm_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Churn Risk Report (CSV)",
        data=csv,
        file_name='churn_risk_report.csv',
        mime='text/csv',
    )
