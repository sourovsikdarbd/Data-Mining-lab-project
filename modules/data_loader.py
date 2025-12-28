import pandas as pd
import streamlit as st

# This function will load the data and cache it 
# so that time is not wasted loading it repeatedly.
@st.cache_data
def load_dataset():
    file_path = "dataset/online_retail_II.xlsx"
    
    # We are taking the Year 2010-2011 sheet because it is the latest 
    # Your PC is fast, so there is no problem in reading the whole thing
    df = pd.read_excel(file_path, sheet_name='Year 2010-2011')
    return df

def clean_data(df):
    # 1. Remove missing customer ID
    df = df.dropna(subset=['Customer ID'])
    
    # 2. Converting Customer ID from Float to String
    df['Customer ID'] = df['Customer ID'].astype(str).str.split('.').str[0]
    
    # 3. Remove Cancelled Orders (Starting with Invoice C)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    # 4. Creating a Total Price Column (Quantity * Price)
    df['TotalAmount'] = df['Quantity'] * df['Price']
    
    # 5. Get invoice date in datetime format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df