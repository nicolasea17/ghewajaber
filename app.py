import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

# Function to download the model from Google Drive
def download_model(url, model_path):
    if not os.path.exists(model_path):
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

# Google Drive link
model_url = 'https://drive.google.com/uc?export=download&id=1xogKOHuSTUrRaX8Kv-FmLAiYH-AabVPP'
model_path = 'random_forest_model.joblib'

# Download the model
download_model(model_url, model_path)

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load('scaler.joblib')

# Function to preprocess the data
def preprocess(data):
    if 'inspection_date' not in data.columns or 'arrival_date' not in data.columns:
        st.error("The uploaded file must contain 'inspection_date' and 'arrival_date' columns.")
        return None
    
    data['inspection_date'] = pd.to_datetime(data['inspection_date'], dayfirst=True, errors='coerce')
    data['arrival_date'] = pd.to_datetime(data['arrival_date'], dayfirst=True, errors='coerce')

    def random_date(arrival_date):
        if pd.isna(arrival_date):
            return np.nan
        days_to_add = np.random.randint(2, 21)
        return arrival_date + pd.DateOffset(days=days_to_add)

    data['inspection_date'] = data.apply(
        lambda row: random_date(row['arrival_date']) if pd.isna(row['inspection_date']) else row['inspection_date'],
        axis=1
    )

    data['custom_fees'] = data['custom_fees'].replace('[^\d.]', '', regex=True)
    data['custom_fees'] = pd.to_numeric(data['custom_fees'], errors='coerce')

    def determine_exchange_rate(arrival_date):
        if pd.Timestamp('2018-01-01') <= arrival_date < pd.Timestamp('2022-12-01'):
            return 1500
        elif pd.Timestamp('2022-12-01') <= arrival_date < pd.Timestamp('2023-05-01'):
            return 15000
        elif pd.Timestamp('2023-05-01') <= arrival_date < pd.Timestamp('2024-01-01'):
            return 86000
        else:
            return None

    data['exchange_rate'] = data['arrival_date'].apply(determine_exchange_rate)
    data['custom_fees_usd'] = data.apply(lambda x: x['custom_fees'] / x['exchange_rate'] if pd.notna(x['exchange_rate']) else np.nan, axis=1)

    data['delay_days'] = (data['inspection_date'] - data['arrival_date']).dt.days
    data = data[data['delay_days'] >= 0]

    data['arrival_weekday'] = data['arrival_date'].dt.weekday
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_weekday_sin'] = np.sin(2 * np.pi * data['arrival_weekday'] / 7)
    data['arrival_weekday_cos'] = np.cos(2 * np.pi * data['arrival_weekday'] / 7)
    data['arrival_month_sin'] = np.sin(2 * np.pi * data['arrival_month'] / 12)
    data['arrival_month_cos'] = np.cos(2 * np.pi * data['arrival_month'] / 12)

    X_delay = data[['arrival_weekday_sin', 'arrival_weekday_cos', 'arrival_month_sin', 'arrival_month_cos', 'country_of_origin', 'container_type']]
    encoded_features = pd.get_dummies(X_delay[['container_type', 'country_of_origin']])
    X_delay = pd.concat([X_delay.drop(['container_type', 'country_of_origin'], axis=1), encoded_features], axis=1)

    return X_delay

# Streamlit interface
st.title('Import Delay Prediction')

st.write("""
This application allows you to upload your import data, preprocess it, and predict the delay days using a pre-trained model.
""")

uploaded_file = st.file_uploader("Choose an Excel file with your import data", type=['xlsx'])

if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        st.write("Data Uploaded Successfully!")

        st.write("Here is a preview of your data:")
        st.write(data.head())

        X_delay = preprocess(data)
        
        if X_delay is not None:
            X_delay_scaled = scaler.transform(X_delay)
            predictions = model.predict(X_delay_scaled)

            data['predicted_delay_days'] = predictions

            st.write("Here are the predictions for your data:")
            st.write(data)

            data.to_excel('predicted_imports.xlsx', index=False)
            st.download_button(
                label="Download predictions as Excel",
                data=data.to_excel(index=False),
                file_name='predicted_imports.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an Excel file to proceed.")
