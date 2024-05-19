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
    required_columns = ['inspection date', 'arrival date', 'custom fees', 'country of origin', 'container type']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
        return None
    
    # Drop unnecessary columns early
    data = data.drop(columns=['container number', 'x', 'color code', 'container'], errors='ignore')

    st.write("After dropping unnecessary columns, here are the remaining columns:")
    st.write(data.columns)
    
    # Convert date columns to datetime format
    for col in ['inspection date', 'arrival date']:
        data[col] = pd.to_datetime(data[col], dayfirst=True, errors='coerce')
        if data[col].isna().sum() > 0:
            st.warning(f"Some dates in the column '{col}' could not be converted and are set to NaT.")

    def random_date(arrival_date):
        if pd.isna(arrival_date):
            return np.nan
        days_to_add = np.random.randint(2, 21)
        return arrival_date + pd.DateOffset(days=days_to_add)

    data['inspection date'] = data.apply(
        lambda row: random_date(row['arrival date']) if pd.isna(row['inspection date']) else row['inspection date'],
        axis=1
    )

    data['custom fees'] = data['custom fees'].replace('[^\d.]', '', regex=True)
    data['custom fees'] = pd.to_numeric(data['custom fees'], errors='coerce')

    def determine_exchange_rate(arrival_date):
        if pd.Timestamp('2018-01-01') <= arrival_date < pd.Timestamp('2022-12-01'):
            return 1500
        elif pd.Timestamp('2022-12-01') <= arrival_date < pd.Timestamp('2023-05-01'):
            return 15000
        elif pd.Timestamp('2023-05-01') <= arrival_date < pd.Timestamp('2024-01-01'):
            return 86000
        else:
            return None

    data['exchange_rate'] = data['arrival date'].apply(determine_exchange_rate)
    data['custom_fees_usd'] = data.apply(lambda x: x['custom fees'] / x['exchange_rate'] if pd.notna(x['exchange_rate']) else np.nan, axis=1)

    data['delay_days'] = (data['inspection date'] - data['arrival date']).dt.days
    data = data[data['delay_days'] >= 0]

    data['arrival_weekday'] = data['arrival date'].dt.weekday
    data['arrival_month'] = data['arrival date'].dt.month
    data['arrival_weekday_sin'] = np.sin(2 * np.pi * data['arrival_weekday'] / 7)
    data['arrival_weekday_cos'] = np.cos(2 * np.pi * data['arrival_weekday'] / 7)
    data['arrival_month_sin'] = np.sin(2 * np.pi * data['arrival_month'] / 12)
    data['arrival_month_cos'] = np.cos(2 * np.pi * data['arrival_month'] / 12)

    X_delay = data[['arrival_weekday_sin', 'arrival_weekday_cos', 'arrival_month_sin', 'arrival_month_cos', 'country of origin', 'container type']]
    encoded_features = pd.get_dummies(X_delay[['container type', 'country of origin']], prefix=['container_type', 'country_of_origin'])
    X_delay = pd.concat([X_delay.drop(['container type', 'country of origin'], axis=1), encoded_features], axis=1)

    return X_delay

# Streamlit interface
st.title('Import Delay Prediction')

st.write("""
This application allows you to upload your import data, preprocess it, and predict the delay days using a pre-trained model.
""")

uploaded_file = st.file_uploader("Choose an Excel file with your import data", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Load the data, ensuring that all columns are read as strings initially
        data = pd.read_excel(uploaded_file, dtype=str)
        st.write("Data Uploaded Successfully!")

        st.write("Here is a preview of your data:")
        st.write(data.head())

        # Check for necessary columns
        required_columns = ['inspection date', 'arrival date']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
        else:
            # Ensure the necessary columns are of the correct type
            data['inspection date'] = pd.to_datetime(data['inspection date'], dayfirst=True, errors='coerce')
            data['arrival date'] = pd.to_datetime(data['arrival date'], dayfirst=True, errors='coerce')

            X_delay = preprocess(data)
            
            if X_delay is not None:
                X_delay_scaled = scaler.transform(X_delay)

                # Ensure the columns in X_delay match those seen by the model
                model_columns = scaler.get_feature_names_out()
                for col in model_columns:
                    if col not in X_delay.columns:
                        X_delay[col] = 0
                X_delay = X_delay[model_columns]

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
