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
    # Remove specified columns
    data = data.drop(columns=['container number', 'x', 'color code', 'container'], errors='ignore')

    # Rename specified columns
    data = data.rename(columns={
        'container type': 'container_type',
        'custom fees': 'custom_fees',
        'inspection date': 'inspection_date',
        'country of origin': 'country_of_origin',
        'arrival date': 'arrival_date',
    })

    # Convert date columns to datetime format with error handling
    data['inspection_date'] = pd.to_datetime(data['inspection_date'], dayfirst=True, errors='coerce')
    data['arrival_date'] = pd.to_datetime(data['arrival_date'], dayfirst=True, errors='coerce')

    # Function to generate a random date between 2 to 20 days after 'arrival date'
    def random_date(arrival_date):
        if pd.isna(arrival_date):
            return np.nan
        days_to_add = np.random.randint(2, 21)
        return arrival_date + pd.DateOffset(days=days_to_add)

    # Apply the function to fill null 'inspection date'
    data['inspection_date'] = data.apply(
        lambda row: random_date(row['arrival_date']) if pd.isna(row['inspection_date']) else row['inspection_date'],
        axis=1
    )

    # Drop rows with any null values
    data = data.dropna()

    # Cleaning non-numeric characters and converting to float
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

    # Calculate 'custom_fees_usd' taking care to handle NaN values in 'exchange_rate'
    data['custom_fees_usd'] = data.apply(lambda x: x['custom_fees'] / x['exchange_rate'] if pd.notna(x['exchange_rate']) else np.nan, axis=1)

    data['delay_days'] = (data['inspection_date'] - data['arrival_date']).dt.days
    data = data[data['delay_days'] >= 0]
    data['arrival_weekday'] = data['arrival_date'].dt.weekday
    data['arrival_month'] = data['arrival_date'].dt.month

    data2 = data[['arrival_weekday', 'arrival_month', 'country_of_origin', 'container_type', 'delay_days']]

    # Remove rows where any cell in that row is NA
    data2 = data2.dropna()
    X_delay = data2[['arrival_weekday', 'arrival_month', 'country_of_origin', 'container_type']]
    y_delay = data2['delay_days']

    # Encode categorical features
    encoded_features = pd.get_dummies(X_delay[['container_type', 'country_of_origin']], prefix=['container_type', 'country_of_origin'])

    # Drop original columns from X_delay and concatenate with the new one-hot encoded columns
    X_delay = pd.concat([X_delay.drop(['container_type', 'country_of_origin'], axis=1), encoded_features], axis=1)

    # Cyclic Encoding for Weekday and month
    X_delay['arrival_weekday_sin'] = np.sin(2 * np.pi * X_delay['arrival_weekday'] / 7)
    X_delay['arrival_weekday_cos'] = np.cos(2 * np.pi * X_delay['arrival_weekday'] / 7)
    X_delay['arrival_month_sin'] = np.sin(2 * np.pi * X_delay['arrival_month'] / 12)
    X_delay['arrival_month_cos'] = np.cos(2 * np.pi * X_delay['arrival_month'] / 12)

    X_delay = X_delay.drop(['arrival_weekday', 'arrival_month'], axis=1)

    return X_delay, y_delay

# Streamlit interface
st.title('Import Delay Prediction')

st.write("""
This application allows you to upload your import data, preprocess it, and predict the delay days using a pre-trained model.
""")

uploaded_file = st.file_uploader("Choose an Excel file with your import data", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Load the data
        data = pd.read_excel(uploaded_file)
        st.write("Data Uploaded Successfully!")

        st.write("Here is a preview of your data:")
        st.write(data.head())

        # Preprocess the data
        X_delay, y_delay = preprocess(data)

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
