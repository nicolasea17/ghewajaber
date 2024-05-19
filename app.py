import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import matplotlib.pyplot as plt

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
def preprocess(data, expected_features):
    # Rename specified columns
    data = data.rename(columns={
        'container type': 'container_type',
        'custom fees': 'custom_fees',
        'inspection date': 'inspection_date',
        'country of origin': 'country_of_origin',
        'arrival date': 'arrival_date',
    })

    # Remove specified columns
    data = data.drop(columns=['container number', 'x', 'color code', 'container'], errors='ignore')

    # Convert date columns to datetime format with error handling
    data['inspection_date'] = pd.to_datetime(data['inspection_date'], dayfirst=True, errors='coerce')
    data['arrival_date'] = pd.to_datetime(data['arrival_date'], dayfirst=True, errors='coerce')

    # Function to generate a random date between 2 to 20 days after 'arrival_date'
    def random_date(arrival_date):
        if pd.isna(arrival_date):
            return np.nan
        days_to_add = np.random.randint(2, 21)
        return arrival_date + pd.DateOffset(days=days_to_add)

    # Apply the function to fill null 'inspection_date'
    data['inspection_date'] = data.apply(
        lambda row: random_date(row['arrival_date']) if pd.isna(row['inspection_date']) else row['inspection_date'],
        axis=1
    )

    # Drop rows with any null values
    data = data.dropna()

    # Cleaning non-numeric characters and converting to float
    data['custom_fees'] = data['custom_fees'].replace('[^\d.]', '', regex=True)
    data['custom_fees'] = pd.to_numeric(data['custom_fees'], errors='coerce')

    # Determine exchange rate based on arrival date
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

    # Calculate 'custom_fees_usd'
    data['custom_fees_usd'] = data.apply(lambda x: x['custom_fees'] / x['exchange_rate'] if pd.notna(x['exchange_rate']) else np.nan, axis=1)

    # Add 'delay_days' column
    data['delay_days'] = (data['inspection_date'] - data['arrival_date']).dt.days
    data = data[data['delay_days'] >= 0]

    # Add cyclic encoding for 'arrival_weekday' and 'arrival_month'
    data['arrival_weekday'] = data['arrival_date'].dt.weekday
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_weekday_sin'] = np.sin(2 * np.pi * data['arrival_weekday'] / 7)
    data['arrival_weekday_cos'] = np.cos(2 * np.pi * data['arrival_weekday'] / 7)
    data['arrival_month_sin'] = np.sin(2 * np.pi * data['arrival_month'] / 12)
    data['arrival_month_cos'] = np.cos(2 * np.pi * data['arrival_month'] / 12)

    # Encode categorical features
    X_delay = data[['arrival_weekday_sin', 'arrival_weekday_cos', 'arrival_month_sin', 'arrival_month_cos', 'country_of_origin', 'container_type']]
    y_delay = data['delay_days']
    encoded_features = pd.get_dummies(X_delay[['container_type', 'country_of_origin']], prefix=['container_type', 'country_of_origin'])
    X_delay = pd.concat([X_delay.drop(['container_type', 'country_of_origin'], axis=1), encoded_features], axis=1)

    # Ensure all expected features are present in X_delay
    for feature in expected_features:
        if feature not in X_delay.columns:
            X_delay[feature] = 0

    # Reorder columns to match expected features
    X_delay = X_delay[expected_features]

    return X_delay, data

# Streamlit interface
st.title('Import Delay Prediction')

st.write("""
This application allows you to upload your import data, preprocess it, and predict the delay days using a pre-trained model.
""")

uploaded_file = st.file_uploader("Choose an Excel file with your import data", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Load the data
        data = pd.read_excel(uploaded_file, dtype=str)

        # Standardize column names immediately after reading the data
        data = data.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))

        # Check for necessary columns
        required_columns = ['inspection_date', 'arrival_date', 'custom_fees', 'country_of_origin', 'container_type']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
        else:
            # Get the expected feature names from the scaler
            expected_features = scaler.get_feature_names_out()

            # Preprocess the data
            X_delay, data = preprocess(data, expected_features)

            if X_delay is not None:
                # Scale the data
                X_delay_scaled = scaler.transform(X_delay)

                # Make predictions
                predictions = model.predict(X_delay_scaled)

                # Add predictions to the dataframe
                data['predicted_delay_days'] = predictions

                # Display predictions
                st.write("Here are the predictions for your data:")
                st.write(data.head())

                # Display statistics
                st.write("Statistics of predicted delay days:")
                st.write(data['predicted_delay_days'].describe())

                # Prepare file for download
                output_file = 'data_with_predictions.xlsx'
                data.to_excel(output_file, index=False)

                with open(output_file, 'rb') as f:
                    st.download_button('Download Predictions', f, file_name=output_file)

    except Exception as e:
        st.error(f"An error occurred: {e}")
