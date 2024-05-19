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
def preprocess(data, expected_features):
    st.write("Step: Renaming columns")
    # Rename specified columns
    data = data.rename(columns={
        'container type': 'container_type',
        'custom fees': 'custom_fees',
        'inspection date': 'inspection_date',
        'country of origin': 'country_of_origin',
        'arrival date': 'arrival_date',
    })
    st.write(data.columns)

    st.write("Step: Dropping unnecessary columns")
    # Remove specified columns
    data = data.drop(columns=['container number', 'x', 'color code', 'container'], errors='ignore')
    st.write(data.columns)

    st.write("Step: Converting date columns to datetime")
    # Convert date columns to datetime format with error handling
    data['inspection_date'] = pd.to_datetime(data['inspection_date'], dayfirst=True, errors='coerce')
    data['arrival_date'] = pd.to_datetime(data['arrival_date'], dayfirst=True, errors='coerce')
    st.write(data[['inspection_date', 'arrival_date']].head())

    st.write("Step: Filling null 'inspection_date'")
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
    st.write(data[['inspection_date', 'arrival_date']].head())

    st.write("Step: Dropping rows with any null values")
    # Drop rows with any null values
    data = data.dropna()
    st.write(f"Data shape after dropping nulls: {data.shape}")

    st.write("Step: Cleaning 'custom_fees' column")
    # Cleaning non-numeric characters and converting to float
    data['custom_fees'] = data['custom_fees'].replace('[^\d.]', '', regex=True)
    data['custom_fees'] = pd.to_numeric(data['custom_fees'], errors='coerce')
    st.write(data['custom_fees'].head())

    st.write("Step: Calculating 'exchange_rate'")
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
    st.write(data['exchange_rate'].head())

    st.write("Step: Calculating 'custom_fees_usd'")
    # Calculate 'custom_fees_usd' taking care to handle NaN values in 'exchange_rate'
    data['custom_fees_usd'] = data.apply(lambda x: x['custom_fees'] / x['exchange_rate'] if pd.notna(x['exchange_rate']) else np.nan, axis=1)
    st.write(data[['custom_fees', 'exchange_rate', 'custom_fees_usd']].head())

    st.write("Step: Calculating 'delay_days'")
    data['delay_days'] = (data['inspection_date'] - data['arrival_date']).dt.days
    data = data[data['delay_days'] >= 0]
    st.write(data[['delay_days']].head())

    st.write("Step: Adding 'arrival_weekday' and 'arrival_month'")
    data['arrival_weekday'] = data['arrival_date'].dt.weekday
    data['arrival_month'] = data['arrival_date'].dt.month
    st.write(data[['arrival_weekday', 'arrival_month']].head())

    data2 = data[['arrival_weekday', 'arrival_month', 'country_of_origin', 'container_type', 'delay_days']]
    st.write("Data2 shape: ", data2.shape)

    st.write("Step: Dropping rows where any cell in that row is NA")
    # Remove rows where any cell in that row is NA
    data2 = data2.dropna()
    st.write(f"Data2 shape after dropping NA: {data2.shape}")

    X_delay = data2[['arrival_weekday', 'arrival_month', 'country_of_origin', 'container_type']]
    y_delay = data2['delay_days']
    st.write("X_delay and y_delay shapes: ", X_delay.shape, y_delay.shape)

    st.write("Step: Encoding categorical features")
    # Encode categorical features
    encoded_features = pd.get_dummies(X_delay[['container_type', 'country_of_origin']], prefix=['container_type', 'country_of_origin'])
    st.write(f"Encoded features shape: {encoded_features.shape}")
    st.write(encoded_features.head())

    st.write("Step: Dropping original columns and concatenating encoded features")
    # Drop original columns from X_delay and concatenate with the new one-hot encoded columns
    X_delay = pd.concat([X_delay.drop(['container_type', 'country_of_origin'], axis=1), encoded_features], axis=1)
    st.write(X_delay.head())

    st.write("Step: Adding cyclic encoding for 'arrival_weekday' and 'arrival_month'")
    # Cyclic Encoding for Weekday and month
    X_delay['arrival_weekday_sin'] = np.sin(2 * np.pi * X_delay['arrival_weekday'] / 7)
    X_delay['arrival_weekday_cos'] = np.cos(2 * np.pi * X_delay['arrival_weekday'] / 7)
    X_delay['arrival_month_sin'] = np.sin(2 * np.pi * X_delay['arrival_month'] / 12)
    X_delay['arrival_month_cos'] = np.cos(2 * np.pi * X_delay['arrival_month'] / 12)

    X_delay = X_delay.drop(['arrival_weekday', 'arrival_month'], axis=1)
    st.write(X_delay.head())

    st.write("Step: Ensuring all expected features are present in X_delay")
    # Ensure all expected features are present in X_delay
    for feature in expected_features:
        if feature not in X_delay.columns:
            X_delay[feature] = 0

    # Reorder columns to match expected features
    X_delay = X_delay[expected_features]
    st.write("Final X_delay shape: ", X_delay.shape)
    st.write(X_delay.head())

    return X_delay, y_delay, data2

# Streamlit interface
st.title('Import Delay Prediction')

st.write("""
This application allows you to upload your import data, preprocess it, and predict the delay days using a pre-trained model.
""")

uploaded_file = st.file_uploader("Choose an Excel file with your import data", type=['xlsx'])

if uploaded_file is not None:
    try:
        st.write("Step: Loading the data")
        # Load the data
        data = pd.read_excel(uploaded_file, dtype=str)
        st.write("Data Uploaded Successfully!")

        st.write("Step: Standardizing column names")
        # Standardize column names immediately after reading the data
        data = data.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
        st.write("Standardized column names: ", data.columns)

        st.write("Here is a preview of your data:")
        st.write(data.head())

        st.write("Step: Checking for necessary columns")
        # Check for necessary columns
        required_columns = ['inspection_date', 'arrival_date', 'custom_fees', 'country_of_origin', 'container_type']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
        else:
            st.write("Step: Getting expected feature names from the scaler")
            # Get the expected feature names from the scaler
            expected_features = scaler.get_feature_names_out()
            st.write("Expected features: ", expected_features)

            st.write("Step: Preprocessing the data")
            # Preprocess the data
            X_delay, y_delay = preprocess(data, expected_features)

            if X_delay is not None:
                st.write("Step: Scaling the data")
                X_delay_scaled = scaler.transform(X_delay)
                st.write(f"Scaled data shape: ", X_delay_scaled.shape)
                st.write(X_delay_scaled[:5])  # Display the first 5 rows of scaled data

                st.write("Step: Making predictions")
                predictions = model.predict(X_delay_scaled)
                st.write(f"Predictions made: {predictions[:5]}")  # Display the first 5 predictions

                st.write("Step: Adding predictions to the dataframe")
                data['predicted_delay_days'] = predictions

                st.write("Step: Preparing the output file")
                st.write(data.head())

                # Ensure only the necessary columns are included in the output file
                output_columns = data.columns.tolist()
                data.to_excel('predicted_imports.xlsx', index=False, columns=output_columns)
                st.download_button(
                    label="Download predictions as Excel",
                    data=data.to_excel(index=False, columns=output_columns),
                    file_name='predicted_imports.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an Excel file to proceed.")
