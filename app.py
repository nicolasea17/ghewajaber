import streamlit as st
import pandas as pd
import joblib
import logging
import sklearn
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize login status in session state
if 'login_successful' not in st.session_state:
    st.session_state.login_successful = False

# Login Page
def login_page():
    st.title("Welcome to Incoding's Page")
    col1, col2 = st.columns(2)

    with col1:
        st.image('https://github.com/nicolasea17/Capstone_Project/blob/main/Incoding%20Picture.png?raw=true', width=250)

    with col2:
        st.image('https://github.com/nicolasea17/Capstone_Project/blob/main/OSB%20Picture.png?raw=true', width=250)

    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Sign In'):
        if username == 'admin' and password == '1234':
            st.session_state.login_successful = True
            st.experimental_rerun()
        else:
            st.error('Invalid credentials')

# Load model and label encoders safely
def load_resources():
    try:
        model = joblib.load('random_forest_hourly_rate_model.joblib')
        encoders = joblib.load('label_encoders.joblib')
        return model, encoders
    except Exception as e:
        st.error(f"Failed to load resources. Error: {e}")
        return None, None

# Prediction Page
def prediction_page():
    logging.info("Prediction page called.")
    st.markdown("<h1 style='text-align: center; font-size: 24px;'>Website Development Hourly Rate Prediction</h1>", unsafe_allow_html=True)
    col_image = st.columns([1, 2, 1])

    with col_image[1]:
        st.image('https://github.com/nicolasea17/Capstone_Project/blob/main/MachineLearning_PriceElasticity.png?raw=true', width=300)

    if model and label_encoders:
        with st.form("prediction_form"):
            st.header("Enter the details:")
            job_type = st.selectbox('Select Job Type', label_encoders['Job Type'].classes_)
            ex_level_demand = st.selectbox('Select Experience Level Demand', label_encoders['EX_level_demand'].classes_)
            industry = st.selectbox('Select Industry', label_encoders['Industry'].classes_)
            technical_tool = st.selectbox('Select Technical Tool', label_encoders['Technical_Tool'].classes_)
            competitive_level = st.selectbox('Select Competitive Level', label_encoders['Competitive Level'].classes_)
            client_country = st.selectbox('Select Client Country', label_encoders['Client_Country'].classes_)
            submitted = st.form_submit_button("Predict Hourly Rate")

        if submitted:
            # Encode the inputs using the loaded label encoders
            input_data = pd.DataFrame([[
                label_encoders['Job Type'].transform([job_type])[0],
                label_encoders['EX_level_demand'].transform([ex_level_demand])[0],
                label_encoders['Industry'].transform([industry])[0],
                label_encoders['Technical_Tool'].transform([technical_tool])[0],
                label_encoders['Competitive Level'].transform([competitive_level])[0],
                label_encoders['Client_Country'].transform([client_country])[0]
            ]], columns=['Job Type', 'EX_level_demand', 'Industry', 'Technical_Tool', 'Competitive Level', 'Client_Country'])

            try:
                # Predict
                prediction = model.predict(input_data)
                st.write(f"The predicted hourly rate is ${prediction[0]:.2f}")
            except Exception as e:
                st.error(f"An error occurred during prediction. Error: {e}")

# Main application logic
if not st.session_state.login_successful:
    login_page()
else:
    model, label_encoders = load_resources()
    prediction_page()
