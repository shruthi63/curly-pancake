import urllib.request
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

url = 'https://github.com/shruthi63/curly-pancake/blob/main/model2.joblib?raw=true'
filename = 'model2.joblib'
urllib.request.urlretrieve(url, filename)
model = joblib.load(filename)

# Define the input fields
input_fields = ['auth_status', 'member_health_plan_id', 'primary_cpt', 'vendor_id', 'plan_option1_coinsurance_member',
                'appointment_number', 'plan_option1_deductible', 'plan_option1_maximum_out_of_pocket']

# Define the range for the plan option input fields
plan_option_range = (0, 10000)

# Define the default values for the input fields
default_values = {
    'auth+status': '',
    'member_health_plan_id': '',
    'primary_cpt': '',
    'Vendor ID': '',
    'Plan Option 1 Coinsurance (Member)': 20,
    'Appointment Number': '',
    'Plan Option 1 Deductible': 5000,
    'Plan Option 1 Maximum Out of Pocket': 10000
}

# Create a function to get user input values as a DataFrame
def get_input_df():
    input_dict = {}
    for field in input_fields:
        if field.startswith('plan_option'):
            input_dict[field] = st.slider(field, *plan_option_range, value=default_values[field])
        elif field == 'auth_status':
            input_dict[field] = st.selectbox(field, ['Claim Received', 'Approved', 'Cancelled', 'Submitted For Cancellation', 'Progyny Cover Cost'], index=0)
        else:
            input_dict[field] = st.text_input(field, default_values[field])
    return pd.DataFrame([input_dict])

# Create the Streamlit app
st.title('Cancellation Flag Predictor')

# Get user input values when submit button is clicked
if st.button('Submit'):
    input_df = get_input_df()

    # If no input was provided, use the provided test example
    if input_df.empty:
        input_dict = {
            'auth+status': 'Claim Received',
            'member_health_plan_id': 'HP000001',
            'primary_cpt': 'A123',
            'Vendor ID': '29177',
            'Appointment Number': '2758983',
            'Plan Option 1 Coinsurance (Member)': 20,
            'Plan Option 1 Deductible': 3224876,
            'Plan Option 1 Maximum Out of Pocket': 0.0
        }
        input_df = pd.DataFrame([input_dict])

    # Make predictions using the machine learning model
    prediction = model.predict(input_df)

    # Display the predicted cancellation flag value
    st.write('Predicted Cancellation Flag:', prediction[0])
