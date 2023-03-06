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
    'auth_status': '',
    'member_health_plan_id': '',
    'primary_cpt': '',
    'vendor_id': '',
    'plan_option1_coinsurance_member': 20,
    'appointment_number': '',
    'plan_option1_deductible': 5000,
    'plan_option1_maximum_out_of_pocket': 10000
}

# Create a function to get user input values as a DataFrame
def get_input_df():
    input_dict = {}
    for i, field in enumerate(input_fields):
        if field.startswith('plan_option'):
            input_dict[field] = st.slider(field, *plan_option_range, value=default_values[field], key=f"{field}_{i}")
        elif field == 'auth_status':
            auth_status = st.selectbox(field, ['Claim Received', 'Approved', 'Cancelled', 'Submitted For Cancellation', 'Progyny Cover Cost'], index=0, key=f"{field}_{i}")
            # Map the auth_status value to a numerical value
            if auth_status == 'Cancelled':
                input_dict[field] = 1
            elif auth_status == 'Approved':
                input_dict[field] = 0
            elif auth_status == 'Claim Received':
                input_dict[field] = 2
            else:
                input_dict[field] = 3
        else:
            input_dict[field] = st.text_input(field, default_values[field], key=f"{field}_{i}")
    return pd.DataFrame([input_dict])

# Create the Streamlit app
st.title('Cancellation Flag Predictor')
input_df = get_input_df()
# Get user input values when submit button is clicked
if st.button('Submit'):
    input_df = get_input_df()

    # If no input was provided, use the provided test example
    if input_df.empty:
        input_dict = {
            'auth_status': 1,
            'member_health_plan_id': 'HP000001',
            'primary_cpt': 'A123',
            'vendor_id': '29177',
            'appointment_number': '2758983',
            'plan_option1_coinsurance_member': 20,
            'plan_option1_deductible': 3224876,
            'plan_option1_maximum_out_of_pocket': 0.0
        }
        input_df = pd.DataFrame([input_dict])

    # Make predictions using the machine learning model
    prediction = model.predict(input_df)

    # Display the predicted cancellation flag value
    st.write('Predicted Cancellation Flag:', prediction[0])
