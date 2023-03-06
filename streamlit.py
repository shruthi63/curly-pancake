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

try:
    urllib.request.urlretrieve(url, filename)
    model2 = joblib.load(filename)
except:
    st.error("Failed to download model. Please check the URL or try again later.")
    st.stop()

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
    try:
        input_df = get_input_df()

        # If no input was provided, use the provided test example
        if input_df.empty:
            input_dict = {
                'auth_status': 'Claim Received',
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
        prediction = model2.predict(input_df)

        # Display the predicted cancellation flag value
        st.write('Predicted Cancellation Flag:', prediction[0])
    except:
        st.error("Failed to make predictions. Please check your inputs and try again.")
