import urllib.request
import streamlit as st
import pandas as pd
import joblib
import numpy as np

url = 'https://github.com/shruthi63/curly-pancake/blob/main/model.joblib?raw=true'
filename = 'model.joblib'
urllib.request.urlretrieve(url, filename)
model = joblib.load(filename)

# Define the input fields
input_fields = ['vendor_id', 'client_key', 'appointment_number', 'plan_option1_deductible', 'plan_option1_maximum_out_of_pocket']

# Define the range for the plan option input fields
plan_option_range = (0, 10000)

# Define the default values for the input fields
default_values = {
    'Vendor ID': '',
    'Client Key': '',
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
            'Vendor ID': '29177',
            'Client Key': '1834049',
            'Appointment Number': '2758983',
            'Plan Option 1 Deductible': 3224876,
            'Plan Option 1 Maximum Out of Pocket': 0.0
        }
        input_df = pd.DataFrame([input_dict])

    # Make predictions using the machine learning model
    prediction = model.predict(input_df)

    # Display the predicted cancellation flag value
    st.write('Predicted Cancellation Flag:', prediction[0])
