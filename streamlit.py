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
    'vendor_id': '',
    'client_key': '',
    'appointment_number': '',
    'plan_option1_deductible': 5000,
    'plan_option1_maximum_out_of_pocket': 10000
}

# Create a function to get user input values as a numpy array
def get_input_array():
    input_dict = {}
    for field in input_fields:
        if field.startswith('plan_option'):
            input_dict[field] = st.slider(field, *plan_option_range, value=default_values[field])
        else:
            input_dict[field] = st.text_input(field, default_values[field])
    return np.array([list(input_dict.values())])

# Create the Streamlit app
st.title('Cancellation Flag Predictor')

# Get user input values
input_array = get_input_array()

# Create a submit button to trigger the prediction
if st.button('Predict'):
    # Make predictions using the machine learning model
    prediction = model.predict(input_array)

    # Display the predicted cancellation flag value
    st.write('Predicted Cancellation Flag:', prediction[0])
