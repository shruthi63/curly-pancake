import urllib.request
import streamlit as st
import pandas as pd
import joblib

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

# Create a function to get user input values as a DataFrame
def get_input_df():
    input_dict = {}
    for field in input_fields:
        if field.startswith('plan_option'):
            input_dict[field] = st.slider(f"{field} ({default_values[field]})", *plan_option_range, value=default_values[field])
        else:
            input_dict[field] = st.text_input(field, default_values[field])
    input_df = pd.DataFrame([input_dict])
    return input_df

# Create the Streamlit app
st.title('Cancellation Flag Predictor')

# Get user input values
if st.button('Predict'):
    input_df = get_input_df()

    # Make predictions using the machine learning model
    prediction = model.predict(input_df)

    # Display the predicted cancellation flag value
    st.write('Predicted Cancellation Flag:', prediction[0])

# Add a test example to prefill the input fields
st.write('Test Example:', 'vendor_id=29177, client_key=1834049, appointment_number=2758983, plan_option1_deductible=3224876, plan_option1_maximum_out_of_pocket=0.0, 1500.0')
