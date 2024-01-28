import streamlit as st
import numpy as np
import pickle
from geopy.distance import geodesic

# Load the trained model
file_path = 'delivery_time_predictor.pkl'
model = pickle.load(open(file_path, 'rb'))

# Title of the app
st.title('Food Delivery Time Prediction App')

# Getting user input for all required features
age = st.number_input('Enter the delivery person\'s age', min_value=18, max_value=100, value=25, step=1)
ratings = st.number_input('Enter the delivery person\'s ratings', min_value=0.0, max_value=5.0, value=4.5, step=0.1)

# Add input fields for other features as needed
restaurant_latitude = st.number_input('Enter the restaurant latitude', value=0.0, step=0.0001)
restaurant_longitude = st.number_input('Enter the restaurant longitude', value=0.0, step=0.0001)
delivery_latitude = st.number_input('Enter the delivery location latitude', value=0.0, step=0.0001)
delivery_longitude = st.number_input('Enter the delivery location longitude', value=0.0, step=0.0001)


# Calculate distance using geodesic if it's not a direct input
distance = geodesic((restaurant_latitude, restaurant_longitude), (delivery_latitude, delivery_longitude)).km

# When 'Predict' button is clicked
if st.button('Predict'):
    # Create a DataFrame or array from the input data
    input_data = np.array([[age, ratings, restaurant_latitude, restaurant_longitude,
                            delivery_latitude, delivery_longitude,
                             distance]])


    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'Estimated delivery time: {prediction[0]:.2f} minutes')

# Run the app with `streamlit run app.py` from your command line
