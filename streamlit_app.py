import streamlit as st
import pickle
import numpy as np

# Load the trained model
import requests
import io

# URL to the hosted model file
url = "https://drive.google.com/uc?id=1xeDav1T6ZhPahojCAhoKOr5iQZI559fi"

try:
    # Download the model file from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for HTTP issues

    # Load the model directly from the downloaded content
    model = pickle.load(io.BytesIO(response.content))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



# App title
st.title("California Housing Price Predictor")

st.write(
    """
    This app predicts **California Housing Prices** based on several features.
    Adjust the sliders below to input feature values and see the predicted price.
    """
)

# User input for features using sliders
MedInc = st.slider("Median Income", 0.0, 150000.0, 50000.0)
HouseAge = st.slider("House Age", 0, 52, 25)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 0, 40000, 1500)
AveOccup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 37.0)
Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

# Prepare the input for prediction
input_data = np.array([[MedInc/10000, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Prediction
prediction = model.predict(input_data)
st.subheader("Predicted Housing Price")
st.write(f"**${prediction[0]*100000:.2f}**")
