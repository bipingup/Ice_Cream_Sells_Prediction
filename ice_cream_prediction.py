import streamlit as st
import numpy as np
import pickle 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Load the saved model
def load_model():
    with open("ice_cream_sells_prediction.pkl", 'rb') as file:
        ploy_model = pickle.load(file)
    return ploy_model

# Predict function
def predict_data(data):
    ploy_model = load_model()
    prediction = ploy_model.predict(data)
    return prediction

# Streamlit app
def main():
    st.title("🍦 Ice Cream Sales Predictor Based on Temperature")
    st.write("Enter the temperature (in °C) to predict expected ice cream sales (in units).")

    temperature = st.number_input("Enter Temperature (°C):", format="%.2f")

    if st.button("Predict the Sales"):
        user_data = np.array([[temperature]])  # Ensure correct shape
        prediction = predict_data(user_data)
        st.success(f"📈 Predicted Ice Cream Sales: {prediction[0]:.2f} units")

if __name__ == "__main__":
    main()
