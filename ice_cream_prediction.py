import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from pymongo import MongoClient

# MongoDB connection
MONGO_URI = "mongodb+srv://Bipin:bipin1234@cluster0.iwsqoru.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["ice_cream_db"]
collection = db["predictions"]

# Function to save the prediction to MongoDB
def save_prediction(temp, prediction):
    collection.insert_one({
        "temperature": temp,
        "prediction": float(prediction)
    })

# Function to load the trained model
def load_model():
    with open("ice_cream_sells_prediction.pkl", 'rb') as file:
        ploy_model = pickle.load(file)
    return ploy_model

# Function to make predictions using the model
def predict_data(temperature):
    ploy_model = load_model()
    input_data = np.array([[temperature]])  # input must be 2D
    prediction = ploy_model.predict(input_data)
    return prediction[0]  # return scalar

# Main Streamlit app
def main():
    st.title("Ice Cream Sales Predictor Based on Temperature")
    st.write("Enter the Temperature to Predict the Ice-Cream Sells")

    temperature = st.number_input("Temperature")

    if st.button("Predict the Sells"):
        # Predict the sales
        prediction = predict_data(temperature)

        # Save the prediction to MongoDB
        save_prediction(temperature, prediction)

        # Display the prediction result
        st.success(f"Your Prediction Result is {prediction:.2f} units")

if __name__ == "__main__":
    main()
