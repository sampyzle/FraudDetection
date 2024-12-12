import streamlit as st
import pandas as pd
import joblib
import json

# Load the trained model
model = joblib.load('rf_model.pkl')

# Load the column order from the JSON file
with open('column_order.json', 'r') as file:
    column_order = json.load(file)

# App title
st.title("Credit Card Fraud Detection")

# Input form for a single transaction
st.header("Enter Transaction Details")
input_data = {}
for column in column_order:
    if column == "id":
        input_data[column] = st.number_input(f"{column} (Transaction ID)", value=1, step=1)
    elif column == "Amount":
        input_data[column] = st.number_input(f"{column} (Transaction Amount)", value=0.0, step=0.1)
    else:
        input_data[column] = st.number_input(f"{column} (Feature Value)", value=0.0)

# Prediction button
if st.button("Predict Fraud"):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=column_order)

    # Make prediction using the model
    prediction = model.predict(input_df)
    result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

    # Display prediction result
    st.subheader("Prediction Result")
    st.success(f"The transaction is **{result}**.")

# Batch prediction for multiple transactions
st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with transaction details", type=["csv"])
if uploaded_file is not None:
    # Read uploaded CSV file
    batch_data = pd.read_csv(uploaded_file)

    # Ensure uploaded file has the correct columns
    if set(column_order).issubset(batch_data.columns):
        # Reorder columns to match training order
        batch_data = batch_data[column_order]

        # Make predictions
        batch_predictions = model.predict(batch_data)

        # Display batch predictions
        st.write("Batch Predictions:")
        batch_data["Prediction"] = ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in batch_predictions]
        st.write(batch_data)
    else:
        st.error("Uploaded file does not match the required column structure.")
