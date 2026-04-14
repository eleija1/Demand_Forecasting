import streamlit as st
import pandas as pd
import pickle

st.title("📦 Store Sales Predictor")

# Load model
data = pickle.load(open("model_bundle.pkl", "rb"))
model = data["model"]
columns = data["columns"]

st.write("Enter product details:")

Item_MRP = st.number_input("Item MRP", value=100.0)
Item_Visibility = st.number_input("Item Visibility", value=0.05)
Outlet_Location_Type = st.selectbox("Outlet Location", [1, 2, 3])

# Create input dataframe
input_dict = {
    "Item_MRP": Item_MRP,
    "Item_Visibility": Item_Visibility,
    "Outlet_Location_Type": Outlet_Location_Type
}

input_df = pd.DataFrame([input_dict])

# Match training columns
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# Predict
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Sales: ${prediction[0]:,.2f}")