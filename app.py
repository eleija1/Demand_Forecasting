import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Store Sales Predictor", page_icon="📦")

st.title("📦 Store Sales Predictor")
st.write("Enter the product and outlet details below.")

with open("model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
columns = bundle["columns"]


 #Creates a function that prepares user input for the model
def preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame: #def preprocess_input = Clean and prepare input data before prediction
    df = raw_df.copy()                                     #raw_df = the data coming from Streamlit form
 #df = raw_df.copy()   =     creates a copy of the input data. stores it in df
    
    

    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
        "low fat": "Low Fat",
        "LF": "Low Fat",
        "reg": "Regular"
    })

    df["Item_Type"] = df["Item_Type"].replace(
        {
            'Dairy':'Food',
            'Soft Drinks':'Drinks',
            'Meat':'Food',
            'Fruits and Vegetables':'Food',
            'Household':'Others',
            'Baking Goods':'Others',
            'Snack Foods':'Food',
            'Frozen Foods':'Food',
            'Breakfast':'Food',
            'Health and Hygiene':'Others',
            'Hard Drinks':'Drinks',
            'Canned':'Food',
            'Breads':'Food',
            'Starchy Foods':'Food',
            'Others':'Others',
            'Seafood':'Food'
        }
    )

    df["Outlet_Location_Type"] = df["Outlet_Location_Type"].replace(
        {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
    )

    df["cbr_Item_Visibility"] = np.cbrt(df["Item_Visibility"])

    category_cols = ["Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Type"]
    df = pd.get_dummies(df, columns=category_cols, drop_first=False)

    df = df.reindex(columns=columns, fill_value=0)
    return df

col1, col2 = st.columns(2)


#Create input boxes so the user can enter product information
#"Item Weight"	label shown to user
#min_value=0.0	cannot go below 0
#value=12.5	default value
#step=0.1	increments by 0.1


with col1:
    item_weight = st.number_input("Item Weight", min_value=0.0, value=12.5, step=0.1)
    item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
    item_visibility = st.number_input("Item Visibility", min_value=0.0, value=0.05, step=0.001, format="%.3f")
    item_type = st.selectbox(
        "Item Type",
        [
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
            "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
            "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
            "Breads", "Starchy Foods", "Others", "Seafood"])

 
 
    
with col2:
    item_mrp = st.number_input("Item MRP", min_value=0.0, value=150.0, step=1.0)
    outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_type = st.selectbox(
        "Outlet Type",
        ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
    )

if st.button("Predict Sales"):
    raw_input = pd.DataFrame([{
        "Item_Weight": item_weight,
        "Item_Fat_Content": item_fat_content,
        "Item_Visibility": item_visibility,
        "Item_Type": item_type,
        "Item_MRP": item_mrp,
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": outlet_location_type,
        "Outlet_Type": outlet_type
    }])

    processed_input = preprocess_input(raw_input)
    prediction = model.predict(processed_input)[0]

    st.success(f"Predicted Item Outlet Sales: ${prediction:,.2f}")
