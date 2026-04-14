import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Store Sales Predictor", page_icon="📦")

st.title("📦 Store Sales Predictor")
st.write("Enter product and outlet details to predict Item Outlet Sales.")

# -----------------------------
# Load trained model bundle
# -----------------------------
with open("model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
columns = bundle["columns"]

# -----------------------------
# Helper preprocessing function
# Must match the notebook logic
# -----------------------------
def preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Standardize Item_Fat_Content
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
        "low fat": "Low Fat",
        "LF": "Low Fat",
        "reg": "Regular"
    })

    # Group Item_Type like notebook
    item_type_map = {
        "Dairy": "Food",
        "Soft Drinks": "Drinks",
        "Meat": "Food",
        "Fruits and Vegetables": "Food",
        "Household": "Others",
        "Baking Goods": "Others",
        "Snack Foods": "Food",
        "Frozen Foods": "Food",
        "Breakfast": "Food",
        "Health and Hygiene": "Others",
        "Hard Drinks": "Drinks",
        "Canned": "Food",
        "Breads": "Food",
        "Starchy Foods": "Food",
        "Others": "Others",
        "Seafood": "Food"
    }
    df["Item_Type"] = df["Item_Type"].replace(item_type_map)

    # Convert Outlet_Location_Type to numeric like notebook
    location_map = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
    df["Outlet_Location_Type"] = df["Outlet_Location_Type"].replace(location_map)

    # Feature engineering
    df["cbr_Item_Visibility"] = np.cbrt(df["Item_Visibility"])

    # One-hot encode same categorical columns as notebook
    category_cols = ["Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Type"]
    df_encoded = pd.get_dummies(df, columns=category_cols, drop_first=False)

    # Align to training columns
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    return df_encoded

# -----------------------------
# Input form
# -----------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    
with st.form("prediction_form"):
    st.subheader("Product Information")
    col1, col2 = st.columns(2)

    with col1:
        item_weight = st.number_input("Item Weight", min_value=0.0, value=12.5, step=0.1)
        item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        item_visibility = st.number_input("Item Visibility", min_value=0.0, value=0.05, step=0.001, format="%.3f")

    with col2:
        item_type = st.selectbox(
            "Item Type",
            [
                "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
                "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
                "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
                "Breads", "Starchy Foods", "Others", "Seafood"
            ]
        )
        item_mrp = st.number_input("Item MRP", min_value=0.0, value=150.0, step=1.0)

    st.subheader("Outlet Information")
    col3, col4 = st.columns(2)

    with col3:
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])

    with col4:
        outlet_type = st.selectbox(
            "Outlet Type",
            ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
        )

    col_a, col_b = st.columns(2)
    with col_a:
        submitted = st.form_submit_button("Predict Sales")
    with col_b:
        reset = st.form_submit_button("Clear Prediction")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
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

    model_input = preprocess_input(raw_input)
    st.session_state.prediction = model.predict(model_input)[0]

  if reset:
    st.session_state.prediction = None

if st.session_state.prediction is not None:
    st.success(f"Predicted Item Outlet Sales: ${st.session_state.prediction:,.2f}")
# -----------------------------
# Prediction
# -----------------------------
if submitted:
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

    model_input = preprocess_input(raw_input)
    prediction = model.predict(model_input)[0]

    st.success(f"Predicted Item Outlet Sales: ${prediction:,.2f}")

    with st.expander("Show processed input used by the model"):
        st.dataframe(model_input)
