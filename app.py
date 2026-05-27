
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("aml_randomforest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AML Detection System", layout="wide")

st.title("💳 Anti-Money Laundering (AML) Detection")
st.write("Predict whether a transaction is suspicious (SAR likelihood).")

# -----------------------------
# Sidebar Inputs (User-Friendly)
# -----------------------------
st.sidebar.header("Enter Transaction Details")

# Numeric Inputs
transaction_amount = st.sidebar.number_input("💰 Transaction Amount", min_value=0.0, value=1000.0)
hour = st.sidebar.slider("⏰ Transaction Hour", 0, 23, 12)

# -----------------------------
# Mapping Dictionaries
# -----------------------------
risk_map = {"Low": 0, "Medium": 1, "High": 2}
customer_type_map = {"Corporate": 0, "Individual": 1}

sector_map = {
    "Retail": 0,
    "Mining": 1,
    "Car Dealing": 2,
    "Agriculture": 3,
    "Fuel": 4,
    "Transport": 5
}

transaction_type_map = {
    "Deposit": 0,
    "Withdrawal": 1,
    "POS": 2,
    "Transfer": 3,
    "Mobile": 4
}

channel_map = {
    "Agent": 0,
    "ATM": 1,
    "Branch": 2,
    "Internet Banking": 3,
    "Mobile App": 4
}

city_map = {
    "Harare": 0,
    "Bulawayo": 1,
    "Gweru": 2,
    "Mutare": 3,
    "Kadoma": 4
}

# -----------------------------
# User Selections (Friendly UI)
# -----------------------------
customer_risk = st.sidebar.selectbox("⚠️ Customer Risk Rating", list(risk_map.keys()))
customer_type_ui = st.sidebar.selectbox("👤 Customer Type", list(customer_type_map.keys()))
sector_ui = st.sidebar.selectbox("🏭 Sector", list(sector_map.keys()))
transaction_type_ui = st.sidebar.selectbox("💳 Transaction Type", list(transaction_type_map.keys()))
channel_ui = st.sidebar.selectbox("📱 Channel", list(channel_map.keys()))
city_ui = st.sidebar.selectbox("📍 Origin City", list(city_map.keys()))

# -----------------------------
# Convert to Numeric (Model Input)
# -----------------------------
customer_risk_rating = risk_map[customer_risk]
customer_type = customer_type_map[customer_type_ui]
sector = sector_map[sector_ui]
transaction_type = transaction_type_map[transaction_type_ui]
channel = channel_map[channel_ui]
origin_city = city_map[city_ui]

# -----------------------------
# Feature Vector (CORRECT ORDER)
# -----------------------------
feature_order = [
    'transaction_amount',
    'transaction_type',
    'channel',
    'origin_city',
    'customer_risk_rating',
    'customer_type',
    'Sector',
    'hour'
]

input_df = pd.DataFrame([[
    transaction_amount,
    transaction_type,
    channel,
    origin_city,
    customer_risk_rating,
    customer_type,
    sector,
    hour
]], columns=feature_order)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Analyze Transaction"):

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("🚨 High Risk Transaction Detected (SAR Likely)")
    else:
        st.success("✅ Transaction Appears Normal")

    st.metric("📈 SAR Risk Probability", f"{probability:.2%}")

    # -----------------------------
    # Display Input Summary
    # -----------------------------
    st.subheader("🧾 Transaction Summary")

    st.write({
        "Transaction Amount": transaction_amount,
        "Transaction Hour": hour,
        "Risk Rating": customer_risk,
        "Customer Type": customer_type_ui,
        "Sector": sector_ui,
        "Transaction Type": transaction_type_ui,
        "Channel": channel_ui,
        "City": city_ui
    })

