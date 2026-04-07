import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load files
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

# Page config
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

# Title
st.title("💳 UPI Fraud Detection System")

# Accuracy
st.subheader("📊 Model Accuracy")
st.success(f"Accuracy: {accuracy:.2f}")

# =========================
# 🔹 MANUAL INPUT SECTION
# =========================
st.sidebar.header("🔍 Manual Transaction Input")

amount = st.sidebar.number_input("Amount", 0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance", 0.0)
newbalanceOrig = st.sidebar.number_input("New Balance", 0.0)

if st.sidebar.button("Predict Fraud"):
    input_data = pd.DataFrame({
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig]
    })

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=columns, fill_value=0)

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.sidebar.error(f"⚠️ Fraud Detected! Probability: {prob:.2f}")
    else:
        st.sidebar.success(f"✅ Safe Transaction. Probability: {prob:.2f}")

# =========================
# 🔹 CSV UPLOAD SECTION
# =========================
st.subheader("📁 Upload transaction CSV file")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(data.head())

    # Drop unnecessary columns
    if 'nameOrig' in data.columns:
        data = data.drop(['nameOrig', 'nameDest'], axis=1)

    # Convert categorical
    data = pd.get_dummies(data)

    # Align columns
    data = data.reindex(columns=columns, fill_value=0)

    # Predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    data['Fraud Prediction'] = predictions
    data['Fraud Probability'] = probabilities

    st.subheader("🔍 Prediction Results")
    st.dataframe(data)

    # Summary
    total = len(data)
    fraud = (predictions == 1).sum()
    normal = (predictions == 0).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Fraud Detected", fraud)
    col3.metric("Normal Transactions", normal)

    # Alert
    if fraud > 0:
        st.error(f"⚠️ {fraud} Fraud Transactions Detected!")
    else:
        st.success("✅ No Fraud Detected")

    # Chart
    st.subheader("📊 Fraud vs Normal")
    st.bar_chart(data['Fraud Prediction'].value_counts())