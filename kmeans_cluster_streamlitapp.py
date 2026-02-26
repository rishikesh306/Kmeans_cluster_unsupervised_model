import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Customer Segmentation using K-Means")

st.write("Enter customer details to find cluster")

# User inputs
# age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Annual Income", min_value=10000, max_value=200000)
spending = st.slider("Spending Score", 1, 100)

# Cluster meaning mapping
cluster_meaning = {
    0: "Low / Medium Income, High Spending",
    1: "High Income, Low Spending",
    2: "Low Income, Low Spending"
}

if st.button("Find Cluster"):
    data = np.array([[income, spending]])
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]
    st.success(f" 🧩 Cluster: {cluster} -> {cluster_meaning[cluster]}")
    