import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("team_model.pkl")

st.title("🏆 MPL Team Champion Predictor")
st.write("Input team stats below to estimate if the team could win MPL S15")

inputs = {
    'Match point': st.number_input("Match Point", 0, 50, 20),
    'Net Game Win': st.number_input("Net Game Win", -10, 20, 5),
    'Kills': st.number_input("Kills", 0, 1000, 300),
    'Deaths': st.number_input("Deaths", 0, 1000, 250),
    'Assists': st.number_input("Assists", 0, 2000, 600),
    'Gold': st.number_input("Gold", 0, 1000000, 500000),
    'Damage': st.number_input("Damage", 0, 1000000, 450000),
    'Lord Kills': st.number_input("Lord Kills", 0, 20, 5),
    'Tortoise Kills': st.number_input("Tortoise Kills", 0, 30, 10),
    'Tower Destroy': st.number_input("Tower Destroy", 0, 50, 20),
}

input_df = pd.DataFrame([inputs])
prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.markdown(f"### 🔮 Prediction: {'🏆 Likely Champion' if prediction == 1 else '❌ Unlikely to Win'}")
st.markdown(f"Confidence: **{prob*100:.2f}%**")
