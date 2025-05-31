# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Title
st.title("🏆 MPL Season 15 - Team Prediction")
st.write("Enter player stats to predict the team they belong to.")

# Input fields
inputs = {
    'Total Kills': st.slider("Total Kills", 0, 100, 10),
    'Total Deaths': st.slider("Total Deaths", 0, 100, 10),
    'Total Assists': st.slider("Total Assists", 0, 200, 20),
    'AVG Kills': st.slider("AVG Kills", 0.0, 10.0, 3.0),
    'AVG Deaths': st.slider("AVG Deaths", 0.0, 10.0, 3.0),
    'AVG Assists': st.slider("AVG Assists", 0.0, 15.0, 5.0),
    'KDA Ratio': st.slider("KDA Ratio", 0.0, 10.0, 3.0),
    'Kill Participation': st.slider("Kill Participation", 0.0, 1.0, 0.5)
}

user_input = pd.DataFrame([inputs])

# Train the model here (or load from file)
# You can replace this with joblib.load("model.pkl") if you saved the model before

# --- TEMPORARY TRAINING SETUP (For prototype) ---
df = pd.read_excel("MPL_DATA_SEASON15.xlsx")
player_df = df[df['Player'].notna()].dropna(subset=list(inputs.keys()) + ['Team'])

X = player_df[list(inputs.keys())]
y = player_df['Team']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
# --------------------------------------------------

# Predict
prediction = model.predict(user_input)[0]
st.success(f"Predicted Team: **{prediction}**")
