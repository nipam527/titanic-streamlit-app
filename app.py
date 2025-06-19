import streamlit as st
import pandas as pd
import joblib
import os

# --- Config ---
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")

# --- Light Theme CSS (white & black only) ---
st.markdown("""
<style>
body {
    background-color: #ffffff;
    color: #000000;
}
label, .stSelectbox label, .stSlider label {
    font-weight: bold;
}
.css-1cpxqw2, .css-1d391kg {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# --- Load model safely ---
MODEL_PATH = "nipam_titanic_model.pkl"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    st.error("❌ Model file is missing or corrupted. Please check 'nipam_titanic_model.pkl'.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Title ---
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details below. Fields include placeholder values from real data.")

# --- User Input with Placeholders ---
sex = st.selectbox("Sex", ["Male", "Female"], index=0)
age = st.slider("Age", min_value=1, max_value=80, value=29)
fare = st.slider("Fare Paid ($)", min_value=0.0, max_value=500.0, value=32.20)
pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
embarked = st.selectbox("Embarked Port", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)
sibsp = st.slider("Siblings/Spouses Aboard", min_value=0, max_value=5, value=1)
parch = st.slider("Parents/Children Aboard", min_value=0, max_value=5, value=0)

# --- Preprocess Input ---
embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
input_data = pd.DataFrame([{
    "sex": 0 if sex == "Male" else 1,
    "age": age,
    "fare": fare,
    "pclass": pclass,
    "family_size": sibsp + parch,
    "embarked": embarked_map[embarked],
    "sibsp": sibsp,
    "parch": parch
}])

# --- Prediction ---
if st.button("🎯 Predict Survival"):
    try:
        proba = model.predict_proba(input_data)[0][1]
        threshold = 0.40
        result = "✅ Survived" if proba >= threshold else "❌ Not Survived"
        st.success(f"🧠 **Survival Probability:** `{proba:.2%}`")
        st.markdown(f"### 🎯 Prediction: **{result}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
