# import streamlit as st
# import pandas as pd
# import joblib
# import os

# # --- Config ---
# st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")

# # --- Light Theme CSS (white & black only) ---
# st.markdown("""
# <style>
# body {
#     background-color: #ffffff;
#     color: #000000;
# }
# label, .stSelectbox label, .stSlider label {
#     font-weight: bold;
# }
# .css-1cpxqw2, .css-1d391kg {
#     background-color: #ffffff;
# }
# </style>
# """, unsafe_allow_html=True)

# # --- Load model safely ---
# MODEL_PATH = "nipam_titanic_model.pkl"

# if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
#     st.error("❌ Model file is missing or corrupted. Please check 'nipam_titanic_model.pkl'.")
#     st.stop()

# try:
#     model = joblib.load(MODEL_PATH)
# except Exception as e:
#     st.error(f"❌ Error loading model: {e}")
#     st.stop()

# # --- Title ---
# st.title("🚢 Titanic Survival Prediction App")
# st.markdown("Enter passenger details below. Fields include placeholder values from real data.")

# # --- User Input with Placeholders ---
# sex = st.selectbox("Sex", ["Male", "Female"], index=0)
# age = st.slider("Age", min_value=1, max_value=80, value=29)
# fare = st.slider("Fare Paid ($)", min_value=0.0, max_value=500.0, value=32.20)
# pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
# embarked = st.selectbox("Embarked Port", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)
# sibsp = st.slider("Siblings/Spouses Aboard", min_value=0, max_value=5, value=1)
# parch = st.slider("Parents/Children Aboard", min_value=0, max_value=5, value=0)

# # --- Preprocess Input ---
# embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
# input_data = pd.DataFrame([{
#     "sex": 0 if sex == "Male" else 1,
#     "age": age,
#     "fare": fare,
#     "pclass": pclass,
#     "family_size": sibsp + parch,
#     "embarked": embarked_map[embarked],
#     "sibsp": sibsp,
#     "parch": parch
# }])

# # --- Prediction ---
# if st.button("🎯 Predict Survival"):
#     try:
#         proba = model.predict_proba(input_data)[0][1]
#         threshold = 0.40
#         result = "✅ Survived" if proba >= threshold else "❌ Not Survived"
#         st.success(f"🧠 **Survival Probability:** `{proba:.2%}`")
#         st.markdown(f"### 🎯 Prediction: **{result}**")
#     except Exception as e:
#         st.error(f"❌ Error during prediction: {e}")


import streamlit as st
import pandas as pd
import joblib
import os

# --- Page config ---
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")

# --- Light Theme CSS ---
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

# --- Language selection ---
language = st.selectbox("🌐 Select Language / ભાષા પસંદ કરો / भाषा चुनें", ["English", "ગુજરાતી", "हिन्दी"])

# --- Language Texts ---
texts = {
    "English": {
        "title": "🚢 Titanic Survival Prediction App",
        "subtitle": "Enter passenger details below. Fields include placeholder values from real data.",
        "sex": "Sex",
        "age": "Age",
        "fare": "Fare Paid ($)",
        "pclass": "Passenger Class",
        "embarked": "Embarked Port",
        "sibsp": "Siblings/Spouses Aboard",
        "parch": "Parents/Children Aboard",
        "predict": "🎯 Predict Survival",
        "survived": "✅ Survived",
        "not_survived": "❌ Not Survived",
        "prob": "🧠 Survival Probability:",
        "prediction": "🎯 Prediction"
    },
    "ગુજરાતી": {
        "title": "🚢 ટાઇટેનિક જીવતા બચવાનો અનુમાન એપ",
        "subtitle": "નીચે મુસાફર વિગતો દાખલ કરો. ફીલ્ડમાં અસલી ડેટાના પ્રમાણ તરીકે મૂલ્યો છે.",
        "sex": "લિંગ",
        "age": "ઉંમર",
        "fare": "ચૂકવેલ ભાડું ($)",
        "pclass": "મુસાફરી વર્ગ",
        "embarked": "ચઢ્યા હોય તે પોર્ટ",
        "sibsp": "સગા/પતિ-પત્ની સાથે",
        "parch": "માતા-પિતા/બાળકો સાથે",
        "predict": "🎯 જીવતા બચવાનો અનુમાન કરો",
        "survived": "✅ બચ્યા છે",
        "not_survived": "❌ ન બચ્યા",
        "prob": "🧠 જીવતા બચવાની શક્યતા:",
        "prediction": "🎯 અનુમાન"
    },
    "हिन्दी": {
        "title": "🚢 टाइटैनिक जीवन रक्षा भविष्यवाणी ऐप",
        "subtitle": "नीचे यात्री की जानकारी दर्ज करें। फ़ील्ड में असली डेटा से उदाहरण मान हैं।",
        "sex": "लिंग",
        "age": "आयु",
        "fare": "भुगतान किया गया किराया ($)",
        "pclass": "यात्रा वर्ग",
        "embarked": "जहाँ से चढ़े",
        "sibsp": "सगे भाई-बहन/पति-पत्नी साथ",
        "parch": "माता-पिता/बच्चे साथ",
        "predict": "🎯 जीवन रक्षा का पूर्वानुमान लगाएं",
        "survived": "✅ बच गए",
        "not_survived": "❌ नहीं बच सके",
        "prob": "🧠 बचने की संभावना:",
        "prediction": "🎯 पूर्वानुमान"
    }
}

# --- Load Model ---
MODEL_PATH = "nipam_titanic_model.pkl"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    st.error("❌ Model file is missing or corrupted. Please check 'nipam_titanic_model.pkl'.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Title and Subtitle ---
st.title(texts[language]["title"])
st.markdown(texts[language]["subtitle"])

# --- User Inputs ---
sex = st.selectbox(texts[language]["sex"], 
                   ["Male", "Female"] if language == "English" 
                   else ["પુરુષ", "સ્ત્રી"] if language == "ગુજરાતી" 
                   else ["पुरुष", "महिला"])

age = st.slider(texts[language]["age"], 1, 80, 29)
fare = st.slider(texts[language]["fare"], 0.0, 500.0, 32.20)
pclass = st.selectbox(texts[language]["pclass"], [1, 2, 3])

embarked_options = {
    "English": ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
    "ગુજરાતી": ["શેર્બોર્ગ (C)", "ક્વીનસ્ટાઉન (Q)", "સાઉથહેમ્પ્ટન (S)"],
    "हिन्दी": ["शेरबर्ग (C)", "क्वीनस्टाउन (Q)", "साउथहैम्प्टन (S)"]
}
embarked = st.selectbox(texts[language]["embarked"], embarked_options[language])

sibsp = st.slider(texts[language]["sibsp"], 0, 5, 1)
parch = st.slider(texts[language]["parch"], 0, 5, 0)

# --- Preprocessing ---
sex_val = 0 if sex in ["Male", "પુરુષ", "पुरुष"] else 1

embarked_map = {
    "English": {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2},
    "ગુજરાતી": {"શેર્બોર્ગ (C)": 0, "ક્વીનસ્ટાઉન (Q)": 1, "સાઉથહેમ્પ્ટન (S)": 2},
    "हिन्दी": {"शेरबर्ग (C)": 0, "क्वीनस्टाउन (Q)": 1, "साउथहैम्प्टन (S)": 2}
}
embarked_val = embarked_map[language][embarked]

input_data = pd.DataFrame([{
    "sex": sex_val,
    "age": age,
    "fare": fare,
    "pclass": pclass,
    "family_size": sibsp + parch,
    "embarked": embarked_val,
    "sibsp": sibsp,
    "parch": parch
}])

# --- Prediction ---
if st.button(texts[language]["predict"]):
    try:
        proba = model.predict_proba(input_data)[0][1]
        threshold = 0.40
        result = texts[language]["survived"] if proba >= threshold else texts[language]["not_survived"]
        st.success(f"{texts[language]['prob']} `{proba:.2%}`")
        st.markdown(f"### {texts[language]['prediction']}: **{result}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
