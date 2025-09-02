import streamlit as st, joblib

model = joblib.load("spam_model.joblib")

st.title("SMS Spam Detector")
msg = st.text_area("Enter message:")

if st.button("Predict"):
    prob = model.predict_proba([msg])[0,1]
    label = "SPAM" if prob >= 0.5 else "HAM"
    st.write(f"Prediction: {label} (Spam prob: {prob:.2f})")
