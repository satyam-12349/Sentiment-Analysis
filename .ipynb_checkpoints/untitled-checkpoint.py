import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vect.pkl")

# App Title
st.title("📝 Sentiment Analysis App")
st.write("Enter any sentence below.")

# Input text
text = st.text_area("Enter Text")

# Prediction
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Convert text to vector
        text_vector = vectorizer.transform([text])

        # Predict
        prediction = model.predict(text_vector)

        # Display result
        if prediction[0] == 1 or prediction[0] == "Positive":
            st.success("😊 Positive Sentiment")
        elif prediction[0] == 0 or prediction[0] == "Negative":
            st.error("😞 Negative Sentiment")
        else:
            st.info(f"Prediction: {prediction[0]}")