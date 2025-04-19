import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("Fake News Detector")
st.write("Enter A News Article Below to Check Whether It Is Fake or Real.")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0]==1:
            st.success("The News Is Real! ")
        else:
            st.error("The News Is Fake!")
    else:
        st.warning("Please Enter Some Text To Analyze. ")            

