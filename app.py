import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
    model =  pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)


st.set_page_config(page_title="Spam message Detector", page_icon="📩")

st.title("Spam Message Detector")
st.write("Paste a message below to check whether it is a spam message or not")

message = st.text_area(
    "Paste your message:",
    height=120,
    placeholder="e.g. Congratulations! You won a free iPhone. Click now!"
)

if st.button("Check Spam"):
    if message.strip() == "":
        st.warning("Please enter a message first!")
    else:
        vec = cv.transform([message])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred]

        if pred == 1:
            st.error(f"This is a spam: confidence {prob:.2%}")
        else:
            st.success(f"This is not a spam message: confidence {prob:.2%}")