import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Junayed\spam-detection-model\spam_data.csv", encoding= "latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    df["label_dif"] = (df["label"] == "spam").astype(int)
    return df

@st.cache_resource
def train_model(df):
    cv = CountVectorizer(stop_words='english', max_features=5000)
    X = cv.fit_transform(df["message"])
    Y = df["label_dif"]
    model = MultinomialNB()
    model.fit(X,Y)
    return cv,model

