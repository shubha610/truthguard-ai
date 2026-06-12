import streamlit as st
import pickle
import re
import string
import nltk
import os

from nltk.corpus import stopwords

# ---------------------------
# NLTK SETUP (Streamlit safe)
# ---------------------------
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

stop_words = set(stopwords.words("english"))

# ---------------------------
# LOAD MODEL + VECTORIZER
# ---------------------------
with open("model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ---------------------------
# TEXT CLEANING FUNCTION
# (IMPORTANT: keep SAME as training)
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ---------------------------
# UI
# ---------------------------
st.title("📰 Fake News Detection System")
st.write("Enter a news article and get prediction with confidence score.")

text = st.text_area("Enter News Text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:

        # preprocess
        cleaned = clean_text(text)

        # vectorize
        vector_input = vectorizer.transform([cleaned])

        # prediction
        prediction = model.predict(vector_input)

        # confidence (IMPORTANT PART)
        prob = model.predict_proba(vector_input)

        confidence = max(prob[0]) * 100

        # label mapping (based on your training)
        # 1 = FAKE, 0 = REAL

        if prediction[0] == 1:
            st.error(f"🚨 FAKE NEWS")
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.success(f"✅ REAL NEWS")
            st.write(f"Confidence: {confidence:.2f}%")