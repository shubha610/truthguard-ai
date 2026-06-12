
import streamlit as st
import pickle
import re
import string

from nltk.corpus import stopwords

# Download stopwords
import nltk
import os

nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

# Load saved model
with open("model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open("model/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Stopwords
stop_words = set(stopwords.words("english"))

# Page config
st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="📰",
    layout="centered"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 20px;
    }

    .stTextArea textarea {
        font-size: 16px;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<div class="title">TruthGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Fake News Detection using NLP & Machine Learning</div>',
    unsafe_allow_html=True
)

# Text cleaning function
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

# User input
news_input = st.text_area(
    "Paste a news article or headline:",
    height=220,
    placeholder="Enter news content here..."
)

# Prediction button

if st.button("Analyze News"):

    if news_input.strip() == "":
        st.warning("Please enter some news text.")

    else:

        cleaned_text = clean_text(news_input)

        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]

        probabilities = model.predict_proba(vectorized_text)

        confidence_score = max(probabilities[0]) * 100

        st.subheader("Prediction Result")

    if prediction > 50:

         st.success("✅ This news appears to be REAL.")

    else:

        st.error("⚠️ This news appears to be FAKE.")

        st.write(f"Confidence Score: {confidence_score:.2f}%")



# Sidebar
st.sidebar.title("About Project")

st.sidebar.info(
    """
    TruthGuard AI is an NLP-based Fake News Detection system
    built using Machine Learning and TF-IDF Vectorization.

    Technologies Used:
    - Python
    - Scikit-learn
    - NLP
    - Streamlit
    """
)

