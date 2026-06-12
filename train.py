import pandas as pd
import re
import string
import pickle
import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("data/fake_or_real_news.csv")

# Convert labels
data["label"] = data["label"].map({
    "FAKE": 1,
    "REAL": 0
})

# Keep required columns
data = data[["text", "label"]]

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Stopwords
stop_words = set(stopwords.words("english"))

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

# Apply preprocessing
data["text"] = data["text"].apply(clean_text)

# Features and labels
X = data["text"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# TF-IDF Vectorization

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.75,
    min_df=2,
    ngram_range=(1,2)
)



X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

# Train model


model = LogisticRegression(
    max_iter=3000,
    random_state=42
)





model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# Save model
with open("model/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save vectorizer
with open("model/vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully.")

