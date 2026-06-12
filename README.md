# TruthGuard AI - Fake News Detection System

TruthGuard AI is a Machine Learning and Natural Language Processing (NLP) based web application that detects whether a news article is real or fake.

## Features

* NLP-based text preprocessing
* TF-IDF vectorization
* Machine Learning classification
* Interactive web interface using Streamlit
* Real-time prediction with confidence score

## Technologies Used

* Python
* Scikit-learn
* NLP
* Streamlit
* TF-IDF Vectorization

## Machine Learning Workflow

1. Data preprocessing
2. Text cleaning
3. Feature extraction using TF-IDF
4. Model training using Logistic Regression
5. Model evaluation and prediction

## Model Performance

* Accuracy: 93.21%

## Project Structure

```txt
fake-news-detector/
│
├── data/
├── model/
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run model training:

```bash
py train.py
```

Run Streamlit app:

```bash
py -m streamlit run app.py
```

## Future Improvements

* Transformer-based NLP models
* BERT integration
* Fact-checking APIs
* Real-time news verification
