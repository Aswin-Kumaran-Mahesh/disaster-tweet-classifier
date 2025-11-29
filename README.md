# 🌪️ Disaster Tweet Classifier

Live demo: https://huggingface.co/spaces/Aswin92/disaster-tweet-classifier-v2  

This project predicts whether a tweet describes a real disaster or not.  
It uses three different models and compares how they perform on the Kaggle Disaster Tweets dataset.

- 🧠 DeBERTa v3
- ⚡ DistilBERT
- 🔁 BiLSTM (RNN)

Each model outputs P(disaster). If the probability crosses a chosen threshold, the tweet is classified as a disaster.

---

## 🚀 Features

- Classifies tweets as disaster or not disaster
- Runs three models side by side
- Shows model confidence scores for each prediction
- Threshold based decision logic per model
- Simple web interface for testing your own tweets

---

## 🧱 Models

### DeBERTa v3
- Highest accuracy and macro F1 in this project
- Handles context and long range dependencies very well

### DistilBERT
- Lightweight and fast
- Strong performance while using fewer parameters

### BiLSTM (RNN)
- Custom RNN baseline
- Useful to compare classic sequence models with transformer based models

---

## 🗂️ Project Structure

Example structure:

```text
.
├── app.py                 # Gradio or Spaces app
├── requirements.txt       # Python dependencies
├── model_utils.py         # Loading and inference helpers
├── preprocessing.py       # Text cleaning and tokenization
├── README.md
└── notebooks/
    ├── training_deberta.ipynb
    ├── training_distilbert.ipynb
    └── training_bilstm.ipynb
