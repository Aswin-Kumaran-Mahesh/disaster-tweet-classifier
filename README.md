🌪️ Disaster Tweet Classifier V2

Live demo: https://huggingface.co/spaces/Aswin92/disaster-tweet-classifier-v2

This project predicts whether a tweet describes a real disaster or not. It compares three different NLP models and shows how each one behaves on noisy social media text.

🔍 Models Used

🧠 DeBERTa v3
Best overall model with the highest accuracy and macro F1.

⚡ DistilBERT
Lightweight and fast, strong performance for its size.

🔁 BiLSTM (RNN)
Classic sequence model used as a baseline.

🚀 What the App Does

Takes a tweet as input

Predicts the probability of the tweet being a disaster

Applies thresholds to classify each model's prediction

Displays all three model outputs side by side

Example test input:
“Firefighters are trying to rescue people from burning buildings after the explosion.”

All three models correctly flag this as a disaster.

🛠️ Tech Stack

Python

Streamlit

Hugging Face Transformers

PyTorch

Gradio or Streamlit UI (depending on your app file)

📂 Project Structure
.
├── streamlit_app.py          # Main app
├── model_loader.py           # Model loading utilities
├── requirements.txt
├── README.md
└── models/
    ├── deberta_v3/
    ├── distilbert/
    └── bilstm/

📚 Dataset

Kaggle: Natural Language Processing with Disaster Tweets
Labels:

1 = disaster

0 = not disaster

🧠 Key Learnings

Comparing transformer models with RNNs

Threshold-based classification

Handling social media text

Deploying ML models on Hugging Face Spaces

🔗 Links

Demo: https://huggingface.co/spaces/Aswin92/disaster-tweet-classifier-v2

Repo: https://github.com/Aswin-Kumaran-Mahesh/disaster-tweet-classifier
