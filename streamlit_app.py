import os
import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
import torch.nn as nn

# ------------------- Device -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- BiLSTM class -------------------
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
        num_classes=2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.num_dirs = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_dirs, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embeds = self.embedding(input_ids)
        outputs, (h_n, c_n) = self.lstm(embeds)
        last_layer_h = h_n[-self.num_dirs:, :, :]
        last_h = last_layer_h.transpose(0, 1).contiguous().view(input_ids.size(0), -1)
        x = self.dropout(last_h)
        logits = self.fc(x)
        return logits


# ------------------- Shared tokenizer for BiLSTM -------------------
@st.cache_resource
def load_shared_tokenizer():
    """Load tokenizer once for BiLSTM (uses DistilBERT tokenizer)"""
    model_name = "Aswin92/distilbert-disaster-tweets"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


# ------------------- Individual model loaders -------------------
@st.cache_resource
def load_deberta():
    model_name = "Aswin92/deberta-v3-disaster-tweets"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_distilbert():
    model_name = "Aswin92/distilbert-disaster-tweets"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_bilstm():
    # Use shared tokenizer instead of loading DistilBERT model
    tokenizer = load_shared_tokenizer()

    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
        num_classes=2,
    )

    # Load BiLSTM weights directly
    state_path = "bilstm_state_dict.pt"
    
    if not os.path.exists(state_path):
        current_dir = os.getcwd()
        files_in_dir = os.listdir(current_dir)
        raise FileNotFoundError(
            f"BiLSTM weights file '{state_path}' not found.\n"
            f"Current directory: {current_dir}\n"
            f"Files available: {files_in_dir}\n"
            f"Please upload 'bilstm_state_dict.pt' directly to your Space root (not in a zip)."
        )
    
    state_dict = torch.load(state_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ------------------- Prediction helper -------------------
def predict_text(text, model_name, threshold):
    if model_name == "DeBERTa-v3":
        tokenizer, model = load_deberta()
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        # Move tensors to device
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

    elif model_name == "DistilBERT":
        tokenizer, model = load_distilbert()
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        # DistilBERT does NOT accept token_type_ids
        if "token_type_ids" in encoded:
            encoded.pop("token_type_ids")

        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

    elif model_name == "BiLSTM (RNN)":
        tokenizer, model = load_bilstm()
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=-1)[0]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    prob_not = float(probs[0].item())
    prob_dis = float(probs[1].item())
    pred_label = "Disaster" if prob_dis >= threshold else "Not disaster"
    return pred_label, prob_not, prob_dis


# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")

st.title("🌪️ Disaster Tweet Classifier")
st.write(
    "NLP project on the Kaggle **Disaster Tweets** dataset.\n\n"
    "Compare **DeBERTa-v3**, **DistilBERT**, and a custom **BiLSTM (RNN)** "
    "to decide whether a tweet describes a real disaster."
)

# -------- Sidebar controls --------
with st.sidebar:
    st.header("⚙️ Model Selection")
    
    # Let user select which models to run
    run_deberta = st.checkbox("Run DeBERTa-v3", value=True)
    run_distilbert = st.checkbox("Run DistilBERT", value=True)
    run_bilstm = st.checkbox("Run BiLSTM (RNN)", value=True)
    
    st.markdown("---")
    st.header("⚙️ Thresholds")

    thr_deberta = st.slider(
        "DeBERTa-v3 threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.60,
        step=0.05,
        disabled=not run_deberta,
    )
    thr_distil = st.slider(
        "DistilBERT threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.80,
        step=0.05,
        disabled=not run_distilbert,
    )
    thr_bilstm = st.slider(
        "BiLSTM (RNN) threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        disabled=not run_bilstm,
    )

    st.caption(
        "Each model predicts `P(disaster)`. If that probability is "
        "≥ its threshold, we classify it as **disaster**."
    )

# -------- Main input area --------
st.subheader("📝 Input Tweet")

default_example = (
    "Firefighters are trying to rescue people from burning buildings after the explosion."
)

tweet_text = st.text_area(
    "Enter a tweet:",
    value=default_example,
    height=120,
)

if st.button("Classify Tweet"):
    text = tweet_text.strip()
    if not text:
        st.warning("Please type a tweet first.")
    else:
        # Build list of models to run based on checkboxes
        configs = []
        if run_deberta:
            configs.append(("DeBERTa-v3", thr_deberta))
        if run_distilbert:
            configs.append(("DistilBERT", thr_distil))
        if run_bilstm:
            configs.append(("BiLSTM (RNN)", thr_bilstm))
        
        if not configs:
            st.warning("Please select at least one model to run.")
        else:
            try:
                with st.spinner(f"Running {len(configs)} model(s)..."):
                    rows = []
                    for name, thr in configs:
                        pred_label, prob_not, prob_dis = predict_text(text, name, thr)
                        rows.append(
                            {
                                "Model": name,
                                "Threshold": thr,
                                "P_not_disaster": prob_not,
                                "P_disaster": prob_dis,
                                "Predicted_label": pred_label,
                            }
                        )

                # ---- Table view ----
                st.subheader("📋 Model outputs")
                df = pd.DataFrame(rows)
                # Nice formatting for display
                df_display = df.copy()
                df_display["P_not_disaster"] = df_display["P_not_disaster"].map(lambda x: f"{x:.3f}")
                df_display["P_disaster"] = df_display["P_disaster"].map(lambda x: f"{x:.3f}")
                df_display["Threshold"] = df_display["Threshold"].map(lambda x: f"{x:.2f}")
                st.dataframe(df_display, use_container_width=True)

                # ---- Interactive bar chart comparing P(disaster) ----
                if len(rows) > 1:
                    st.subheader("📊 P(disaster) comparison")
                    chart_df = df[["Model", "P_disaster"]].set_index("Model")
                    st.bar_chart(chart_df)

                # ---- Per-model summary text ----
                st.subheader("🔎 Per-model decisions")
                for row in rows:
                    name = row["Model"]
                    thr = row["Threshold"]
                    p_dis = row["P_disaster"]
                    p_not = row["P_not_disaster"]
                    label = row["Predicted_label"]

                    st.markdown(f"**{name}**")
                    st.write(
                        f"- P(disaster = 1): `{p_dis:.3f}`\n"
                        f"- P(not disaster = 0): `{p_not:.3f}`\n"
                        f"- Threshold: `{thr:.2f}` → prediction = `{label}`"
                    )
                    st.markdown("---")
                    
            except FileNotFoundError as e:
                st.error(f"❌ {str(e)}")
                st.info("Please upload `bilstm_state_dict.pt` to the root of your Space repository.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

st.markdown("---")
st.caption(
    "Team 20 · DeBERTa-v3, DistilBERT, and BiLSTM comparative study on disaster tweet classification."
)