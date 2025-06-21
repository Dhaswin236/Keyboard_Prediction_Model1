import streamlit as st
import torch
import random
from collections import Counter
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------
# Utilities
# ---------------------
def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()

def build_vocab(tokens, max_vocab=5000):
    counter = Counter(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(counter.most_common(max_vocab - 2), 2):
        vocab[word] = i
    return vocab

def encode(tokens, vocab):
    return [vocab.get(w, vocab["<UNK>"]) for w in tokens]

# ---------------------
# Dataset Simulation
# ---------------------
class MultimodalKeyboardDataset:
    def __init__(self, text, vocab, seq_len=4):
        tokens = tokenize(text)
        self.vocab = vocab
        self.seq_len = seq_len
        self.samples = []
        encoded = encode(tokens, vocab)
        for i in range(len(encoded) - seq_len):
            seq = encoded[i:i+seq_len]
            label = encoded[i+seq_len]
            speeds = [random.uniform(0.1, 1.0) for _ in seq]
            self.samples.append((seq, speeds, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ---------------------
# Feature Preparation
# ---------------------
def prepare_rf_data(dataset):
    X, y = [], []
    for seq, speed, label in dataset:
        X.append(seq + speed)
        y.append(label)
    return np.array(X), np.array(y)

# ---------------------
# Train RF Model
# ---------------------
def train_model(corpus):
    tokens = tokenize(corpus)
    vocab = build_vocab(tokens)
    dataset = MultimodalKeyboardDataset(corpus, vocab)
    X, y = prepare_rf_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, vocab

# ---------------------
# Predict Function
# ---------------------
def predict_next_word(model, vocab, input_text, typing_speeds, seq_len=4):
    tokens = tokenize(input_text)
    encoded = encode(tokens, vocab)[-seq_len:]
    speeds = typing_speeds[-seq_len:]

    if len(encoded) < seq_len:
        encoded = [0] * (seq_len - len(encoded)) + encoded
        speeds = [0.5] * (seq_len - len(speeds)) + speeds

    features = np.array(encoded + speeds).reshape(1, -1)
    pred_id = model.predict(features)[0]

    for word, idx in vocab.items():
        if idx == pred_id:
            return word
    return "<UNK>"

# ---------------------
# Streamlit UI
# ---------------------
st.title("🧠 Predictive Keyboard with Typing Speed")
corpus = "we are building a predictive keyboard model using typing speed and deep learning"
model, vocab = train_model(corpus)

input_text = st.text_input("Enter text (last few words):", "keyboard model using")
typing_input = st.text_input("Typing speeds (comma-separated):", "0.5, 0.6, 0.4, 0.3")

if st.button("Predict Next Word"):
    try:
        speeds = list(map(float, typing_input.split(',')))
        predicted_word = predict_next_word(model, vocab, input_text, speeds)
        st.success(f"🔮 Predicted Next Word: **{predicted_word}**")
    except:
        st.error("Please enter valid comma-separated typing speeds.")
