# ==============
# Import Library
# ==============

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import sys

# ==============
# Data
# ==============
# Process the dataset by removing punctuation for TF-IDF
def preprocess(text):
    text = re.sub(r'[^A-Za-z0-9]+', " ", text)
    text = text.lower()
    return text

# ==============
# Model
# ==============
class LSTM_REGR(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(LSTM_REGR, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=n_layers, 
            batch_first=True
        )
        self.linear = nn.Linear(
            hidden_size, 
            1
        )

    def forward(self, x):
        x, _ = self.lstm(x.float())
        x = self.linear(x)
        return x

# ==============
# Training
# ==============
def train_model(model, optimizer, loss_fn, loader, X_valid, y_valid, n_epochs = 10, display_prog = True, n_display = 10):
    valid_rmse_list = []
    for epoch in range(n_epochs):
        model.train()
        progress_bar = tqdm(loader)
        for X_batch, y_batch in progress_bar:
            y_pred = model(X_batch.long())
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_rmse = RMSE(y_pred, y_batch)
            progress_bar.set_description(f"Epoch {epoch+1}/{n_epochs}: Train RMSE = {train_rmse:.4f}")
        
        # Validation
        if ((epoch+1) % n_display != 0):
            continue
        _, valid_rmse = evaluate(model, X_valid, y_valid)
        if display_prog:
            print(f"Epoch {epoch+1}: Valid RMSE = {valid_rmse:.4f}")
        
        # Save model
        torch.save(model.state_dict(), f"models/sleep_train/LSTM_epoch_{epoch+1}.pt")
        valid_rmse_list.append(valid_rmse)
    
    return (np.argmin(valid_rmse_list) + 1) * n_display, valid_rmse_list
        
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X.long())
        rmse = RMSE(y_pred, y)
    return y_pred, rmse

def predict(model, X):
    X = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(X.long())
    return y_pred

def RMSE(y_pred, y):
    return np.sqrt(np.mean(
        np.square(y_pred.detach().numpy() - y.detach().numpy())
    ))

# ==========================================
# Define Model and Begin Training Process
# ==========================================
def main():
    # ===========
    # DATA
    # ===========
    # Read and preprocess data
    data = pd.read_csv("data/data.csv")
    data["text"] = data.get("text").apply(preprocess)

    # Remove Outliers
    data = data[data['birth_yr'] >= 1400]
    data['text_len'] = data['text'].str.len()
    t = data['text_len'].mean() + 3 * data['text_len'].std()
    data = data.loc[data['text_len'] < t]

    # Create Tf-IDF Vectorizer
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        analyzer='word',
        max_features=2000,
        tokenizer=word_tokenize,
        stop_words=stopwords.words("english")
    )

    # Split Data
    X = tfidf.fit_transform(data["text"]).toarray()
    y = np.array(data["birth_yr"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    # Reformat Data for Pytorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
     
    # ================
    # MODEL Training
    # ================
    # define model
    input_size = tfidf.max_features
    hidden_size = 200

    model = LSTM_REGR(input_size, hidden_size, n_layers=6)

    # define training parameters
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    try:
        n_epochs = int(sys.argv[1])
    except:
        raise AssertionError("ERROR: enter number of epochs to train in the following format: ./{PATH}/model_testing.py {n_epochs}")

    # train model
    best_epoch, valid_rmse_list = train_model(
        model, 
        optimizer, 
        loss_fn, 
        train_loader, 
        X_valid,
        y_valid,
        n_epochs=n_epochs
    )
    # record results
    with open("models/sleep_train/log.txt", "w") as log_file:
        log_file.write(f"Best Epoch: {best_epoch}\n{valid_rmse_list}")

if __name__ == "__main__":
    INVALID_EPOCH_MSG = "Enter number of epochs to train in the following format: ./{PATH}/model_testing.py {n_epochs}"
    assert len(sys.argv) == 2, INVALID_EPOCH_MSG

    print("============================================")
    print(f"Running model_testing.py with {sys.argv[1]} epochs")
    print("============================================")
    main()
    print("============================================")
    print("model_testing.py process finished")
    print("============================================")