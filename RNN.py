import torch
import torch.nn as nn
import matplotlib.pyplot as plt

seed = 1
torch.manual_seed(seed)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import unidecode

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from torch.utils.data import Dataset , DataLoader
from sklearn.model_selection import train_test_split



dataset_path = 'dataset/all-data.csv'
headers = ['sentiment', 'content']
df = pd.read_csv(
    dataset_path ,
    names = headers ,
    encoding ='ISO-8859-1'
)

print(df.head())

classes = {class_name : idx for idx , class_name in enumerate(df['sentiment'].unique().tolist())}
df['sentiment'] = df['sentiment'].apply(lambda x: classes [x])

english_stop_words = stopwords.words('english ')
stemmer = PorterStemmer()

def text_normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([ word for word in text.split(' ')if word not in
    english_stop_words ])
    text = ' '.join([stemmer.stem(word)for word in text.split(' ')])

    return text

df['content'] = df['content'].apply(lambda x: text_normalize(x))


vocab = []
for sentence in df['content'].tolist():
    tokens = sentence.split()
    for token in tokens :
        if token not in vocab :
            vocab.append(token)

vocab.append('UNK')
vocab.append('PAD')
word_to_idx = { word : idx for idx , word in enumerate(vocab)}
vocab_size = len(vocab)


import pickle
with open('vocab.pkl', 'wb') as f:
     pickle.dump(vocab, f)
     
     
def transform(text , word_to_idx , max_seq_len):
    tokens = []
    for w in text.split():
        try :
            w_ids = word_to_idx [w]
        except :
            w_ids = word_to_idx ['UNK']
        tokens.append(w_ids)

    if len(tokens)< max_seq_len :
        tokens += [ word_to_idx ['PAD']] *(max_seq_len - len(tokens))
    elif len(tokens)> max_seq_len :
        tokens = tokens [: max_seq_len ]

    return tokens


val_size = 0.2
test_size = 0.125
is_shuffle = True
texts = df['content'].tolist()
labels = df['sentiment'].tolist()

X_train , X_val , y_train , y_val = train_test_split(
                                                    texts , labels ,
                                                    test_size = val_size ,
                                                    random_state =seed ,
                                                    shuffle = is_shuffle)

X_train , X_test , y_train , y_test = train_test_split(
                                                    X_train , y_train ,
                                                    test_size = val_size ,
                                                    random_state =seed ,
                                                    shuffle = is_shuffle)


class FinancialNews(Dataset):
    def __init__(self ,X, y,word_to_idx ,max_seq_len ,transform = None):
        self.texts = X
        self.labels = y
        self.word_to_idx = word_to_idx
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self , idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.transform :
            text = self.transform(text ,self.word_to_idx ,self.max_seq_len)
        text = torch.tensor(text)

        return text , label
    
max_seq_len = 32

train_dataset = FinancialNews(
                X_train , y_train ,
                word_to_idx = word_to_idx ,
                max_seq_len = max_seq_len ,
                transform = transform)

val_dataset = FinancialNews(
                X_val , y_val ,
                word_to_idx = word_to_idx ,
                max_seq_len = max_seq_len ,
                transform = transform)

test_dataset = FinancialNews(
                X_test , y_test ,
                word_to_idx = word_to_idx ,
                max_seq_len = max_seq_len ,
                transform = transform)

train_batch_size = 128
test_batch_size = 8

train_loader = DataLoader(
                train_dataset ,
                batch_size = train_batch_size ,
                shuffle = True)

val_loader = DataLoader(
                val_dataset ,
                batch_size = test_batch_size ,
                shuffle = False)

test_loader = DataLoader(
                test_dataset ,
                batch_size = test_batch_size ,
                shuffle = False)


class SentimentClassifier(nn.Module):
    def __init__(
        self , vocab_size , embedding_dim ,
        hidden_size , n_layers , n_classes ,
        dropout_prob):
        
        super(SentimentClassifier , self).__init__()
        self.embedding = nn.Embedding(vocab_size , embedding_dim)
        self.rnn = nn.RNN(embedding_dim , hidden_size , n_layers , batch_first =True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size , 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16 , n_classes)

    def forward(self , x):
        x = self.embedding(x)
        x, hn = self.rnn(x)
        x = x[: , -1 , :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
    
n_classes = len(list(classes.keys()))
embedding_dim = 64
hidden_size = 64
n_layers = 2
dropout_prob = 0.2
device = 'cuda' if torch.cuda.is_available()else 'cpu'

model = SentimentClassifier(
        vocab_size = vocab_size ,
        embedding_dim = embedding_dim ,
        hidden_size = hidden_size ,
        n_layers = n_layers ,
        n_classes = n_classes ,
        dropout_prob = dropout_prob
      ).to(device)

lr = 1e-4
epochs = 50

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

def evaluate(model , dataloader , criterion , device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for inputs , labels in dataloader :
            inputs , labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs , labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data , 1)
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()
        
    loss = sum(losses)/ len(losses)
    acc = correct / total
    
    return loss , acc


def fit(model ,train_loader ,val_loader ,criterion ,optimizer ,device ,epochs):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []
        model.train()
        for idx ,(inputs , labels)in enumerate(train_loader):
            inputs , labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()
            
            batch_train_losses.append(loss.item())
            
        train_loss = sum(batch_train_losses)/ len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss , val_acc = evaluate(model , val_loader ,criterion , device)
        val_losses.append(val_loss)

        print(f'EPOCH { epoch + 1}:\ tTrain loss : { train_loss :.4f}\ tVal loss : {val_loss :.4f}')

    torch.save(model.state_dict(), 'model.pth')
    return train_losses , val_losses
    
train_losses , val_losses = fit(model , train_loader , val_loader , criterion , optimizer , device , epochs )


val_loss , val_acc = evaluate (
                    model ,
                    val_loader ,
                    criterion ,
                    device
                    )

test_loss , test_acc = evaluate (
                    model ,
                    test_loader ,
                    criterion ,
                    device
                    )

print ('Evaluation on val / test dataset ')
print ('Val accuracy : ', val_acc )
print ('Test accuracy : ', test_acc )


plt.figure(figsize = (10 , 6))
plt.title('Losses')
plt.plot(train_losses , label = 'Train loss')
plt.plot(val_losses , label = 'Val loss')
plt.legend()
plt.show()

