import streamlit as st
import torch
import torch.nn     as nn
from torchtext.data.utils import get_tokenizer
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unidecode
import re

class SentimentClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim,
        hidden_size, n_layers, n_classes,
        dropout_prob
    ):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            n_layers,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the vocabulary and model
with open('vocab.pkl', 'rb') as f:
    vocab_list = pickle.load(f)
    vocab = {word: idx  for idx, word in enumerate(vocab_list)}

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#The Device must be match with the type of device you train model!
#For example: if you use cuda to train model and save as model.pth, then you must use cuda device.

device = torch.device('cpu')

# The hyperparameters must be the same with training
model = SentimentClassifier(
    vocab_size=len(vocab),
    embedding_dim=64,  
    hidden_size=64,
    n_layers=2,
    n_classes=3,
    dropout_prob=0.2
)

model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()  # Set to evaluation mode
model.dropout.train(False)  # Explicitly disable dropout

tokenizer = get_tokenizer('basic_english')

# Streamlit app
st.title('Financial News Sentiment Analysis')
st.markdown("## Vocabulary")
st.markdown("**Model loaded with vocab size:**")
st.write(len(vocab))
st.markdown("**Available tokens:**")
st.write([word for word in vocab if word in ['PAD', 'UNK']])

st.markdown("## Input News")
text_input = st.text_area("Enter your financial news text here:", height=200)

# The data processing is like training because we want the input is same with training
# ============================================================================================
english_stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def text_normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split(' ') if word not in english_stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split(' ')])
    return text
# ============================================================================================

# When you click streamlit button, this logic will execute
if st.button('Analyze Sentiment'):
    if text_input:
        # Normalize, tokenize and convert to tensor
        normalized_text = text_normalize(text_input)
        tokens = tokenizer(normalized_text)
        
        # Debug tokenization
        st.markdown("## Process Infor")
        st.markdown("**Normalized** **text:** " f"{normalized_text}")
        st.markdown("**Tokens:** " f"{tokens}")
        
        # Common max length, adjust as needed
        # Because if the max_length is to large, we will have a lotof padding token, then the result will allway Negative
        max_seq_len = 25 
        indices = []
        unknown_tokens = []
        for token in tokens:
            if token in vocab:
                indices.append(vocab[token])
            else:
                indices.append(vocab['UNK'])
                unknown_tokens.append(token)
                
        # Debug unknown tokens
        if unknown_tokens:
            st.markdown("**Unknown tokens:**" f"{unknown_tokens}")
                
        # Pad or truncate sequence
        if len(indices) < max_seq_len:
            indices += [vocab['PAD']] * (max_seq_len - len(indices))
        else:
            indices = indices[:max_seq_len]
            
        st.write(indices)
        tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(tensor)

            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction = torch.argmax(output, dim=1).item()
            
        # Map prediction to sentiment
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map[prediction]
        
        st.markdown("## Sentiment: " f"{sentiment}")
        
        # Display confidence scores
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        st.markdown("\n ## Confidence Scores:")
        for sent, prob in zip(sentiment_map.values(), probabilities):
            st.markdown(f"**{sent}:** {prob:.2%}")
    else:
        st.write("Please enter some text to analyze.")