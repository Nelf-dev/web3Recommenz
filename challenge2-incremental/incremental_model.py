import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from opacus import PrivacyEngine
import pdb
import random
from collections import OrderedDict
from typing import List

# TODO: Nelson needs this import to download NLTK MODULE
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Download nltk module
nltk.download('punkt')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def build_markov_model(data, n_gram=2):
    model = {}
    for sentence in data:
        sentence = word_tokenize(sentence.lower())
        for i in range(len(sentence) - n_gram):
            gram = tuple(sentence[i:i + n_gram])
            next_word = sentence[i + n_gram]
            if gram not in model:
                model[gram] = {}
            if next_word not in model[gram]:
                model[gram][next_word] = 0
            model[gram][next_word] += 1
    return model

def generate_sentence(model, max_length=20):
    sentence = []
    n_gram = list(model.keys())[random.randint(0, len(model) - 1)]
    sentence.extend(n_gram)
    for _ in range(max_length - len(n_gram)):
        last_gram = tuple(sentence[-len(n_gram):])
        if last_gram in model:
            next_words = model[last_gram]
            next_word = random.choices(list(next_words.keys()), weights=next_words.values())[0]
            sentence.append(next_word)
        else:
            break
    return ' '.join(sentence)

def submodel_zero(dataset):
    captions = dataset['Caption'].tolist()
    markov_model = build_markov_model(captions)

    synthesized_captions = []
    for _ in range(5):
        synthesized_captions.append(generate_sentence(markov_model))

    return pd.DataFrame(synthesized_captions, columns=['Caption'])

# Function to check and balance the dataset
def balance_dataset(df):
    sentiment_counts = df['Sentiment'].value_counts()
    if any(sentiment_counts != sentiment_counts[0]):
        ros = RandomOverSampler(random_state=42)
        df_balanced, _ = ros.fit_resample(df, df['Sentiment'])
        return df_balanced
    return df

# Text preprocessing steps
def preprocess_text(text):
    # Text cleaning
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)

    # Normalization (using lemmatization here)
    lemmatizer = WordNetLemmatizer()
    normalized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in normalized_tokens if token not in stop_words]

    return filtered_tokens

# Function to create embeddings
def create_embeddings(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Function to do padding
def pad_or_truncate_sequences(sequences, fixed_length, vector_size):
    # Initialize a zero-filled 3D array: number of sequences x fixed_length x vector_size
    adjusted_sequences = np.zeros((len(sequences), fixed_length, vector_size))
    
    for i, sequence in enumerate(sequences):
        sequence_length = len(sequence)
        if sequence_length > fixed_length:
            # Truncate the sequence
            adjusted_sequences[i, :, :] = np.array(sequence[:fixed_length])
        else:
            # Pad the sequence with zeros
            adjusted_sequences[i, :sequence_length, :] = np.array(sequence)
    
    return adjusted_sequences

# Custom Neural Network Model
class SentimentAnalysisModel_global(nn.Module):
    def __init__(self, input_size):
        super(SentimentAnalysisModel_global, self).__init__()
        self.fc1 = nn.Linear(input_size, 2)
        self.fc2 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # Replace BatchNorm with GroupNorm
        self.groupnorm = nn.GroupNorm(1, 2)  # GroupNorm with 1 group and 2 channels

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.batchnorm(x)
        x = self.groupnorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def submodel_one(state_dict, df):
    # Step 1: Check and balance dataset
    df = balance_dataset(df)

    # Step 2: Preprocess text data
    df['processed_caption'] = df['Caption'].apply(preprocess_text)
    sentences = df['processed_caption'].tolist()
    word2vec_model = create_embeddings(sentences)

    vectorized_sentences = [[word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv] for sentence in sentences]

    # Adjusting sequences to the fixed input size of 100
    # Assuming each word vector from Word2Vec is of size 100
    vector_size = 100
    fixed_input_size = 100
    modified_input_size = fixed_input_size * vector_size  # 10000
    adjusted_sequences = pad_or_truncate_sequences(vectorized_sentences, fixed_input_size, vector_size)

    # Flatten the sequences for input to the neural network
    adjusted_sequences = adjusted_sequences.reshape(len(adjusted_sequences), -1)

    # Convert target variable to numerical format
    target = pd.get_dummies(df['Sentiment']).values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(adjusted_sequences, target, test_size=0.25, random_state=33)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64)

    # Load model with pretrained parameters
    model = SentimentAnalysisModel_global(input_size=modified_input_size)
    model.load_state_dict(state_dict)
    model.train()

    # Step 3: Train model with differential privacy
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0
    )

    # Training loop
    for epoch in range(10):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Return updated model parameters
    return model.state_dict()


def submodel_two(models: List[OrderedDict]) -> OrderedDict:
    # Helper function to calculate IQR-based non-outliers for a list of values
    def non_outlier_values(values):
        q1, q3 = torch.quantile(torch.tensor(values), torch.tensor([0.25, 0.75]))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [v for v in values if v >= lower_bound and v <= upper_bound]

    # Initialize a dictionary to store sums and counts for averaging
    sums = OrderedDict((key, torch.zeros_like(models[0][key])) for key in models[0])
    counts = OrderedDict((key, torch.zeros_like(models[0][key])) for key in models[0])

    # Iterate through each key and each element in the tensors
    for key in models[0].keys():
        for i in range(models[0][key].numel()):
            # Extract the same element from all models
            values = [model[key].view(-1)[i].item() for model in models]

            # Calculate non-outlier values
            non_outliers = non_outlier_values(values)

            # Update sums and counts
            if non_outliers:
                sums[key].view(-1)[i] = sum(non_outliers)
                counts[key].view(-1)[i] = len(non_outliers)

    # Calculate element-wise averages for non-outliers and remove the '_module' prefix
    averages = OrderedDict()
    for key in sums.keys():
        new_key = key.replace('_module.', '')  # Remove '_module.' prefix
        with torch.no_grad():  # Ensure no gradient is computed during division
            averages[new_key] = sums[key] / counts[key].clamp(min=1)  # Avoid division by zero

    return averages


