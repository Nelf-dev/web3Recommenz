import fastai 
from fastai.text.all import *
import pandas as pd

# Read the CSV file
import_path = r'data\video' 
df = pd.read_csv(f'{import_path}\\node1_data.csv')
train_df = df.iloc[:9] 
test_df = df.iloc[9:]

# Create data bunches from the DataFrames
dls = TextDataLoaders.from_df(train_df, text_col='Caption', label_col='Sentiment',n_workers=0)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.3) 

# Specify optimizer and learning rate
learn.opt_func = torch.optim.Adam
learn.lr = 0.1

# Define loss function
learn.loss_func = nn.CrossEntropyLoss()

learn.predict("The chicken is delicious")

learn.save('model')

model_path = r'models'
state_dict = torch.load(f'{model_path}\\model.pth')