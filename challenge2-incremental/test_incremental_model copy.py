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

# learn.predict("The chicken is delicious")

learn.save('model')

model_path = r'models'
state_dict1 = torch.load(f'{model_path}\\model.pth')
state_dict2 = torch.load(f'{model_path}\\model.pth')

len(state_dict1)
type(state_dict1)

state_dict1 = [{'model': {'0.module.encoder.weight': {}, '0.module.encoder_dp.emb.weight': {}, '0.module.rnns.0.weight_hh_l0_raw': {}, '0.module.rnns.0.module.weight_ih_l0': {}, '0.module.rnns.0.module.bias_ih_l0': {}, '0.module.rnns.0.module.bias_hh_l0': {}, '0.module.rnns.1.weight_hh_l0_raw': {}, '0.module.rnns.1.module.weight_ih_l0': {}, '0.module.rnns.1.module.bias_ih_l0': {}, '0.module.rnns.1.module.bias_hh_l0': {}, '0.module.rnns.2.weight_hh_l0_raw': {}, '0.module.rnns.2.module.weight_ih_l0': {}, '0.module.rnns.2.module.bias_ih_l0': {}, '0.module.rnns.2.module.bias_hh_l0': {}, '1.layers.0.0.weight': {}, '1.layers.0.0.bias': {}, '1.layers.0.0.running_mean': {}, '1.layers.0.0.running_var': {}, '1.layers.0.0.num_batches_tracked': {}, '1.layers.0.2.weight': {}, '1.layers.1.0.weight': {}, '1.layers.1.0.bias': {}, '1.layers.1.0.running_mean': {}, '1.layers.1.0.running_var': {}, '1.layers.1.0.num_batches_tracked': {}, '1.layers.1.2.weight': {}}, 'opt': {'state': [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {'force_train': True, 'do_wd': False}, {'force_train': True, 'do_wd': False}, {}, {'force_train': True, 'do_wd': False}, {'force_train': True, 'do_wd': False}, {}], 'hypers': {'items': [{'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}]}}}]
state_dict2 = [{'model': {'0.module.encoder.weight': {}, '0.module.encoder_dp.emb.weight': {}, '0.module.rnns.0.weight_hh_l0_raw': {}, '0.module.rnns.0.module.weight_ih_l0': {}, '0.module.rnns.0.module.bias_ih_l0': {}, '0.module.rnns.0.module.bias_hh_l0': {}, '0.module.rnns.1.weight_hh_l0_raw': {}, '0.module.rnns.1.module.weight_ih_l0': {}, '0.module.rnns.1.module.bias_ih_l0': {}, '0.module.rnns.1.module.bias_hh_l0': {}, '0.module.rnns.2.weight_hh_l0_raw': {}, '0.module.rnns.2.module.weight_ih_l0': {}, '0.module.rnns.2.module.bias_ih_l0': {}, '0.module.rnns.2.module.bias_hh_l0': {}, '1.layers.0.0.weight': {}, '1.layers.0.0.bias': {}, '1.layers.0.0.running_mean': {}, '1.layers.0.0.running_var': {}, '1.layers.0.0.num_batches_tracked': {}, '1.layers.0.2.weight': {}, '1.layers.1.0.weight': {}, '1.layers.1.0.bias': {}, '1.layers.1.0.running_mean': {}, '1.layers.1.0.running_var': {}, '1.layers.1.0.num_batches_tracked': {}, '1.layers.1.2.weight': {}}, 'opt': {'state': [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {'force_train': True, 'do_wd': False}, {'force_train': True, 'do_wd': False}, {}, {'force_train': True, 'do_wd': False}, {'force_train': True, 'do_wd': False}, {}], 'hypers': {'items': [{'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}, {'wd': 0.01, 'sqr_mom': 0.99, 'lr': 0.001, 'mom': 0.9, 'eps': 1e-05}]}}}]

state_dict1[0]

weights = {}
for key in state_dict1[0]:
    if key.endswith("weight"):
        weights[key] = []

for state_dict in state_dict1:
    for key, tensor in state_dict.items():
        if key in weights:
            weights[key].append(tensor.cpu())




for key, value in state_dict1.items():
    print(key, value)

 def federated_average(state_dicts):
    # Check if there are any state_dicts to average
    if not state_dicts:
        return None

    # Create a dictionary to store the sum of values
    sum_dict = {}
    num_state_dicts = len(state_dicts)

    # Iterate over each state_dict
    for state_dict in state_dicts:
        for key, value in state_dict.items():
            if key not in sum_dict:
                # If the key is not in the sum_dict, initialize it
                sum_dict[key] = value.copy()  # Use copy to avoid modifying the original values
            else:
                # If the key is already in the sum_dict, add the values
                sum_dict[key] += value

    # Calculate the average by dividing each value by the number of state_dicts
    average_dict = {key: value / num_state_dicts for key, value in sum_dict.items()}

    return average_dict


federated_average_result = federated_average([state_dict1, state_dict2])

# Example usage with the provided state_dicts
state_dicts = [
    {'model': {'0.module.encoder.weight': {}, '0.module.encoder_dp.emb.weight': {}, ... }},
    {'model': {'0.module.encoder.weight': {}, '0.module.encoder_dp.emb.weight': {}, ... }},
]

federated_avg_result = federated_average([state_dict1[0], state_dict2[0]])

# Print the federated average result
print(federated_avg_result)