import requests
import time
import sys
import random
import pickle
import base64
import pdb
import pandas as pd
import json
from collections import OrderedDict
import torch
import os


# Assume this is leos model
import incremental_model

server_location = "http://localhost:4001"

def json_to_ordered_dict(json_str):
    json_data = json.loads(json_str)
    # If the original values were torch tensors, convert them back
    for key, value in json_data.items():
        if isinstance(value, list):
            json_data[key] = torch.tensor(value)

    ordered_dict = OrderedDict(json_data)
    return ordered_dict

def ordered_dict_to_json(ordered_dict):
    model_state_dict_serializable = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in ordered_dict.items()}
    return json.dumps(model_state_dict_serializable)

def get_initial_server_data():
    response = requests.get(server_location+"/get_initial_server_data")
    return json_to_ordered_dict(response.json())

def post_server_data(updated_weight):
    formatted_data = ordered_dict_to_json(updated_weight)
    requests.post(server_location+"/post_server_data", data=formatted_data)

def post_server_incremental_data(updated_weight):
    formatted_data = ordered_dict_to_json(updated_weight)
    requests.post(server_location+"/post_server_incremental_data", data=formatted_data)

def get_all_weights():
    response = requests.get(server_location+"/get_all_weights")
    all_weights = response.json()
    return all_weights

def get_all_incremental_weights():
    response = requests.get(server_location+"/get_all_incremental_weights")
    all_weights = response.json()
    return all_weights

def update_global_model(updated_weight):
    formatted_data = ordered_dict_to_json(updated_weight)
    requests.post(server_location+"/update_global_model", data=formatted_data)

def get_global_weights():
    response = requests.get(server_location+"/get_global_weights")
    weights = response.json()
    return weights

# Convert data set to Pandas dataframe
def get_data_set(file_path):
    data_set = pd.read_csv(file_path)
    return data_set

def test():
    response = requests.get(server_location+"/test")
    return response

def export_captions_to_json(df):
    # Check if the DataFrame is empty
    if df.empty:
        return "The DataFrame is empty."

    # Check if 'Caption' column exists
    if 'Caption' not in df.columns:
        return "The 'Caption' column is not found in the DataFrame."

    # Concatenate the text from the 'Caption' column
    captions = '. '.join(df['Caption'].astype(str))

    # If there is only one row, remove the trailing full stop
    if len(df) == 1:
        captions = captions.rstrip('.')

    # Prepare the data for JSON
    data = {'captions': captions}

    # Write to a JSON file
    with open('captions.json', 'w') as json_file:
        json.dump(data, json_file)

    return "Captions exported to captions.json."

def incremental_loop(data_set, new_row):
    model = incremental_model
    global_weights = get_global_weights()
    formatted_global_weight = json_to_ordered_dict(global_weights)
    # Send formatted_global_weight + Synthetic data to sub model 3
    synthetic_local_text_data = model.submodel_three(data_set)
    reccomendations, new_data_set, noisy_incremental_weights = model.submodel_four(
        formatted_global_weight,
        synthetic_local_text_data,
        new_row)
    print(reccomendations)
    export_captions_to_json(reccomendations)
    del synthetic_local_text_data
    del new_row

    # replace current local data set with new_data_set
    new_data_set.to_csv('./data/video/incremental_data.csv', index=False)

    post_server_incremental_data(noisy_incremental_weights)

    time.sleep(10)

    # Get List of weights from server
    all_incremental_weights = get_all_incremental_weights()
    all_incremental_weights_formatted_to_ordered_dict = [];

    # Loop through each weights the reformat back to ordered Dictionary type
    for weight in all_incremental_weights:
        json_str = json.dumps(weight, indent=2)
        formatted_weight = json_to_ordered_dict(json_str)
        all_incremental_weights_formatted_to_ordered_dict.append(formatted_weight)

    # Process all formatted weights in submodel two get aggregated and arfed'd weights
    arfed_incremental_weights = model.submodel_two(all_incremental_weights_formatted_to_ordered_dict)

    # STEP 4
    # Update global model in the server with new weights
    update_global_model(arfed_incremental_weights)

def main():
    while True:
        test()
        # STEP 1 
        # Get Initial Global Model Weights
        initial_weights = get_initial_server_data()
        model = incremental_model

        # Get Training Data Set
        training_data = get_data_set('./data/video/node1_data.csv')

        # Load Weights and Training Data and Process it through submodel One
        updated_weights = model.submodel_one(initial_weights, training_data)

        # Upload updated weights to server
        post_server_data(updated_weights)
        # Await other clients that send their weights to server
        time.sleep(10)

        # Get List of weights from server
        all_weights = get_all_weights()
        all_weights_formatted_to_ordered_dict = [];

        # Loop through each weights the reformat back to ordered Dictionary type
        for weight in all_weights:
            json_str = json.dumps(weight, indent=2)
            formatted_weight = json_to_ordered_dict(json_str)
            all_weights_formatted_to_ordered_dict.append(formatted_weight)

        # Process all formatted weights in submodel two get aggregated and arfed'd weights
        arfed_weights = model.submodel_two(all_weights_formatted_to_ordered_dict)

        # Update global model in the server with new weights
        update_global_model(arfed_weights)
        # ###STEP 3 + 4###

        # Find out how many new row of data theres,loop for that many times
        # so have like initial data and then another updated dataset
        #  check is past dataset count < new dataset count if yes
        # Loop for difference between past and new datasetnHh
        training_data_new = get_data_set('./data/video/node1_data_new.csv')
        if len(training_data) < len(training_data_new):
            difference = len(training_data) - len(training_data_new)
            while difference < 0:
                print(difference)
                # The reason we call it again is that the original dataset is updated
                new_row = training_data_new.iloc[difference]
                print(new_row)
                if os.path.exists('.data/video/incremental_data.csv'):
                    print("The file '.data/video/incremental_data.csv' exists.")
                    incremental_learning_local_dataset = get_data_set('.data/video/incremental_data.csv')
                    incremental_loop(incremental_learning_local_dataset, new_row)
                    difference += 1
                else:
                    print("The file '.data/video/incremental_data.csv' does not exist.")
                    incremental_learning_local_dataset = get_data_set('./data/video/node1_data.csv')
                    stripped_dataset = incremental_learning_local_dataset[['Sentiment', 'Caption']]
                    stripped_dataset.to_csv('./data/video/incremental_data.csv', index=False)
                    incremental_loop(stripped_dataset, new_row)
                    difference += 1
            pdb.set_trace()


if __name__=='__main__':
    main()

