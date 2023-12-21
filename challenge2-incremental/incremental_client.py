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

def incremental_loop(model, data_set, new_row):
        global_weights = get_global_weights()
        formatted_global_weight = json_to_ordered_dict(global_weights)
        
        # Send formatted_global_weight + Synthetic data to sub model 3
        synthetic_local_text_data = model.submodel_three(formatted_global_weight, data_set)

        # Receive new row of data
        new_row_of_data = get_data_set('./data/video/node1_incremental_1.csv')

        reccomendations, new_data_set, noisy_incremental_weights = model.submodel_four(
            global_weights,
            synthetic_local_text_data,
            new_row_of_data)
        
        del synthetic_local_text_data
        del new_row_of_data

        # replace current local data set with new_data_set

        post_server_incremental_data(noisy_incremental_weights)

        time.sleep(0.5)

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
        time.sleep(0.5)

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
                incremental_learning_local_dataset = get_data_set('./data/video/node1_data.csv')
                stripped_dataset = incremental_learning_local_dataset[['Sentiment', 'Caption']]

                new_row = training_data_new.iloc[difference]
                print(new_row)

                # incremental_loop(model, stripped_dataset, new_row)
                difference += 1




if __name__=='__main__':
    main()

