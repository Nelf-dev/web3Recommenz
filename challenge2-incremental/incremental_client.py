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
def get_data_set():
    csv_file_path = './data/video/node1_data.csv'
    data_set = pd.read_csv(csv_file_path)
    return data_set

def test():
    response = requests.get(server_location+"/test")
    return response

def incremental_loop(model):
        global_weights = get_global_weights()
        formatted_global_weight = json_to_ordered_dict(global_weights)
        # Append new row of data to synthetic_local_text_data
        
        # Send formatted_global_weight + Synthetic data to sub model 3
        noisy_incremental_weights = model.submodel_three(formatted_global_weight, synthetic_local_text_data)

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
        training_data = get_data_set()

        # Create synthetic data via submodel zero
        synthetic_local_text_data = model.submodel_zero(training_data)

        # Load Weights and Training Data and Process it through submodel One
        updated_weights = model.submodel_one(initial_weights, training_data)

        # Delete Training Data to Ensure Privacy
        del training_data

        # Upload updated weights to server
        post_server_data(updated_weights)
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
        # Need to put into loop only run when there is new data
        incremental_loop(model)




if __name__=='__main__':
    main()

