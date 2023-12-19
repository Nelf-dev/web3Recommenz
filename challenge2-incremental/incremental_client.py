import requests
import time
import sys
import random
import pickle
import base64
import pdb
import pandas as pd
import json

# Assume this is leos model
from incremental_testModel import Trainer
from example_weights import weight1, weight2

server_location = "http://localhost:4001"

# def get_server_data():
#     data = requests.get(server_location+"/get_data").json()
#     return data['params'], data['epochs'], data['current_weights']

def get_initial_server_data():
    response = requests.get(server_location+"/get_initial_server_data")
    state_dict = response.json()
    return state_dict

def post_server_data(updated_weight):
    data = updated_weight
    requests.post(server_location+"/post_server_data", json=data)

def get_federated_average():
    response = requests.get(server_location+"/get_federated_average")
    federated_average = response.json()
    return federated_average

def update_global_model(updated_weight):
    response = requests.post(server_location+"/update_global_model", json=updated_weight)

def get_global_weights():
    response = requests.get(server_location+"/get_global_weights")
    weights = response.json()
    return weights

# Convert data set to Pandas dataframe
def get_data_set():
    csv_file_path = './data/video/node1_data.csv'
    data_set = pd.read_csv(csv_file_path)
    return data_set

def subtract_dicts(dict1, dict2):
    result = {}
    for key, value1 in dict1.items():
        if isinstance(value1, dict):
            # Recursively subtract nested dictionaries
            result[key] = subtract_dicts(value1, dict2.get(key, {}))
        elif isinstance(value1, list):
            # Handle the case when values in lists are dictionaries
            if all(isinstance(v, dict) for v in value1):
                result[key] = [subtract_dicts(v1, v2) for v1, v2 in zip(value1, dict2.get(key, []))]
            else:
                result[key] = [v1 - v2 for v1, v2 in zip(value1, dict2.get(key, []))]
        else:
            # Subtract numeric values or use the value from dict1 if key is not in dict2
            result[key] = value1 - dict2.get(key, 0)
    return result


def main():
    while True:
        ###STEP 1###
        # Get Hard Coded weights from Server
        initial_params = get_initial_server_data()
        # model = #LEOS MODEL
        # # #load weights into the local model
        # model.load_state_dict(initial_params)
        # # Send data set to local model
        training_data = get_data_set()
        updated_weights = Trainer.train(training_data)
        # # Delete data_set for privacy protections
        del training_data
        # Assuming the training function for our model returns noisy updates to the weight
        post_server_data(initial_params)
        # Wait ten seconds - So that other nodes can also update the server
        time.sleep(0.5)

        # Get federated_average created from server
        averaged_weights = get_federated_average()

        # # subModel2Updates = send completed JSON data to ARFED submodel
        arfed_weights = Trainer.trainARFED(averaged_weights)

        # ###STEP 2###
        # # newWeights = subtract previousweights with subModel2Updates.weights
        subtracted_weights = subtract_dicts(weight1(), weight2())
        # #  post the new global model weights 
        update_global_model(subtracted_weights)

        # ###STEP 3###
        # get request to get the new global model weights
        global_weights = get_global_weights()
        # model.load_state_dict(initial_params)


        # # if new_data_set.count > old_data_set.count:
        # streamed_data = './data/video/node1_incremental_1.csv'
        # streamed_row = pd.read_csv(streamed_data)
        # streamed_last_row = data_set.iloc[-1]

        # updated_state_dict = pickle.loads(base64.b64decode(updated_serialized_data))
        # #load weights into the local model
        # model.load_state_dict(updated_state_dict)

        # # subModel3Updates = send newWeights into incremental submodel
        # incrementedLearningUpdates = AutoEncoderTrainer.trainIncremental(updated_serialized_data, streamed_last_row)
        # #delete the streamed row of data for data privacy
        # del streamed_last_row

        # # Send updates to server
        # post_server_data(incrementedLearningUpdates)

        # #wait 10 seconds for other clients to also send their incremental updates
        # time.sleep(10)

        # # get all noisy incremental learning model updates for ALL clients
        # update_completed_json_file = get_avg_data()

        # # ARFED the JSON file of all the model updates
        # arfed_weights = AutoEncoderTrainer.trainARFED(completed_json_file)

        # # repeat this if there is another new row of data



if __name__=='__main__':
    main()

