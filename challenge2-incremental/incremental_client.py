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
import torch

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
    requests.post(server_location+"/get_federated_average")
    federated_average = response.json()
    return federated_average

# Convert data set to Pandas dataframe
csv_file_path = './data/video/node1_data.csv'
data_set = pd.read_csv(csv_file_path)
count = len(data_set)
last_row = data_set.iloc[-1]


def main():
    while True:
        ###STEP 1###
        # Get Hard Coded weights from Server
        initial_params = get_initial_server_data()
        # model = #LEOS MODEL
        # #load weights into the local model
        # model.load_state_dict(initial_params)
        # # Send data set to local model
        # updated_weights = model.train(data_set)
        # # Delete data_set for privacy protections
        # del data_set
        # Assuming the training function for our model returns noisy updates to the weight
        post_server_data(initial_params)
        # Wait ten seconds - So that other nodes can also update the server
        time.sleep(5)

        get_federated_average()

        # # Get updated data that has been AVGed by the server
        # completed_json_file = get_avg_data()

        # ###STEP 2###
        # # subModel2Updates = send completed JSON data to ARFED submodel
        # arfed_weights = AutoEncoderTrainer.trainARFED(completed_json_file)

        # # newWeights = subtract previousweights with subModel2Updates.weights
        # new_weights = [a - b for a, b in zip(previous_weights, arfed_weights)]

        # #  post the new global model weights 
        # post_server_data(new_weights)


        # ###STEP 3###
        # # get request to get the new global model weights
        # updated_serialized_data, updated_epochs, updated_current_weights = get_server_data()
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

