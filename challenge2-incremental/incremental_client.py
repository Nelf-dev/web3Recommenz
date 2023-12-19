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
import submodel_one
from example_weights import weight1, weight2

server_location = "http://localhost:4001"

# def get_server_data():
#     data = requests.get(server_location+"/get_data").json()
#     return data['params'], data['epochs'], data['current_weights']

def get_initial_server_data():
    response = requests.get(server_location+"/get_initial_server_data")
    return response.json()

def post_server_data(updated_weight):
    data = updated_weight
    requests.post(server_location+"/post_server_data", json=data)

def get_federated_average():
    response = requests.get(server_location+"/get_federated_average")
    federated_average = response.json()
    return federated_average

def update_global_model(updated_weight):
    requests.post(server_location+"/update_global_model", json=updated_weight)

def get_global_weights():
    response = requests.get(server_location+"/get_global_weights")
    weights = response.json()
    return weights

# Convert data set to Pandas dataframe
def get_data_set():
    csv_file_path = './data/video/node1_data.csv'
    data_set = pd.read_csv(csv_file_path)
    return data_set

def main():
    while True:
        ###STEP 1###
        # Get Hard Coded weights from Server
        initial_params = get_initial_server_data()
        pdb.set_trace()
        model = submodel_one
        # #load weights into the local model
        # # Send data set to local model
        training_data = get_data_set()
        updated_weights = model.submodel_one(initial_params,training_data)
        # # Delete data_set for privacy protections
        del training_data
        # Assuming the training function for our model returns noisy updates to the weight
        post_server_data(initial_params) # may need to convert if wrong format
        # Wait ten seconds - So that other nodes can also update the server
        time.sleep(0.5)

        # Get federated_average created from server
        averaged_weights = get_federated_average()

        # # subModel2Updates = send completed JSON data to ARFED submodel
        arfed_weights = model.submodel_two(averaged_weights)

        # ###STEP 2###
        pdb.set_trace()

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

