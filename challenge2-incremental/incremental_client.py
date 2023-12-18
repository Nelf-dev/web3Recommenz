import requests
import time
import sys
import random
import pickle
import base64
from incremental_model import AutoEncoderTrainer, AE, reset_seeds #replace AutoEncoderTrainer to Trainer Leo will make
import pdb
import pandas as pd

server_location = "http://localhost:4001"

def get_server_data():
    data = requests.get(server_location+"/get_data").json()
    return data['params'], data['epochs'], data['current_weights']

def get_initial_server_data():
    response = requests.get(server_location+"/get_initial_server_data")
    state_dict = response.json()
    return state_dict

def post_server_data(params, segment, loss, epochs):
    data = {"params": params, "segment": segment, "loss": loss, "epochs": epochs}
    requests.post(server_location+"/post_data", json=data)

csv_file_path = './data/video/node1_data.csv'
data_set = pd.read_csv(csv_file_path)
count = len(data_set)
last_row = data_set.iloc[-1]


def main():
    epoch = 0
    segment = int(sys.argv[1])
    client_id = sys.argv[2]

    while True:
        initial_data = get_initial_server_data()
        pdb.set_trace()
        ###STEP 1###
        # Get Hard Coded weights from Server and save it to a variable(previous weights) if epoch = 0
        serialized_data, epochs, current_weights = get_server_data()

        # Liase with Corey how to add defaultWeights Parameter
        weights = current_weights

        pdb.set_trace()
        #weights = #Previous weights
        if epochs == 0:
            weights = serialized_data

        model = AE()

        #convert decode serialized data to get weights
        state_dict = pickle.loads(base64.b64decode(serialized_data))
        #load weights into the local model
        model.load_state_dict(state_dict)
        # Convert data set to Pandas dataframe
        data_set = pd.read_csv(csv_file_path)
        # Send data set to local model
        loss = AutoEncoderTrainer.train(model, 1, segment, data_set)
        # Delete data_set for privacy protections
        del data_set
        # Assuming the training function for our model returns noisy updates to the weight
        # not sure how to just send noisy local model weights updates, maybe as a new parameter?
        post_server_data(base64.b64encode(pickle.dumps(model.state_dict())).decode("utf-8"), segment, loss, epochs)
        # Wait ten seconds - So that other nodes can also update the server
        time.sleep(10)

        # Get updated data that has been AVGed by the server
        completed_json_file = get_avg_data()

        ###STEP 2###
        # subModel2Updates = send completed JSON data to ARFED submodel
        arfed_weights = AutoEncoderTrainer.trainARFED(completed_json_file)

        # newWeights = subtract previousweights with subModel2Updates.weights
        new_weights = [a - b for a, b in zip(previous_weights, arfed_weights)]

        #  post the new global model weights 
        post_server_data(new_weights)


        ###STEP 3###
        # get request to get the new global model weights
        updated_serialized_data, updated_epochs, updated_current_weights = get_server_data()
        # if new_data_set.count > old_data_set.count:
        streamed_data = './data/video/node1_incremental_1.csv'
        streamed_row = pd.read_csv(streamed_data)
        streamed_last_row = data_set.iloc[-1]

        updated_state_dict = pickle.loads(base64.b64decode(updated_serialized_data))
        #load weights into the local model
        model.load_state_dict(updated_state_dict)

        # subModel3Updates = send newWeights into incremental submodel
        incrementedLearningUpdates = AutoEncoderTrainer.trainIncremental(updated_serialized_data, streamed_last_row)
        #delete the streamed row of data for data privacy
        del streamed_last_row

        # Send updates to server
        post_server_data(incrementedLearningUpdates)

        #wait 10 seconds for other clients to also send their incremental updates
        time.sleep(10)

        # get all noisy incremental learning model updates for ALL clients
        update_completed_json_file = get_avg_data()

        # ARFED the JSON file of all the model updates
        arfed_weights = AutoEncoderTrainer.trainARFED(completed_json_file)

        # repeat this if there is another new row of data



if __name__=='__main__':
    main()

