import requests
import time
import sys
import random
import pickle
import base64
from incremental_model import AutoEncoderTrainer, AE, reset_seeds #replace AutoEncoderTrainer to Trainer Leo will make
# import spreadsheet

server_location = "http://localhost:4001"

def get_server_data():
    data = requests.get(server_location+"/get_data").json()
    return data['params'], data['epochs']

def post_server_data(params, segment, loss, epochs):
    data = {"params": params, "segment": segment, "loss": loss, "epochs": epochs}
    requests.post(server_location+"/post_data", json=data)



def main():
    epoch = 0
    segment = int(sys.argv[1])
    client_id = sys.argv[2]

    while True:
        # reset_seeds()
        # model = AE() #Change this to Leos Model
        # serialized_data, epochs = get_server_data()
        # state_dict = pickle.loads(base64.b64decode(serialized_data))
        # model.load_state_dict(state_dict)
        # loss = AutoEncoderTrainer.train(model, 1, segment)
        # post_server_data(base64.b64encode(pickle.dumps(model.state_dict())).decode("utf-8"), segment, loss, epochs)
        # print("Client:",client_id, "Epochs:", epochs, "Loss:", loss, "Segment: "+sys.argv[1])
        # time.sleep(1)

        ###################################

        # import oldDataSet
        # import newDataSet (To simulate streamed data)

        # Get Hard Coded weights from Server and save it to a variable(previous weights) if epoch = 0
        defaultWeights,serialized_data, epochs = get_server_data()

        weights = #Previous weights

        if epochs == 0:
            weights = defaultWeights

        # subModel1Updates = Send these weights and OLD local dataset to submodel1
        model = AE()
        state_dict = pickle.loads(base64.b64decode(serialized_data))
        model.load_state_dict(state_dict)

        loss = AutoEncoderTrainer.train(model, 1, segment)


        # send subModel1Updates -  post_server_data(subModel1Updates)
        post_server_data(base64.b64encode(pickle.dumps(model.state_dict())).decode("utf-8"), segment, loss, epochs)

        # Wait ten seconds
        time.sleep()

        # serialized_data, epochs = get_server_data()
        # subModel2Updates = send completed JSON data to ARFED submodel
        # newWeights = subtract previousweights with subModel2Updates.weights
        # subModel3Updates = send newWeights into incremental submodel
        # if newDataSet.count > oldDataSet.count
            # identify the new row of data
            # noisyIncrementalUpdate = send this data to incremental submodel
            # post_server_data(noisyIncrementalUpdate)
            # time.sleep(10)
            # get_server_data()
            # send completed JSON data to ARFED submodel
            # repeat until no new data



if __name__=='__main__':
    main()

