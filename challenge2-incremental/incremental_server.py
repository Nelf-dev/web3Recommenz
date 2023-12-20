from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import pdb
import numpy as np
import torch  # Import torch to use torch.load
import json
from collections import OrderedDict


PORT = os.environ.get("PORT", 4001)

app = FastAPI()

captured_weights = []
captured_incremental_weights = []

def ordered_dict_to_json(ordered_dict):
    model_state_dict_serializable = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in ordered_dict.items()}
    return json.dumps(model_state_dict_serializable)

def json_to_ordered_dict(json_str):
    json_data = json.loads(json_str)
    # If the original values were torch tensors, convert them back
    for key, value in json_data.items():
        if isinstance(value, list):
            json_data[key] = torch.tensor(value)

    ordered_dict = OrderedDict(json_data)
    return ordered_dict

@app.get("/get_initial_server_data")
def get_state_dict():
    ordered_dict = torch.load('./models/global_parameters.pt')
    formatted_data = ordered_dict_to_json(ordered_dict)
    return formatted_data

@app.post("/post_server_data")
def post_data(data: dict):
    captured_weights.append(data)
    print(f"Captured Weights: {len(captured_weights)}")

@app.post("/post_server_incremental_data")
def post_incremental_data(data: dict):
    captured_incremental_weights.append(data)
    print(f"Captured Incremental Weights: {len(captured_incremental_weights)}")

@app.get("/get_all_weights")
def get_all_weights():
    return captured_weights

@app.get("/get_all_incremental_weights")
def get_all_incremental_weights():
    return captured_incremental_weights

@app.post("/update_global_model")
def update_global_model(data: dict):
    file_path = './models/global_parameters.pt'
    json_str = json.dumps(data, indent=2)
    incoming_global_update = json_to_ordered_dict(json_str)
    torch.save(incoming_global_update, file_path)

@app.get("/get_global_weights")
def get_global_model():
    #return global weights
    ordered_dict = torch.load('./models/global_parameters.pt')
    formatted_data = ordered_dict_to_json(ordered_dict)
    return formatted_data

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
