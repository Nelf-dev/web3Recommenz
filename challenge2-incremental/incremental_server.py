from pyhypercycle_aim import SimpleServer, JSONResponseCORS, aim_uri
from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import pdb
import numpy as np
import torch  # Import torch to use torch.load
import json
from collections import OrderedDict


PORT = os.environ.get("PORT", 4001)

# app = FastAPI()

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

class IncrementalExample(SimpleServer):
    manifest = {"name": "IncrementalExample",
                "short_name": "inc-example",
                "version": "0.1",
                "license": "MIT",
                "author": "HyperCycle"
                }
    def __init__(self):
        pass

    @aim_uri(uri="/test", methods=["GET"],
        endpoint_manifest = {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Testing Endpoint",
        })
    def test(self,request):
        data = {"params": "Kung Fu tutorial but with your holding a can of beans"}
        print(data)
        return JSONResponseCORS(data)

    @aim_uri(uri="/get_initial_server_data", methods=["GET"],
        endpoint_manifest = {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Return initial weights"
        })
    def get_state_dict(self, request):
        ordered_dict = torch.load('./models/global_parameters.pt')
        formatted_data = ordered_dict_to_json(ordered_dict)
        return JSONResponseCORS(formatted_data)

    @aim_uri(uri="/post_server_data", methods=["POST"],
        endpoint_manifest = {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Post weights"
        })
    async def post_data(self, data: dict):
        weights = await data.json()
        captured_weights.append(weights)
        print(f"Captured Weights: {len(captured_weights)}")
        return JSONResponseCORS({"updated": "Added weight to server"})

    @aim_uri(uri="/post_server_incremental_data", methods=["POST"],
        endpoint_manifest = {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Post incremental weights"
        })
    async def post_incremental_data(self, data: dict):
        weights = await data.json()
        captured_incremental_weights.append(weights)
        print(f"Captured Incremental Weights: {len(captured_incremental_weights)}")
        return JSONResponseCORS({"updated": "Added Incremental weight to server"})

    @aim_uri(uri="/get_all_weights", methods=["GET"],
        endpoint_manifest = {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Return all weights"
        })
    def get_all_weights(self, request):
        print(f"Returning {len(captured_weights)} Captured Weights")
        return JSONResponseCORS(captured_weights)

    @aim_uri(uri="/get_all_incremental_weights", methods=["GET"],
        endpoint_manifest = {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Return all incremental weights"
        })
    def get_all_incremental_weights(self, request):
        print(f"Returning {len(captured_incremental_weights)} Incremental Captured Weights")
        return JSONResponseCORS(captured_incremental_weights)

# @app.post("/update_global_model")
    @aim_uri(uri="/update_global_model", methods=["POST"],
        endpoint_manifest = {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Update Global Model"
        })
    async def update_global_model(self, data: dict):
        file_path = './models/global_parameters.pt'
        weights = await data.json()
        json_str = json.dumps(weights, indent=2)
        incoming_global_update = json_to_ordered_dict(json_str)
        torch.save(incoming_global_update, file_path)
        return JSONResponseCORS({"updated": "Update Global Model in server"})


# @app.get("/get_global_weights")
    @aim_uri(uri="/get_global_weights", methods=["GET"],
    endpoint_manifest = {
        "input_query": "",
        "input_headers": "",
        "output": {},
        "documentation": "Get Global Model"
    })
    def get_global_model(self, request):
        #return global weights
        ordered_dict = torch.load('./models/global_parameters.pt')
        formatted_data = ordered_dict_to_json(ordered_dict)
        return JSONResponseCORS(formatted_data)

if __name__ == '__main__':
    app = IncrementalExample()
    app.run(uvicorn_kwargs={"port": PORT, "host": "0.0.0.0"})