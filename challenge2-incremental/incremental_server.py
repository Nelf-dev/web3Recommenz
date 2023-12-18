from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import pdb
import numpy as np
import torch  # Import torch to use torch.load

PORT = os.environ.get("PORT", 4001)

app = FastAPI()

captured_weights = []

@app.get("/get_initial_server_data")
def get_state_dict():
    return torch.load('./models/model.pth')

@app.post("/post_server_data")
def post_data(data: dict):
    captured_weights.append(data)

@app.post("/get_federated_average")
def federated_avg():
    arrays = np.array(captured_weights)
    # Calculate the federated average
    return np.mean(arrays, axis=0)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
