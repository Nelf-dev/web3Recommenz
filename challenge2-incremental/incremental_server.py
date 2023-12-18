from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import pdb
import numpy as np
import torch  # Import torch to use torch.load

PORT = os.environ.get("PORT", 4001)

app = FastAPI()

@app.get("/get_initial_server_data")
def get_state_dict():
    return torch.load('./models/model.pth')

# def federated_avg:
#     arrays = np.array(updated_weights_from_clients)
#     # Calculate the federated average
#     return np.mean(arrays, axis=0)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
