## This is our submission for OpenD/I Hackathon Challenge 4 on Sat, 23 Dec 2024 and presented at Tank Stream Labs on Fri, 22 Dec 2024.

See recorded presentation: https://youtu.be/B1YxLKcBsvk?t=6600

To run this example, you can install the `requirements.txt`

  `pip install -r requirements.txt`

Run the server code:

  `python3.7 incremental_server.py`


Then run some clients:

  `python3.7 incremental_client.py`

  `python3.7 incremental_client.py`

Here is the detailed workflow of what happens:

Step 1 - Local model (ie. initial local model without incremental learning)  
The client file sends a get request to get the initial global model weights that are hard coded from the server file (use regularization techniques for neural nets like dropout and batch normalization).

The client file sends the initial global model weights and the local dataset to the model file (to sub-model 1). In sub-model 1, balance out local dataset if the local dataset is imbalanced, then trains the model (use regularization techniques for neural nets like dropout and batch normalization) with the initial global model weights on the local dataset (ie. train at the local model). For Differential Privacy, during local training, noise terms are added to the gradients of the model to produce the noisy local model updates. Next, return these noisy local model updates to the client file (it is an ordered dictionary containing all the noisy local model weights updates). (the model file just contains functions, there’s no data in there).

Client file post the noisy local model updates to the server file.  
Wait for 10 secs for the server file to receive all noisy local model updates for ALL clients, otherwise the client might be getting the existing weights.
Client file sends a get request to the server file to get all noisy local model updates for ALL clients (ie. the completed json file).
Client sends the completed json file to the model file (to DARFED global sub-model 2).

Step 2 - DARFED model - Global model (ie. global model without incremental learning)  
Sub-model 2 does the DARFED algo to identify outliers. Outliers are determined using the Inter-Quartile Range (IQR) method, where any values sitting below (Q1 - 1.5*IQR) and above (Q3 + 1.5*IQR) are treated as outliers. Q1 = 1st quartile (25th percentile). Q3 = 3rd quartile (75th percentile). IQR = Q3 - Q1.
Aggregate (take simple average of) non-outliers noisy local model updates to produce global non-outlier noisy model updates and return this to the client file.
Client file post the new global model weights (from Step 2.2) to the server file and update the global model weights in the server file.

Step 3 - Incremental model - local model (ie. local model with incremental learning)  
In our example, this happens to only 1 client/node, we are not concerned with other clients.  
The client file sends a get request to get the new global model weights from the server file (similar to Step 1.1).  
The client file sends the local dataset to the model file (to sub-model 3) to generate synthetic local text data and returns this synthetic local text data to the client file (for Generative Replay mechanism in Step 3.4). The client file will also receive one new row of local dataset  (Client file sends the new global model weights (obtained from Step 3.2), the synthetic local text data and the one new row of local dataset to the model file (to incremental learning sub-model 4). Delete the local dataset and the one new row of local dataset from the client file for data privacy protections. (similar to Step 1.2).

In sub-model 4, following the Generative Replay mechanism, score up the synthetic local text data using the new global model (obtained from Step 3.3) to synthesize a local dataset. Create a pandas dataframe recommendation dataset with sentiment ratings = Good from this synthesized local dataset. Also, the one new row of dataset is compared to this synthesized local dataset to detect outlier via anomaly detection using the River library. If not outlier, add a new row to this synthesized local dataset, to give us the new local dataset.

In sub-model 4, balance out the new local dataset if the new local dataset is imbalanced, then the new balanced local dataset will be used to train on the new global model (obtained from Step 3.3) (use regularization techniques for neural nets like dropout and batch normalization) to get local incremental learning model updates. For Differential Privacy, during local training, noise terms are added to the gradients of the model to produce the noisy local incremental learning model updates. Next, sub-model 4 returns the pandas dataframe recommendation dataset to the client file. Sub-model 4 also returns the new local dataset (obtained from Step 3.4) to the client file and then update the local dataset in the client file with this new local dataset for doing Step 3.3 (to send to sub-model 3). In addition, sub-model 4 also returns the noisy local incremental learning model updates (obtained from Step 3.5) to the client file (it is an ordered dictionary containing all the noisy local incremental learning model weights updates). (The model file just contains functions, there’s no data in there) (similar to Step 1.3).  
Client file posts the noisy incremental learning model updates to the server file (similar to Step 1.4).

Wait for 10 secs for the server file to receive all noisy incremental learning model updates for ALL clients, otherwise the client might be getting the existing weights (similar to Step 1.5).  
Client file sends a get request to the server file to get all noisy incremental learning model updates for ALL clients (ie. the completed json file) (similar to Step 1.6). 

Client sends the completed json file to the model file (to DARFED global sub-model 2) (similar to Step 1.7).  
Step 4 - Incremental model - DARFED model - Global model (ie. global model with incremental learning) 
Sub-model 2 does the DARFED algo to identify outliers. Outliers are determined using the Inter-Quartile Range (IQR) method, where any values sitting below (Q1 - 1.5*IQR) and above (Q3 + 1.5*IQR) are treated as outliers. Q1 = 1st quartile (25th percentile). Q3 = 3rd quartile (75th percentile). IQR = Q3 - Q1. (similar to Step 2.1).

Aggregate (take simple average of) non-outliers noisy incremental learning model updates to produce global non-outlier noisy incremental learning model updates and return this to the client file (similar to Step 2.2).  
Client file post the new global model weights (from Step 4.2) to the server file and update the global model weights in the server file (similar to Step 2.3).

Step 5 - For another new row of data, loop Steps 3 & 4 over and over again.
