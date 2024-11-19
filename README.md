# Web3Recommenz

## Project Overview

This project is a collaborative effort by Leo, Corey, and Nelson. The team has selected Challenge 4, focusing on the implementation of Federated Learning System and Incremental Learning sub-challenges. The goal is to explore and develop solutions in the field of machine learning that address the challenges posed by distributed and continuously evolving data.

## Team Members

- Leo
- Corey
- Nelson

## Our submission for OpenD/I Hackathon Challenge 4 on Sat, 23 Dec 2023 and presented at Tank Stream Labs on Fri, 22 Dec 2023 is in below folder:
https://github.com/Nelf-dev/web3Recommenz/tree/main/federated-incremental

See recorded presentation: https://youtu.be/B1YxLKcBsvk?t=6600

### We also attempted the genetic algorithm, challenge 4.3 and uploaded a very basic version because we did not have enough time.

See: https://github.com/Nelf-dev/web3Recommenz/tree/main/federated-incremental/BONUS-SECTION-prompt-optimization-genetic-algorithm

## Sub-Challenges

### 1. Decentralized Federated Learning System

In the Federated Learning System sub-challenge, our team aims to build a robust and scalable federated learning system. This involves developing algorithms and protocols that enable machine learning models to be trained across decentralized devices while preserving data privacy and security.

### 2. Decentralized Incremental Learning

The Incremental Learning sub-challenge focuses on the development of models that can adapt and learn continuously as new data becomes available. Our team will explore techniques for updating the model with new information without discarding previously acquired knowledge, making it well-suited for scenarios with evolving datasets.

## Data

Dataset was sourced from https://huggingface.co/datasets/OpenGVLab/InternVid/tree/refs%2Fconvert%2Fparquet/default/train based on this paper https://arxiv.org/abs/2307.06942
Download the 0000.parquet file and use the `create_data.py` to generate the datasets.


## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Nelf-dev/web3Recommenz
   ```

2. Make sure you have python 3.7.9 installed

3. Run the server code:
`python incremental_server.py`

4. Then run some clients:

`python incremental_client.py`

`python incremental_client.py`
