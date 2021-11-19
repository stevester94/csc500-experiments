#! /usr/bin/env python3
import json

###########################################
# Build all experiment json parameters
###########################################
experiment_jsons = []

base_parameters = {}

base_parameters["experiment_name"] = "CNN SampsPerSymbol Train One Test Several Trial 2"
base_parameters["lr"] = 0.001
base_parameters["n_epoch"] = 1000
base_parameters["batch_size"] = 256
base_parameters["patience"] = 10
base_parameters["device"] = "cuda"
base_parameters["source_domains"] = [8]
base_parameters["target_domains"] = [2,6,10,12]

seeds = [
    1337,
    5748,
    14195,
    15493,
    14209,
    15572,
    43,
    179,
    1316,
    6948,
    3854,
    12698,
    15124,
    4954,
    5578,
    3764,
]


base_parameters["x_net"] = [
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},

        {"class": "Linear", "kargs": {"in_features": 50*58, "out_features": 256}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
        {"class": "ReLU", "kargs": {"inplace": True}},

        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 8}},
]


import copy
for seed in seeds:

    parameters = copy.deepcopy(base_parameters)

    parameters["seed"] = seed

    j = json.dumps(parameters, indent=2)
    experiment_jsons.append(j)

###########################################
# Run all experiments using Conductor
###########################################
import os
from steves_utils.conductor import Conductor

conductor = Conductor(
    TRIALS_BASE_PATH=os.path.realpath(os.path.join("./results/", "SampsPerSymbolTrainOneTestSeveral", "trial_2")),
    EXPERIMENT_PATH="./experiment"
)
conductor.conduct_experiments(experiment_jsons)
