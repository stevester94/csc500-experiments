#! /usr/bin/env python3
import json

###########################################
# Build all experiment json parameters
###########################################
experiment_jsons = []

base_parameters = {}

base_parameters["experiment_name"] = "Manual Experiment"
base_parameters["lr"] = 0.001
base_parameters["n_epoch"] = 100
base_parameters["batch_size"] = 128
base_parameters["patience"] = 10
base_parameters["device"] = "cuda"

base_parameters["source_domains"] = [4,6,8]
base_parameters["target_domains"] = [2,10,12,14,16,18,20]

# Original, does not work
# base_parameters["x_net"] = [
#     {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
#     {"class": "ReLU", "kargs": {"inplace": True}},
#     {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
#     {"class": "ReLU", "kargs": {"inplace": True}},
#     {"class": "Dropout", "kargs": {"p": 0.5}},
#     {"class": "Flatten", "kargs": {}},
# ]
# base_parameters["u_net"] = [
#     {"class": "nnReshape", "kargs": {"shape":[-1, 1]}},
# ]
# base_parameters["merge_net"] = [
#     {"class": "Linear", "kargs": {"in_features": 50*58+1, "out_features": 256}},
#     {"class": "ReLU", "kargs": {"inplace": True}},
#     {"class": "Dropout", "kargs": {"p": 0.5}},
# ]
# base_parameters["class_net"] = [
#     # {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
#     # {"class": "ReLU", "kargs": {"inplace": True}},
#     # {"class": "Dropout", "kargs": {"p": 0.5}},

#     {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
#     {"class": "ReLU", "kargs": {"inplace": True}},

#     {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
# ]
# base_parameters["domain_net"] = [
#     {"class": "Linear", "kargs": {"in_features": 256, "out_features": 1}},

#     # {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
#     # {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
#     # {"class": "ReLU", "kargs": {"inplace": True}},
#     # {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
#     # # {"class": "Flatten", "kargs": {"start_dim":0}},
# ]

base_parameters["x_net"] = [
    {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},
]
base_parameters["u_net"] = [
    {"class": "nnReshape", "kargs": {"shape":[-1, 1]}},
    {"class": "Linear", "kargs": {"in_features": 1, "out_features": 10}},
    # {"class": "nnMultiply", "kargs": {"constant":0}},
]
base_parameters["merge_net"] = [
    {"class": "Linear", "kargs": {"in_features": 50*58+10, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
]
base_parameters["class_net"] = [
    # {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
    # {"class": "ReLU", "kargs": {"inplace": True}},
    # {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
    {"class": "ReLU", "kargs": {"inplace": True}},

    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 9}},
]
base_parameters["domain_net"] = [
    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 1}},
    {"class": "nnClamp", "kargs": {"min": 0, "max": 1}},

    # {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
    # {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
    # {"class": "ReLU", "kargs": {"inplace": True}},
    # {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
    {"class": "Flatten", "kargs": {"start_dim":0}},
]



base_parameters["device"] = "cuda"

alphas = [
    "sigmoid",
    0,
    0.25,
    0.5,
    1,
    2
]

base_parameters["num_examples"] = 160000 # This is the size of a single domain

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

import copy
for seed in seeds:
    for alpha in alphas:

        parameters = copy.deepcopy(base_parameters)

        parameters["seed"] = seed
        parameters["alpha"] = alpha

        j = json.dumps(parameters, indent=2)
        experiment_jsons.append(j)

###########################################
# Run all experiments using Conductor
###########################################
import os
from steves_utils.conductor import Conductor

conductor = Conductor(
    TRIALS_BASE_PATH=os.path.realpath(os.path.join("./results/", "trial_1")),
    EXPERIMENT_PATH="./experiment"
)
conductor.conduct_experiments(experiment_jsons)
