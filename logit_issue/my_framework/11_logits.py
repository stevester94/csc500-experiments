#! /usr/bin/env python3
import json

###########################################
# Build all experiment json parameters
###########################################
experiment_jsons = []

base_parameters = {}

base_parameters["experiment_name"] = "logits!"
base_parameters["lr"] = 0.001
base_parameters["n_epoch"] = 10
base_parameters["batch_size"] = 128
base_parameters["patience"] = 10
base_parameters["device"] = "cuda"
base_parameters["source_domains"] = [-18, -12, -6, 0, 6, 12, 18]
base_parameters["target_domains"] = [-18, -12, -6, 0, 6, 12, 18]

seeds = [
    1337,
    82,
    1234,
    9393,
    1984,
    2017,
    1445,
    51116044,
    16432,
    1792,
    4323,
    6801,
    13309,
    3517,
    121405961,
    19872,
    7250,
    16276,
    16267,
    17534,
    6114,
    16017,
]


base_parameters["x_net"] = [
    {"class": "nnReshape", "kargs": {"shape":[-1,1,2,128]}},
    {"class": "ZeroPad2d", "kargs":{"padding":(2,2,0,0),}},

    {"class": "Conv2d", "kargs": {"in_channels":1, "out_channels":256, "kernel_size":(1,3), "stride":1, "padding":0,}},
    {"class": "ReLU", "kargs": {"inplace":True,}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "ZeroPad2d", "kargs":{"padding":(2,2,0,0),}},

    {"class": "Conv2d", "kargs": {"in_channels":256, "out_channels":80, "kernel_size":(2,3), "stride":1, "padding":0,}},
    {"class": "ReLU", "kargs": {"inplace":True,}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 10560, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace":True,}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 11}},
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
    TRIALS_BASE_PATH=os.path.realpath(os.path.join("./results/", "11_logits")),
    EXPERIMENT_PATH="./experiment"
)
conductor.conduct_experiments(experiment_jsons)
