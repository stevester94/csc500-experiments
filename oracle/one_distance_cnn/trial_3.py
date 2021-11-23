#! /usr/bin/env python3
import json

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

###########################################
# Build all experiment json parameters
###########################################
experiment_jsons = []

base_parameters = {}
base_parameters["window_length"]=256 #Will break if not 256 due to model hyperparameters

base_parameters["experiment_name"] = "One Distance ORACLE CNN"
base_parameters["lr"] = 0.0001
base_parameters["n_epoch"] = 1000
base_parameters["batch_size"] = 256
base_parameters["patience"] = 10
base_parameters["device"] = "cuda"
base_parameters["desired_serial_numbers"] = ALL_SERIAL_NUMBERS

group_2_x = [
    {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0, "groups":2 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 5800, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 17}},
]

group_1_x = [
    {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0},},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 5800, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 17}},
]

import copy

# OK so the experiments from trial 2 were randomized so they're kinda all over the place.
# So we need to do 3 one off experiments

for num_examples_per_device in [15000, 75000, 130000]:
    p = copy.deepcopy(base_parameters)
    p["seed"] = 82
    p["x_net"] = group_1_x
    p["desired_runs"] = [1]
    p["window_stride"] = 50
    p["source_domains"] = [20]
    p["target_domains"] = [20]
    p["num_examples_per_device"]=num_examples_per_device
    j = json.dumps(p, indent=2)
    experiment_jsons.append(j)

    p = copy.deepcopy(base_parameters)
    p["seed"] =1234
    p["x_net"] = group_2_x
    p["desired_runs"] = 2
    p["window_stride"] = 1
    p["source_domains"] = [20]
    p["target_domains"] = [20]
    p["num_examples_per_device"]=num_examples_per_device
    j = json.dumps(p, indent=2)
    experiment_jsons.append(j)

    p = copy.deepcopy(base_parameters)
    p["seed"] = 1234
    p["x_net"] = group_1_x
    p["desired_runs"] = 1
    p["window_stride"] = 1
    p["source_domains"] = [20]
    p["target_domains"] = [20]
    p["num_examples_per_device"]=num_examples_per_device
    j = json.dumps(p, indent=2)
    experiment_jsons.append(j)


###########################################
# Run all experiments using Conductor
###########################################
import os
from steves_utils.conductor import Conductor

conductor = Conductor(
    TRIALS_BASE_PATH=os.path.realpath(os.path.join("./results/", "each_distance_each_run_stride_1", "trial_3")),
    EXPERIMENT_PATH="./experiment",
    KEEP_MODEL=True
)
conductor.conduct_experiments(experiment_jsons)
