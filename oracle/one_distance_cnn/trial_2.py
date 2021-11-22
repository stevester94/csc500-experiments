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
base_parameters["num_examples_per_device"]=260000

base_parameters["experiment_name"] = "One Distance ORACLE CNN"
base_parameters["lr"] = 0.0001
base_parameters["n_epoch"] = 1000
base_parameters["batch_size"] = 256
base_parameters["patience"] = 10
base_parameters["device"] = "cuda"
base_parameters["desired_serial_numbers"] = ALL_SERIAL_NUMBERS

# base_parameters["source_domains"] = ALL_DISTANCES_FEET
# base_parameters["target_domains"] = ALL_DISTANCES_FEET
# base_parameters["desired_runs"]=[1]
# base_parameters["window_stride"]=50

# seeds = [1337, 82, 1234, 9393, 1984, 2017, 1445, 511, 
#     16044, 16432, 1792, 4323, 6801, 13309, 3517, 12140,
#     5961, 19872, 7250, 16276, 16267, 17534, 6114, 16017
# ]
seeds = [82, 1234]

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


custom_distances = [
    # [2],
    [20],
    [38],
    [56],
    [62],
]

custom_x_nets = [group_2_x, group_1_x]

custom_runs = [[1],[2]]

custom_strides = [1,25,50]

import copy
for seed in seeds:
    for domains in custom_distances:
        for x_net in custom_x_nets:
            for desired_runs in custom_runs:
                for window_stride in custom_strides:
                    parameters = copy.deepcopy(base_parameters)

                    parameters["seed"] = seed
                    parameters["x_net"] = x_net
                    parameters["desired_runs"] = desired_runs
                    parameters["window_stride"] = window_stride
                    parameters["source_domains"] = domains
                    parameters["target_domains"] = domains

                    j = json.dumps(parameters, indent=2)
                    experiment_jsons.append(j)

import random
random.seed(1337)
random.shuffle(experiment_jsons)
experiment_jsons = experiment_jsons[77:]

###########################################
# Run all experiments using Conductor
###########################################
import os
from steves_utils.conductor import Conductor
from steves_utils.utils_v2 import get_past_runs_dir

conductor = Conductor(
    TRIALS_BASE_PATH=os.path.realpath(os.path.join("./results/", "each_distance_each_run_stride_1", "trial_2")),
    EXPERIMENT_PATH="./experiment"
)
conductor.conduct_experiments(experiment_jsons)
