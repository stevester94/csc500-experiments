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
base_parameters["experiment_name"] = "Prototypical ORACLE Base Case"
base_parameters["lr"] = 0.001
# base_parameters["n_epoch"] = 10
# base_parameters["batch_size"] = 256
# base_parameters["patience"] = 10
# base_parameters["seed"] = 1337
# base_parameters["device"] = "cuda"
base_parameters["desired_serial_numbers"] = [
    "3123D52",
    "3123D65",
    "3123D79",
    "3123D80",
]
base_parameters["source_domains"] = [2]
# base_parameters["target_domains"] = list(set(ALL_DISTANCES_FEET) - set([50,32,8]))

base_parameters["window_stride"]=50
base_parameters["window_length"]=256
base_parameters["desired_runs"]=[1]
base_parameters["num_examples_per_device"]=75000

# base_parameters["n_shot"]  = 
# base_parameters["n_query"]  = 10
# base_parameters["n_train_tasks"] = 500

base_parameters["n_val_tasks"]  = 1000
base_parameters["n_test_tasks"]  = 100
base_parameters["validation_frequency"] = 1000

base_parameters["n_epoch"] = 100
base_parameters["patience"] = 10

custom_x_net = [
    [# droupout, groups, 512 out
        # {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0, "groups":2 },},
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0, },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        # {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},

        {"class": "Linear", "kargs": {"in_features": 5800, "out_features": 512}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        # {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 512, "out_features": 512}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        # {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 512, "out_features": 512}},
    ]


]

custom_n_query = [10, 20, 50]
custom_n_train_tasks = [5000]
custom_seed = [1337, 420, 69, 134231, 98453]

import copy
for seed in custom_seed:
    for n_query in custom_n_query:
        for n_train_tasks in custom_n_train_tasks:
            for x_net in custom_x_net:
                parameters = copy.deepcopy(base_parameters)

                parameters["seed"]          = seed
                parameters["n_query"]       = n_query
                parameters["n_train_tasks"] = n_train_tasks
                parameters["x_net"]         = x_net

                j = json.dumps(parameters, indent=2)
                experiment_jsons.append(j)

# import random
# random.seed(1337)
# random.shuffle(experiment_jsons)


###########################################
# Run all experiments using Conductor
###########################################
import os
from steves_utils.conductor import Conductor
from steves_utils.utils_v2 import get_past_runs_dir

conductor = Conductor(
    TRIALS_BASE_PATH=os.path.realpath(os.path.join("./oracle_results/", "trial_1")),
    EXPERIMENT_PATH="./oracle_experiment",
    KEEP_MODEL=False
)
conductor.conduct_experiments(experiment_jsons)
