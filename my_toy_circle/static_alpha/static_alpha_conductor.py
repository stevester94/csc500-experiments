#! /usr/bin/env python3
import json

###########################################
# Build all experiment json parameters
###########################################
experiment_jsons = []

base_parameters = {}
base_parameters["experiment_name"] = "OShea SNR CIDA "
base_parameters["lr"] = 0.001
base_parameters["n_epoch"] = 3
base_parameters["batch_size"] = 128
base_parameters["patience"] = 10
base_parameters["seed"] = 1337
base_parameters["device"] = "cuda"

# Note that SNRs are used as the arg for the dummy CIDA dataset
base_parameters["source_snrs"] = [-18, -12, -6, 0, 6, 12, 18]
base_parameters["target_snrs"] = [2, 4, 8, 10, -20, 14, 16, -16, -14, -10, -8, -4, -2]

# base_parameters["source_snrs"] = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# base_parameters["target_snrs"] = [0]


base_parameters["x_net"] = [
    {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},
]
base_parameters["u_net"] = [
    {"class": "Identity", "kargs": {}},
]
base_parameters["merge_net"] = [
    {"class": "Linear", "kargs": {"in_features": 50*58+1, "out_features": 256}},
]
base_parameters["class_net"] = [
    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
]
base_parameters["domain_net"] = [
    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
    {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
]

seeds = [1337, 82, 1234, 9393, 1984, 2017, 1445, 511, 
    16044, 16432, 1792, 4323, 6801, 13309, 3517, 12140,
    5961, 19872, 7250, 16276, 16267, 17534, 6114, 16017
]
seeds = [1337]


custom_parameters = [
    {"alpha":"sigmoid"},
    {"alpha":"null"},
]



import copy
for s in seeds:
    for p in custom_parameters:
        parameters = copy.deepcopy(base_parameters)
        for key,val in p.items():
            parameters[key] = val
        parameters["seed"] = s

        j = json.dumps(parameters, indent=2)
        experiment_jsons.append(j)

###########################################
# Run all experiments using Conductor
###########################################
import os
from steves_utils.conductor import Conductor
from steves_utils.utils_v2 import get_past_runs_dir

conductor = Conductor(
    TRIALS_BASE_PATH=os.path.join(get_past_runs_dir(), "test_2")
)
conductor.conduct_experiments(experiment_jsons)