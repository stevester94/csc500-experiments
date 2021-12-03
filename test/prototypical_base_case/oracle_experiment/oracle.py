#! /usr/bin/env python3

import numpy as np

from torch.optim import Adam
import torch

from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.dummy_cida_dataset import Dummy_CIDA_Dataset
from steves_utils.lazy_map import Lazy_Map

from steves_fsl import Steves_Prototypical_Network, split_ds_into_episodes

import steves_utils.ORACLE.torch as ORACLE_Torch
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

seed = 420

import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

torch.use_deterministic_algorithms(True) 

torch.set_default_dtype(torch.float64)


def build_ORACLE_episodic_iterable(
    desired_serial_numbers,
    desired_distances,
    desired_runs,
    window_length,
    window_stride,
    num_examples_per_device,
    seed,
    max_cache_size,
    n_shot,
    n_query,
    n_train_tasks,
    n_val_tasks,
    n_test_tasks,
):

    ds = ORACLE_Torch.ORACLE_Torch_Dataset(
                    desired_serial_numbers=desired_serial_numbers,
                    desired_distances=desired_distances,
                    desired_runs=desired_runs,
                    window_length=window_length,
                    window_stride=window_stride,
                    num_examples_per_device=num_examples_per_device,
                    seed=seed,  
                    max_cache_size=max_cache_size,
                    # transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]),
                    transform_func=lambda x: (torch.from_numpy(x["iq"]), serial_number_to_id(x["serial_number"]), ), # Just (x,y)
                    prime_cache=False
    )

    return split_ds_into_episodes(
        ds=ds,
        n_way=len(desired_serial_numbers),
        n_shot=n_shot,
        n_query=n_query,
        n_train_tasks=n_train_tasks,
        n_val_tasks=n_val_tasks,
        n_test_tasks=n_test_tasks,
        seed=seed,
    )


train_dl, val_dl, test_dl = build_ORACLE_episodic_iterable(
    desired_serial_numbers=[
        "3123D52",
        "3123D65",
        "3123D79",
        "3123D80",
    ],
    # desired_distances=[50,32,8],
    desired_distances=[2],
    desired_runs=[1],
    window_length=256,
    window_stride=50,
    num_examples_per_device=75000,
    seed=420,
    max_cache_size=200000*len(ALL_SERIAL_NUMBERS)*1000,
    n_shot=5,
    n_query=10,
    n_train_tasks=25000,
    n_val_tasks=500,
    n_test_tasks=500,
)

x_net = [
        {"class": "Flatten", "kargs": {}},

        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 512}},
]

x_net = [
        # {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0, "groups":2 },},
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0,  },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},

        {"class": "Linear", "kargs": {"in_features": 5800, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 512}},
]


x_net = build_sequential(x_net)

model = Steves_Prototypical_Network(x_net).cuda()
optimizer = Adam(params=model.parameters())







epoch_train_loss = model.fit(train_dl, optimizer, val_loader=val_dl, validation_frequency=500)

# NUM_EPOCHS = 25

# for epoch in range(NUM_EPOCHS):
#     epoch_train_loss = model.fit(train_dl, optimizer, val_loader=val_dl, validation_frequency=500)
#     accuracy = model.evaluate(test_dl)

#     print(epoch_train_loss)

#     print(f"Average Val Accuracy : {(100 * accuracy):.2f}")
#     print(f"Average Loss: {(epoch_train_loss):.2f}")