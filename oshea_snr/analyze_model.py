#! /usr/bin/env python3
import os

from torch.nn import parameter
from steves_models.configurable_vanilla import Configurable_Vanilla
from steves_utils.vanilla_train_eval_test_jig import  Vanilla_Train_Eval_Test_Jig
import torch
import numpy as np
import os
import sys
import json
import time
from steves_utils.torch_sequential_builder import build_sequential
import matplotlib.pyplot as plt
import matplotlib.gridspec
from steves_utils.oshea_RML2016_ds import OShea_RML2016_DS
from steves_utils.lazy_map import Lazy_Map
from math import floor

###################################
# Parse Args, Set paramaters
###################################
# if len(sys.argv) > 1 and sys.argv[1] == "-":
#     parameters = json.loads(sys.stdin.read())
# elif len(sys.argv) == 1:
#     fake_args = {}
#     fake_args["experiment_name"] = "OShea SNR CNN"
#     fake_args["lr"] = 0.001
#     fake_args["n_epoch"] = 200
#     fake_args["batch_size"] = 128
#     fake_args["patience"] = 10
#     fake_args["seed"] = 1337
#     fake_args["device"] = "cuda"

#     fake_args["source_snrs"] = [-18, -12, -6, 0, 6, 12, 18]
#     fake_args["target_snrs"] = [2, 4, 8, 10, -20, 14, 16, -16, -14, -10, -8, -4, -2]

#     fake_args["normalize_domain"] = True

#     fake_args["x_net"] = [
#         {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
#         {"class": "ReLU", "kargs": {"inplace": True}},
#         {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
#         {"class": "ReLU", "kargs": {"inplace": True}},
#         {"class": "Dropout", "kargs": {"p": 0.5}},
#         {"class": "Flatten", "kargs": {}},
#         {"class": "Linear", "kargs": {"in_features": 50*58, "out_features": 256}},
#         {"class": "ReLU", "kargs": {"inplace": True}},
#         {"class": "Dropout", "kargs": {"p": 0.5}},
#         {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
#         {"class": "ReLU", "kargs": {"inplace": True}},
#         {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
#     ]

#     parameters = fake_args

weights_path = sys.argv[1]

j = """
{
  "experiment_name": "OShea SNR CNN",
  "lr": 0.001,
  "n_epoch": 200,
  "batch_size": 128,
  "patience": 10,
  "seed": 1337,
  "device": "cuda",
  "source_snrs": [
    -18,
    -12,
    -6,
    0,
    6,
    12,
    18
  ],
  "target_snrs": [
    2,
    4,
    8,
    10,
    -20,
    14,
    16,
    -16,
    -14,
    -10,
    -8,
    -4,
    -2
  ],
  "normalize_domain": true,
  "x_net": [
    {
      "class": "Conv1d",
      "kargs": {
        "in_channels": 2,
        "out_channels": 50,
        "kernel_size": 7,
        "stride": 1,
        "padding": 0
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Conv1d",
      "kargs": {
        "in_channels": 50,
        "out_channels": 50,
        "kernel_size": 7,
        "stride": 2,
        "padding": 0
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Dropout",
      "kargs": {
        "p": 0.5
      }
    },
    {
      "class": "Flatten",
      "kargs": {}
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 2900,
        "out_features": 256
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Dropout",
      "kargs": {
        "p": 0.5
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 256,
        "out_features": 80
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 80,
        "out_features": 16
      }
    }
  ]
}
"""
parameters = json.loads(j)

experiment_name            = parameters["experiment_name"]
lr                         = parameters["lr"]
n_epoch                    = parameters["n_epoch"]
batch_size                 = parameters["batch_size"]
patience                   = parameters["patience"]
seed                       = parameters["seed"]
device                     = parameters["device"]
source_snrs                = parameters["source_snrs"]
target_snrs                = parameters["target_snrs"]
normalize_domain           = parameters["normalize_domain"]
x_net = parameters["x_net"]

start_time_secs = time.time()

###################################
# Set the RNGs and make it all deterministic
###################################
import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


###################################
# Build the network(s)
# Note: It's critical to do this AFTER setting the RNG
###################################
x_net           = build_sequential(parameters["x_net"])


###################################
# Build the dataset
###################################
if normalize_domain:
    min_snr=min(source_snrs+target_snrs)
    max_snr=max(source_snrs+target_snrs)

    nrml = (min_snr, max_snr)
else:
    nrml = None
# Strip off SNR
# This gives us a final tuple of
# (Time domain IQ, label)
source_ds = OShea_RML2016_DS(
    normalize_snr=nrml,
    snrs_to_get=source_snrs,
)

target_ds = OShea_RML2016_DS(
    normalize_snr=nrml,
    snrs_to_get=target_snrs,
)



def wrap_in_dataloader(ds):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=50,
        pin_memory=True
    )


# Split our source and target datasets, wrap them in dataloaders. BUT NOT TRAIN
source_train_len = floor(len(source_ds)*0.7)
source_val_len   = floor(len(source_ds)*0.15)
source_test_len  = len(source_ds) - source_train_len - source_val_len
source_train, source_val, source_test = torch.utils.data.random_split(source_ds, [source_train_len, source_val_len, source_test_len], generator=torch.Generator().manual_seed(seed))
source_val, source_test = (
    wrap_in_dataloader(source_val), wrap_in_dataloader(source_test)
)

target_train_len = floor(len(target_ds)*0.7)
target_val_len   = floor(len(target_ds)*0.15)
target_test_len  = len(target_ds) - target_train_len - target_val_len
target_train, target_val, target_test = torch.utils.data.random_split(target_ds, [target_train_len, target_val_len, target_test_len], generator=torch.Generator().manual_seed(seed))
target_val, target_test = (
    wrap_in_dataloader(target_val), wrap_in_dataloader(target_test)
)

train = source_train
train = wrap_in_dataloader(train)




###################################
# Build the model
###################################
model = Configurable_Vanilla(
    x_net=x_net,
    label_loss_object=torch.nn.NLLLoss(),
    learning_rate=lr
)

###################################
# Evaluate
###################################
dev = torch.device(device)

model.load_state_dict(torch.load(weights_path))

def predict(model, device, batch):

    x,y,u = batch
    x = x.to(device)
    y = y.to(device)
    u = u.to(device)
    y_hat = model.forward(x)
    pred = y_hat.data.max(1, keepdim=True)[1]
    pred = torch.flatten(pred).cpu()

    return pred

def de_normalize_snr(nrml_snr, nrml):
        min_snr = nrml[0]
        max_snr_after_min = nrml[1] - min_snr
        # normalizer_func = lambda snr: (snr-min_snr)/max_snr_after_min

        return int(round(nrml_snr*max_snr_after_min + min_snr))

model = model.to(dev)
model.eval()

source_num_correct_by_snr = {}
source_num_incorrect_by_snr = {}


for h in source_ds:
    print(h[2])

print(nrml)


# for batch in source_val:
#     pred = predict(model, dev, batch)
#     # print(pred)

#     X,Y,U = batch

#     for i in range(X.shape[0]):
#         x,y,u = (X[i],Y[i],U[i])

#         y = int(y)
#         u = float(u)
#         denormalized_snr = de_normalize_snr(u, nrml)
#         print(y, u, denormalized_snr)

#         # if y[i] == pred[i]: source_num_correct_by_snr[denormalized_snr] += 1
#         # else: source_num_incorrect_by_snr[denormalized_snr] += 1

#         # sys.exit(1) 


