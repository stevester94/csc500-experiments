#! /usr/bin/env python3
import torch
import numpy as np

from steves_utils.oshea_RML2016_ds import OShea_RML2016_DS

import torch.nn as nn
import torch.optim as optim

# # Parameters relevant to results
# RESULTS_DIR = "./results"
# BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
# LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
# EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# # Parameters relevant to experiment
# NUM_LOGS_PER_EPOCH = 5

# if not os.path.exists(RESULTS_DIR):
#     os.mkdir(RESULTS_DIR)

###################################
# Parse Args, Set paramaters
###################################
# if len(sys.argv) > 1 and sys.argv[1] == "-":
#     parameters = json.loads(sys.stdin.read())
# elif len(sys.argv) == 1:
#     fake_args = {}
#     fake_args["experiment_name"] = "OShea SNR CNN"
#     fake_args["lr"] = 0.001
#     fake_args["n_epoch"] = 100
#     fake_args["batch_size"] = 128
#     fake_args["patience"] = 100
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
#         {"class": "Linear", "kargs": {"in_features": 80, "out_features": 11}},
#     ]

#     parameters = fake_args


# experiment_name            = parameters["experiment_name"]
# lr                         = parameters["lr"]
# n_epoch                    = parameters["n_epoch"]
# batch_size                 = parameters["batch_size"]
# patience                   = parameters["patience"]
# seed                       = parameters["seed"]
# device                     = torch.device(parameters["device"])
# source_snrs                = parameters["source_snrs"]
# target_snrs                = parameters["target_snrs"]

# start_time_secs = time.time()

###################################
# Parameters
###################################
seed = 1337
batch_size = 128


import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True) 

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1),
            nn.ReLU(False),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(50 * 58, 256),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(256, 80),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(80, 11),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data):
        return self.feature(input_data)

device = torch.device("cuda")
net = CNNModel().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

###################################
# Build the dataset
###################################

ds = OShea_RML2016_DS(
    snrs_to_get=[-18, -12, -6, 0, 6, 12, 18],
)

transform_lambda = lambda ex: ex[:2]
ds = list(map(transform_lambda, ds))

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)

###################################
# Build the tet jig, train
###################################

for epoch in range(100):
    print("Begin Epoch", epoch)
    total_epoch_loss = 0
    total_batches_in_epoch = 0
    for x,y in dl:
        x = x.to(device)
        y = y.to(device)

        logits = net.forward(x)

        loss = criterion(logits, y)

        total_epoch_loss += loss
        total_batches_in_epoch += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Average batch error:", total_epoch_loss/total_batches_in_epoch)