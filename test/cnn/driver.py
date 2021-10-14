#! /usr/bin/env python3
from configurable_vanilla import Configurable_Vanilla
from steves_utils.vanilla_train_eval_test_jig import  Vanilla_Train_Eval_Test_Jig
import torch
import numpy as np
import os
import sys
import json
import time
from steves_utils.torch_sequential_builder import build_sequential


# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
SHAPE_DATA = [2,128]
NUM_CLASSES = 16
NUM_LOGS_PER_EPOCH = 5

# TODO: DEBUG
NUM_BATCHES = 10000


###################################
# Parse Args, Set paramaters
###################################
if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    fake_args = {}
    fake_args["experiment_name"] = "Manual Experiment"
    fake_args["lr"] = 0.00001
    fake_args["n_epoch"] = 100
    fake_args["batch_size"] = 1280
    fake_args["patience"] = 10
    fake_args["seed"] = 1337
    fake_args["device"] = "cuda"

    fake_args["source_snrs"] = [0, 2, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
    fake_args["target_snrs"] = [4]

    fake_args["x_net"] = [
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
        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
    ]

    parameters = fake_args


experiment_name = parameters["experiment_name"]
lr              = parameters["lr"]
n_epoch         = parameters["n_epoch"]
batch_size      = parameters["batch_size"]
patience        = parameters["patience"]
seed            = parameters["seed"]
device          = parameters["device"]
x_net           = build_sequential(parameters["x_net"])

start_time_secs = time.time()


###################################
# Build the dataset
###################################
torch.set_default_dtype(torch.float64) # Never leave home without it


x = np.ones(256*NUM_BATCHES, dtype=np.double)
x = np.reshape(x, [NUM_BATCHES] + SHAPE_DATA)
x = torch.from_numpy(x)

y = np.ones(NUM_BATCHES, dtype=np.double)
y = torch.from_numpy(y).long()

dl = torch.utils.data.DataLoader(
    list(zip(x,y)),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)


###################################
# Build the model
###################################
model = Configurable_Vanilla(
    x_net=x_net,
    label_loss_object=torch.nn.NLLLoss(),
    learning_rate=lr
)


###################################
# Build the tet jig, train
###################################
vanilla_tet_jig = Vanilla_Train_Eval_Test_Jig(
    model,
    torch.nn.NLLLoss(),
    BEST_MODEL_PATH,
    torch.device(device)
)

vanilla_tet_jig.train(
    train_iterable=dl,
    val_iterable=dl,
    patience=patience,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
)


###################################
# Colate experiment results
###################################
test_label_accuracy, test_label_loss = vanilla_tet_jig.test(dl)
history = vanilla_tet_jig.get_history()

total_epochs_trained = len(history["epoch_indices"])
total_experiment_time_secs = time.time() - start_time_secs

experiment = {
    "experiment_name": experiment_name,
    "parameters": parameters,
    "results": {
        "test_label_accuracy": test_label_accuracy,
        "test_label_loss": test_label_loss,
        "total_epochs_trained": total_epochs_trained,
        "total_experiment_time_secs": total_experiment_time_secs,
    },
    "history": history,
}

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

vanilla_tet_jig.show_diagram(LOSS_CURVE_PATH)
