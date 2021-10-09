#! /usr/bin/env python3
import os
os.system("rm -rf ./steves_utils")
os.system("rm -rf configurable_cida.py")

from configurable_cida import Configurable_CIDA
from steves_utils.cida_train_eval_test_jig import  CIDA_Train_Eval_Test_Jig
from steves_utils.dummy_cida_dataset import Dummy_CIDA_Dataset
from steves_utils.torch_sequential_builder import build_sequential
import torch
import numpy as np
import os
import sys
import json
import time
import inspect
import shutil
from math import floor

# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
NUM_LOGS_PER_EPOCH = 5


###################################
# Parse Args, Set paramaters
###################################
if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    fake_args = {}
    fake_args["experiment_name"] = "Manual Experiment"
    fake_args["lr"] = 0.0001
    fake_args["n_epoch"] = 25
    fake_args["batch_size"] = 128
    fake_args["patience"] = 10
    fake_args["seed"] = 1337
    fake_args["device"] = "cuda"

    fake_args[""] = "cuda"

    fake_args["x_net"] = [
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},
    ]
    fake_args["u_net"] = [
        {"class": "Identity", "kargs": {}},
        {"class": "nnReshape", "kargs": {"shape":[-1, 1]}},
    ]
    fake_args["merge_net"] = [
        {"class": "Linear", "kargs": {"in_features": 50*58+1, "out_features": 256}},
    ]
    fake_args["class_net"] = [
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
        {"class": "Flatten", "kargs": {}},
    ]
    fake_args["domain_net"] = [
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
        {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
        {"class": "Flatten", "kargs": {}},
    ]

    fake_args["device"] = "cuda"

    parameters = fake_args


experiment_name = parameters["experiment_name"]
lr              = parameters["lr"]
n_epoch         = parameters["n_epoch"]
batch_size      = parameters["batch_size"]
patience        = parameters["patience"]
seed            = parameters["seed"]
device          = parameters["device"]

x_net           = build_sequential(parameters["x_net"])
u_net           = build_sequential(parameters["u_net"])
merge_net       = build_sequential(parameters["merge_net"])
class_net       = build_sequential(parameters["class_net"])
domain_net      = build_sequential(parameters["domain_net"])

start_time_secs = time.time()

###################################
# Copy steves utils and all models
# (We do this for reproducibility)
###################################
import time
import steves_utils.utils_v2

utils_path = os.path.dirname(inspect.getfile(steves_utils.utils_v2))
model_path = inspect.getfile(Configurable_CIDA)
model_filename = os.path.basename(model_path)

os.system("cp -R "+ utils_path + " ./")
os.system("cp {} .".format(model_path))

###################################
# Set the RNGs and make it all deterministic
###################################
import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

torch.use_deterministic_algorithms(True) 

###################################
# Build the dataset
###################################
source_ds = Dummy_CIDA_Dataset(
    x_shape=[2,128],
    domains=[1,2,3,4,5,6],
    num_classes=10,
    num_unique_examples_per_class=5000,
    normalize_domain=10
)

target_ds = Dummy_CIDA_Dataset(
    x_shape=[2,128],
    domains=[7,8,9,10],
    num_classes=10,
    num_unique_examples_per_class=5000,
    normalize_domain=10
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


source_train_len = floor(len(source_ds)*0.7)
source_val_len   = floor(len(source_ds)*0.15)
source_test_len  = len(source_ds) - source_train_len - source_val_len
source_train, source_val, source_test = torch.utils.data.random_split(source_ds, [source_train_len, source_val_len, source_test_len], generator=torch.Generator().manual_seed(seed))
source_train, source_val, source_test = (
    wrap_in_dataloader(source_train), wrap_in_dataloader(source_val), wrap_in_dataloader(source_test)
)

target_train_len = floor(len(target_ds)*0.7)
target_val_len   = floor(len(target_ds)*0.15)
target_test_len  = len(target_ds) - target_train_len - target_val_len
target_train, target_val, target_test = torch.utils.data.random_split(target_ds, [target_train_len, target_val_len, target_test_len], generator=torch.Generator().manual_seed(seed))
target_train, target_val, target_test = (
    wrap_in_dataloader(target_train), wrap_in_dataloader(target_val), wrap_in_dataloader(target_test)
)

def sigmoid(epoch, total_epochs):
    # This is the same as DANN except we ignore batch
    x = epoch/total_epochs
    gamma = 10
    alpha = 2. / (1. + np.exp(-gamma * x)) - 1

    return alpha


alpha_func = sigmoid

# TODO: DEBUG
# alpha_func = lambda e,n: 0 # No alpha

###################################
# Build the model
###################################
model = Configurable_CIDA(
    x_net=x_net,
    u_net=u_net,
    merge_net=merge_net,
    class_net=class_net,
    domain_net=domain_net,
    label_loss_object=torch.nn.NLLLoss(),
    domain_loss_object=torch.nn.L1Loss(),
    learning_rate=lr
)


###################################
# Build the tet jig, train
###################################
cida_tet_jig = CIDA_Train_Eval_Test_Jig(
    model=model,
    path_to_best_model=BEST_MODEL_PATH,
    device=torch.device(device),
    label_loss_object=torch.nn.NLLLoss(),
    domain_loss_object=torch.nn.L1Loss(),
)

cida_tet_jig.train(
    source_train_iterable=source_train,
    source_val_iterable=source_val,
    target_train_iterable=target_train,
    target_val_iterable=target_val,
    patience=patience,
    learning_rate=lr,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
    alpha_func=alpha_func
)


###################################
# Colate experiment results
###################################
source_test_label_accuracy, source_test_label_loss, source_test_domain_loss = cida_tet_jig.test(source_test)
target_test_label_accuracy, target_test_label_loss, target_test_domain_loss = cida_tet_jig.test(target_test)

history = cida_tet_jig.get_history()

total_epochs_trained = len(history["epoch_indices"])
total_experiment_time_secs = time.time() - start_time_secs

experiment = {
    "experiment_name": experiment_name,
    "parameters": parameters,
    "results": {
        "source_test_label_accuracy": source_test_label_accuracy,
        "source_test_label_loss": source_test_label_loss,
        "target_test_label_accuracy": target_test_label_accuracy,
        "target_test_label_loss": target_test_label_loss,
        "source_test_domain_loss": source_test_domain_loss,
        "target_test_domain_loss": target_test_domain_loss,
        "total_epochs_trained": total_epochs_trained,
        "total_experiment_time_secs": total_experiment_time_secs,
    },
    "history": history,
}

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

print("Source Test Label Accuracy:", source_test_label_accuracy, "Target Test Label Accuracy:", target_test_label_accuracy)
cida_tet_jig.show_diagram()
cida_tet_jig.save_loss_diagram(LOSS_CURVE_PATH)