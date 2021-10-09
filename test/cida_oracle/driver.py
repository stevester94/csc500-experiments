#! /usr/bin/env python3
from cida_rf_chapter_2 import CIDA_RF_CNN_Model_Chapter_Two
from steves_utils.cida_train_eval_test_jig import  CIDA_Train_Eval_Test_Jig
from steves_utils.dummy_cida_dataset import Dummy_CIDA_Dataset
from steves_utils.ORACLE.torch import ORACLE_Torch_Dataset
from steves_utils.ORACLE.utils_v2 import ALL_RUNS, ALL_DISTANCES_FEET, ALL_SERIAL_NUMBERS, serial_number_to_id
import torch
import numpy as np
import os
import sys
import json
import time
from math import floor

# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
NUM_CLASSES = -1337
NUM_LOGS_PER_EPOCH = 10


###################################
# Parse Args, Set paramaters
###################################
if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    fake_args = {}
    fake_args["experiment_name"] = "Manual Experiment"
    fake_args["lr"] = 0.0001
    fake_args["n_epoch"] = 100
    fake_args["batch_size"] = 256
    fake_args["patience"] = 50
    fake_args["seed"] = 15474
    fake_args["device"] = "cuda"
    parameters = fake_args


experiment_name = parameters["experiment_name"]
lr              = parameters["lr"]
n_epoch         = parameters["n_epoch"]
batch_size      = parameters["batch_size"]
patience        = parameters["patience"]
seed            = parameters["seed"]
device          = parameters["device"]

start_time_secs = time.time()

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

source_ds = ORACLE_Torch_Dataset(
    desired_distances=[2],
    desired_runs=[1],
    # desired_serial_numbers=["3123D52","3123D65","3123D79"],
    desired_serial_numbers=ALL_SERIAL_NUMBERS,
    # desired_distances=ALL_DISTANCES_FEET,
    # desired_runs=ALL_RUNS,
    window_length=256,
    window_stride=50,
    num_examples_per_device=200000,
    seed=seed,
    max_cache_size=100000*16, 
    transform_func=lambda x: (x["iq"].astype(np.single), serial_number_to_id(x["serial_number"]), x["distance_ft"]/62.0)
    # max_cache_size=1
)

target_ds = ORACLE_Torch_Dataset(
    desired_distances=[8],
    desired_runs=[1],
    # desired_serial_numbers=["3123D52","3123D65","3123D79"],
    desired_serial_numbers=ALL_SERIAL_NUMBERS,
    # desired_distances=ALL_DISTANCES_FEET,
    # desired_runs=ALL_RUNS,
    window_length=256,
    window_stride=50,
    num_examples_per_device=200000,
    seed=seed,
    max_cache_size=100000*16,
    transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]/62.0)
    # max_cache_size=1
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
alpha_func = lambda e,n: 0 # No alpha

###################################
# Build the model
###################################
model = CIDA_RF_CNN_Model_Chapter_Two(
    NUM_CLASSES,
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
