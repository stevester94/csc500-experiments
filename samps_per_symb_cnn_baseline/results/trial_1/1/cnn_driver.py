#! /usr/bin/env python3
import os
import torch
import numpy as np
import os
import sys
import json
import time
from math import floor

from steves_models.configurable_vanilla import Configurable_Vanilla
from steves_utils.vanilla_train_eval_test_jig import  Vanilla_Train_Eval_Test_Jig
from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.lazy_map import Lazy_Map
from steves_utils.sequence_aggregator import Sequence_Aggregator
from steves_utils.oshea_mackey_2020_ds import OShea_Mackey_2020_DS
# from steves_utils.oshea_RML2016_ds import OShea_RML2016_DS


from steves_utils.torch_utils import (
    confusion_by_domain_over_dataloader,
)

from steves_utils.utils_v2 import (
    per_domain_accuracy_from_confusion
)

from do_report import do_report


# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
NUM_LOGS_PER_EPOCH = 5

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


###################################
# Parse Args, Set paramaters
###################################
if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    base_parameters = {}
    base_parameters["experiment_name"] = "manual oshea snr 2"
    base_parameters["lr"] = 0.001
    base_parameters["n_epoch"] = 100
    base_parameters["batch_size"] = 128
    base_parameters["patience"] = 10
    base_parameters["seed"] = 1337
    base_parameters["device"] = "cuda"
    base_parameters["source_domains"] = [8]
    base_parameters["target_domains"] = [2,4,6,10,12,14,16,18,20]

    base_parameters["snrs_to_get"] = [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    base_parameters["x_net"] = [
        # Lol this works
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
        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 12}},
    ]

    parameters = base_parameters


experiment_name         = parameters["experiment_name"]
lr                      = parameters["lr"]
n_epoch                 = parameters["n_epoch"]
batch_size              = parameters["batch_size"]
patience                = parameters["patience"]
seed                    = parameters["seed"]
device                  = torch.device(parameters["device"])

source_domains         = parameters["source_domains"]
target_domains         = parameters["target_domains"]

snrs_to_get     = parameters["snrs_to_get"]

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
# Build the network(s)
# Note: It's critical to do this AFTER setting the RNG
###################################
x_net           = build_sequential(parameters["x_net"])


###################################
# Build the dataset
###################################

# DS_PATH = "/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-utils/oshea_original/dataset/RML2016.10a_dict.dat"
# source_ds = OShea_Mackey_2020_DS(path=DS_PATH, samples_per_symbol_to_get=source_domains)
# target_ds = OShea_Mackey_2020_DS(path=DS_PATH, samples_per_symbol_to_get=target_domains)

# DS_PATH = "/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-utils/oshea_original/dataset/RML2016.10a_dict.dat"
source_ds = OShea_Mackey_2020_DS(snrs_to_get=snrs_to_get, samples_per_symbol_to_get=source_domains)
target_ds = OShea_Mackey_2020_DS(snrs_to_get=snrs_to_get, samples_per_symbol_to_get=target_domains)



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


# Split our source and target datasets into train val and test
source_train_len = floor(len(source_ds)*0.7)
source_val_len   = floor(len(source_ds)*0.15)
source_test_len  = len(source_ds) - source_train_len - source_val_len
source_train_ds, source_val_ds, source_test_ds = torch.utils.data.random_split(source_ds, [source_train_len, source_val_len, source_test_len], generator=torch.Generator().manual_seed(seed))


target_train_len = floor(len(target_ds)*0.7)
target_val_len   = floor(len(target_ds)*0.15)
target_test_len  = len(target_ds) - target_train_len - target_val_len
target_train_ds, target_val_ds, target_test_ds = torch.utils.data.random_split(target_ds, [target_train_len, target_val_len, target_test_len], generator=torch.Generator().manual_seed(seed))

# For CNN We only use X and Y. And we only train on the source.
# Properly form the data using a transform lambda and Lazy_Map. Finally wrap them in a dataloader

transform_lambda = lambda ex: (ex["IQ"], ex["modulation"]) # Strip the tuple to just (x,y)

# CIDA combines source and target training sets into a single dataloader, that's why this one is just called train_dl
train_dl = wrap_in_dataloader(
    Lazy_Map(source_train_ds, transform_lambda)
)

source_val_dl = wrap_in_dataloader(
    Lazy_Map(source_val_ds, transform_lambda)
)
source_test_dl = wrap_in_dataloader(
    Lazy_Map(source_test_ds, transform_lambda)
)

target_val_dl = wrap_in_dataloader(
    Lazy_Map(target_val_ds, transform_lambda)
)
target_test_dl  = wrap_in_dataloader(
    Lazy_Map(target_test_ds, transform_lambda)
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
jig = Vanilla_Train_Eval_Test_Jig(
    model=model,
    path_to_best_model=BEST_MODEL_PATH,
    device=device,
    label_loss_object=torch.nn.NLLLoss(),
)

jig.train(
    train_iterable=train_dl,
    val_iterable=source_val_dl,
    patience=patience,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
)


###################################
# Evaluate the model
###################################
source_test_label_accuracy, source_test_label_loss = jig.test(source_test_dl)
target_test_label_accuracy, target_test_label_loss = jig.test(target_test_dl)

source_val_label_accuracy, source_val_label_loss = jig.test(source_val_dl)
target_val_label_accuracy, target_val_label_loss = jig.test(target_val_dl)

history = jig.get_history()

total_epochs_trained = len(history["epoch_indices"])

transform_lambda = lambda ex: (ex["IQ"], ex["modulation"], ex["snr"]) # Strip the tuple to just (x,y,u)
val_dl = wrap_in_dataloader(
    Sequence_Aggregator(
        (
            Lazy_Map(source_val_ds, transform_lambda), 
            Lazy_Map(target_val_ds, transform_lambda),
        )
    )
)

confusion = confusion_by_domain_over_dataloader(model, device, val_dl, forward_uses_domain=False)
per_domain_accuracy = per_domain_accuracy_from_confusion(confusion)

# Add a key to per_domain_accuracy for if it was a source domain
for domain, accuracy in per_domain_accuracy.items():
    per_domain_accuracy[domain] = {
        "accuracy": accuracy,
        "source?": domain in source_domains
    }

total_experiment_time_secs = time.time() - start_time_secs

###################################
# Write out the results
###################################

experiment = {
    "experiment_name": experiment_name,
    "parameters": parameters,
    "results": {
        "source_test_label_accuracy": source_test_label_accuracy,
        "source_test_label_loss": source_test_label_loss,
        "target_test_label_accuracy": target_test_label_accuracy,
        "target_test_label_loss": target_test_label_loss,
        "source_val_label_accuracy": source_val_label_accuracy,
        "source_val_label_loss": source_val_label_loss,
        "target_val_label_accuracy": target_val_label_accuracy,
        "target_val_label_loss": target_val_label_loss,
        "total_epochs_trained": total_epochs_trained,
        "total_experiment_time_secs": total_experiment_time_secs,
        "confusion": confusion,
        "per_domain_accuracy": per_domain_accuracy,
    },
    "history": history,
}



print("Source Test Label Accuracy:", source_test_label_accuracy, "Target Test Label Accuracy:", target_test_label_accuracy)
print("Source Val Label Accuracy:", source_val_label_accuracy, "Target Val Label Accuracy:", target_val_label_accuracy)

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)


###################################
# Make the report
###################################
do_report(EXPERIMENT_JSON_PATH, LOSS_CURVE_PATH)
