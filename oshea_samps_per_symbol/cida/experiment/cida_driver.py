#! /usr/bin/env python3
import os
from numpy.lib.utils import source
import torch
import numpy as np
import os
import sys
import json
import time
from math import floor

from steves_models.configurable_cida import Configurable_CIDA
from steves_utils.cida_train_eval_test_jig import  CIDA_Train_Eval_Test_Jig
from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.lazy_map import Lazy_Map
from steves_utils.sequence_aggregator import Sequence_Aggregator
from steves_utils.oshea_mackey_2020_ds import OShea_Mackey_2020_DS

from steves_utils.torch_utils import (
    confusion_by_domain_over_dataloader,
)

from steves_utils.utils_v2 import (
    per_domain_accuracy_from_confusion
)

from do_report_cida import do_report


# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
TRAINING_CURVES_PATH = os.path.join(RESULTS_DIR, "training_curves.png")
RESULTS_DIAGRAM_PATH = os.path.join(RESULTS_DIR, "results.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
NUM_LOGS_PER_EPOCH = 5



###################################
# Parse Args, Set paramaters
###################################


if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    base_parameters = {}
    base_parameters["experiment_name"] = "Manual Experiment"
    base_parameters["lr"] = 0.001
    base_parameters["n_epoch"] = 20
    base_parameters["batch_size"] = 256
    base_parameters["patience"] = 10
    base_parameters["seed"] = 1337
    base_parameters["device"] = "cuda"

    base_parameters["source_domains"] = [8]
    base_parameters["target_domains"] = [8]


    base_parameters["x_net"] = [
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},
    ]
    base_parameters["u_net"] = [
        {"class": "nnReshape", "kargs": {"shape":[-1, 1]}},
    ]
    base_parameters["merge_net"] = [
        {"class": "Linear", "kargs": {"in_features": 50*58+1, "out_features": 256}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
    ]
    base_parameters["class_net"] = [
        # {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
        # {"class": "ReLU", "kargs": {"inplace": True}},
        # {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
        {"class": "ReLU", "kargs": {"inplace": True}},

        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
    ]
    base_parameters["domain_net"] = [
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 1}},

        # {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
        # {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
        # {"class": "ReLU", "kargs": {"inplace": True}},
        # {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
        # # {"class": "Flatten", "kargs": {"start_dim":0}},
    ]

    base_parameters["device"] = "cuda"

    base_parameters["alpha"] = "sigmoid"
    base_parameters["alpha"] = 0

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

alpha = parameters["alpha"]

start_time_secs = time.time()

###################################
# Clear out results if it already exists
###################################
os.system("rm -rf "+RESULTS_DIR)
os.mkdir(RESULTS_DIR)


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
u_net           = build_sequential(parameters["u_net"])
merge_net       = build_sequential(parameters["merge_net"])
class_net       = build_sequential(parameters["class_net"])
domain_net      = build_sequential(parameters["domain_net"])

###################################
# Build the dataset
###################################

# We append a 1 or 0 to the source and target ds respectively
# This gives us a final tuple of
# (Time domain IQ, label, domain, source?<this is effectively a bool>)

source_ds = OShea_Mackey_2020_DS(samples_per_symbol_to_get=source_domains)
target_ds = OShea_Mackey_2020_DS(samples_per_symbol_to_get=target_domains)


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
source_train_ds, source_val_ds, source_test_ds = torch.utils.data.random_split(source_ds, [source_train_len, source_val_len, source_test_len], generator=torch.Generator().manual_seed(seed))


target_train_len = floor(len(target_ds)*0.7)
target_val_len   = floor(len(target_ds)*0.15)
target_test_len  = len(target_ds) - target_train_len - target_val_len
target_train_ds, target_val_ds, target_test_ds = torch.utils.data.random_split(target_ds, [target_train_len, target_val_len, target_test_len], generator=torch.Generator().manual_seed(seed))


# Normalize the domain and add a 1 if source domain, 0 if target domain
# min_snr = min(OShea_RML2016_DS.get_snrs())
# max_snr = max(OShea_RML2016_DS.get_snrs())

# source_transform_lbda = lambda ex: (ex[0], ex[1], np.array([normalize_val(min_snr, max_snr, ex[2][0])], dtype=np.single) , 1)
# target_transform_lbda = lambda ex: (ex[0], ex[1], np.array([normalize_val(min_snr, max_snr, ex[2][0])], dtype=np.single) , 0)

# add a 1 if source domain, 0 if target domain
source_transform_lbda = lambda ex: (
        ex["IQ"], ex["modulation"], ex["samples_per_symbol"],1
    )
target_transform_lbda = lambda ex: (
        ex["IQ"], ex["modulation"], ex["samples_per_symbol"],0
    )

# We combine our source and target train set. This lets us use unbalanced datasets (IE if we have more source than target)
train_ds = Sequence_Aggregator(
    [
        Lazy_Map(source_train_ds, source_transform_lbda),
        Lazy_Map(target_train_ds, target_transform_lbda), 
    ]
)


train_dl = wrap_in_dataloader(train_ds)

source_val_dl = wrap_in_dataloader(Lazy_Map(source_val_ds, source_transform_lbda))
source_test_dl = wrap_in_dataloader(Lazy_Map(source_test_ds, source_transform_lbda))

target_val_dl = wrap_in_dataloader(Lazy_Map(target_val_ds, target_transform_lbda))
target_test_dl = wrap_in_dataloader(Lazy_Map(target_test_ds, target_transform_lbda))





if alpha == "sigmoid":
    def sigmoid(epoch, total_epochs):
        # This is the same as DANN except we ignore batch
        x = epoch/total_epochs
        gamma = 10
        alpha = 2. / (1. + np.exp(-gamma * x)) - 1

        return alpha

    alpha_func = sigmoid
elif isinstance(alpha, int) or isinstance(alpha, float):
    alpha_func = lambda e,n: alpha # No alpha
else:
    raise Exception("Unknown alpha requested: " + str(alpha))

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


# Debug snippet
# x,y,u,s = next(iter(train_dl))
# print(x)

# print(x_net(x[0].float()).shape)
# print(u_net(x[2].float()).shape)

# sys.exit(1)
# y_hat, u_hat = model.forward(x, u)
# print(u.shape, u_hat.shape)

# print(domain_net)

# sys.exit(1)

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
    train_iterable=train_dl,
    source_val_iterable=source_val_dl,
    target_val_iterable=target_val_dl,
    patience=patience,
    learning_rate=lr,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
    alpha_func=alpha_func
)


###################################
# Colate experiment results
###################################
source_test_label_accuracy, source_test_label_loss, source_test_domain_loss = cida_tet_jig.test(source_test_dl)
target_test_label_accuracy, target_test_label_loss, target_test_domain_loss = cida_tet_jig.test(target_test_dl)
source_val_label_accuracy, source_val_label_loss, source_val_domain_loss = cida_tet_jig.test(source_val_dl)
target_val_label_accuracy, target_val_label_loss, target_val_domain_loss = cida_tet_jig.test(target_val_dl)

history = cida_tet_jig.get_history()

total_epochs_trained = len(history["epoch_indices"])
total_experiment_time_secs = time.time() - start_time_secs

transform_lbda = lambda ex: (
        ex["IQ"], ex["modulation"], ex["samples_per_symbol"]
    )
val_dl = wrap_in_dataloader(Sequence_Aggregator(
    [
        Lazy_Map(source_val_ds, transform_lbda),
        Lazy_Map(target_val_ds, transform_lbda), 
    ]
))

confusion = confusion_by_domain_over_dataloader(model, device, val_dl, forward_uses_domain=True)
per_domain_accuracy = per_domain_accuracy_from_confusion(confusion)

# Add a key to per_domain_accuracy for if it was a source domain
for domain, accuracy in per_domain_accuracy.items():
    per_domain_accuracy[domain] = {
        "accuracy": accuracy,
        "source?": domain in source_domains
    }

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
        "source_test_label_accuracy":source_test_label_accuracy,
        "source_test_label_loss":source_test_label_loss,
        "source_test_domain_loss":source_test_domain_loss,
        "target_test_label_accuracy":target_test_label_accuracy,
        "target_test_label_loss":target_test_label_loss,
        "target_test_domain_loss":target_test_domain_loss,
        "source_val_label_accuracy":source_val_label_accuracy,
        "source_val_label_loss":source_val_label_loss,
        "source_val_domain_loss":source_val_domain_loss,
        "target_val_label_accuracy":target_val_label_accuracy,
        "target_val_label_loss":target_val_label_loss,
        "target_val_domain_loss":target_val_domain_loss,
        "total_epochs_trained": total_epochs_trained,
        "total_experiment_time_secs": total_experiment_time_secs,
        "confusion": confusion,
        "per_domain_accuracy": per_domain_accuracy,    
    },
    "history": history,
}

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

print("Source Val Label Accuracy:", source_val_label_accuracy, "Target Val Label Accuracy:", target_val_label_accuracy)
print("Source Test Label Accuracy:", source_test_label_accuracy, "Target Test Label Accuracy:", target_test_label_accuracy)

do_report(EXPERIMENT_JSON_PATH, TRAINING_CURVES_PATH, RESULTS_DIAGRAM_PATH)