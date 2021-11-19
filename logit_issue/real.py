#! /usr/bin/env python3
import os
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
import pandas as pds


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
    fake_args = {}
    fake_args["experiment_name"] = "OShea SNR CNN"
    fake_args["lr"] = 0.001
    fake_args["n_epoch"] = 100
    fake_args["batch_size"] = 128
    fake_args["patience"] = 10
    fake_args["seed"] = 1337
    fake_args["device"] = "cuda"

    fake_args["source_snrs"] = [-18, -12, -6, 0, 6, 12, 18]
    fake_args["target_snrs"] = [2, 4, 8, 10, -20, 14, 16, -16, -14, -10, -8, -4, -2]

    fake_args["normalize_domain"] = True

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
        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 11}},
    ]

    parameters = fake_args


experiment_name            = parameters["experiment_name"]
lr                         = parameters["lr"]
n_epoch                    = parameters["n_epoch"]
batch_size                 = parameters["batch_size"]
patience                   = parameters["patience"]
seed                       = parameters["seed"]
device                     = torch.device(parameters["device"])
source_snrs                = parameters["source_snrs"]
target_snrs                = parameters["target_snrs"]

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

source_ds = OShea_RML2016_DS(
    snrs_to_get=source_snrs,
)

target_ds = OShea_RML2016_DS(
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
source_train_ds, source_val_ds, source_test_ds = torch.utils.data.random_split(source_ds, [source_train_len, source_val_len, source_test_len], generator=torch.Generator().manual_seed(seed))


target_train_len = floor(len(target_ds)*0.7)
target_val_len   = floor(len(target_ds)*0.15)
target_test_len  = len(target_ds) - target_train_len - target_val_len
target_train_ds, target_val_ds, target_test_ds = torch.utils.data.random_split(target_ds, [target_train_len, target_val_len, target_test_len], generator=torch.Generator().manual_seed(seed))

# Properly form the data using a transform lambda and Lazy_Map. Finally wrap them in a dataloader
# Train ds gets treated a little special in cida, we mirror the structure here

transform_lambda = lambda ex: ex[:2]

train_ds = source_train_ds
train_dl = wrap_in_dataloader(
    Lazy_Map(train_ds, transform_lambda)
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
vanilla_tet_jig = Vanilla_Train_Eval_Test_Jig(
    model=model,
    path_to_best_model=BEST_MODEL_PATH,
    device=device,
    label_loss_object=torch.nn.NLLLoss(),
)

vanilla_tet_jig.train(
    train_iterable=train_dl,
    val_iterable=source_val_dl,
    patience=patience,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
)


###################################
# Colate experiment results
###################################
source_test_label_accuracy, source_test_label_loss = vanilla_tet_jig.test(source_test_dl)
target_test_label_accuracy, target_test_label_loss = vanilla_tet_jig.test(target_test_dl)

source_val_label_accuracy, source_val_label_loss = vanilla_tet_jig.test(source_val_dl)
target_val_label_accuracy, target_val_label_loss = vanilla_tet_jig.test(target_val_dl)

history = vanilla_tet_jig.get_history()

total_epochs_trained = len(history["epoch_indices"])



def predict(model, device, batch):

    x,y,u = batch
    x = x.to(device)
    y = y.to(device)
    u = u.to(device)
    y_hat = model.forward(x)
    pred = y_hat.data.max(1, keepdim=True)[1]
    pred = torch.flatten(pred).cpu()

    return pred

def accuracy_by_domain(dict, dl):
    for batch in dl:
        pred = predict(model, device, batch)

        X,Y,U = batch

        for i in range(X.shape[0]):
            x,y,u = (X[i],Y[i],U[i])
            p = pred[i]

            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            u = tuple(u.cpu().detach().numpy().tolist())
            p = p.cpu().detach().numpy()

            if y == p: dict[u][0] += 1
            else: dict[u][1] += 1

source_val_dl = wrap_in_dataloader(source_val_ds)
target_val_dl = wrap_in_dataloader(target_val_ds)

source_count_by_domain = {}
target_count_by_domain = {}

for s in source_snrs:
    source_count_by_domain[(s,)] = [0,0]

for s in target_snrs:
    target_count_by_domain[(s,)] = [0,0]


accuracy_by_domain(source_count_by_domain, source_val_dl)
accuracy_by_domain(target_count_by_domain, target_val_dl)

accuracies_by_domain = {
    "source": [],
    "accuracy": [],
    "domain": [],
}

for key,val in source_count_by_domain.items():
    accuracies_by_domain["source"].append(True)
    accuracies_by_domain["accuracy"].append(val[0] / (val[0] + val[1]))
    accuracies_by_domain["domain"].append(key[0]) # Breaking the assumption of vector domains here

for key,val in target_count_by_domain.items():
    accuracies_by_domain["source"].append(False)
    accuracies_by_domain["accuracy"].append(val[0] / (val[0] + val[1]))
    accuracies_by_domain["domain"].append(key[0]) # Breaking the assumption of vector domains here


total_experiment_time_secs = time.time() - start_time_secs



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
        "val_accuracies_by_domain": accuracies_by_domain 
    },
    "history": history,
}

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

print("Source Test Label Accuracy:", source_test_label_accuracy, "Target Test Label Accuracy:", target_test_label_accuracy)
print("Source Val Label Accuracy:", source_val_label_accuracy, "Target Val Label Accuracy:", target_val_label_accuracy)


# We hijack the original loss curves diagram for our own nefarious purposes
plt.rcParams.update({'font.size': 15})
fig, source_train_label_loss_vs_source_val_label_loss = vanilla_tet_jig._do_diagram()

fig.suptitle("Experiment Summary")
fig.set_size_inches(30, 15)


# https://stackoverflow.com/questions/52480756/change-subplot-dimension-of-existing-subplots-in-matplotlib
#
gs = matplotlib.gridspec.GridSpec(2,3)

source_train_label_loss_vs_source_val_label_loss.set_position(gs[5].get_position(fig))


# for i, ax in enumerate(fig.axes):
#     ax.set_position(gs[i].get_position(fig))

ax = fig.add_subplot(gs[1,0])
ax.set_axis_off() 
ax.set_title("Results")
t = ax.table(
    [
        ["Source Val Label Accuracy", "{:.2f}".format(experiment["results"]["source_val_label_accuracy"])],
        ["Source Val Label Loss", "{:.2f}".format(experiment["results"]["source_val_label_loss"])],
        ["Target Val Label Accuracy", "{:.2f}".format(experiment["results"]["target_val_label_accuracy"])],
        ["Target Val Label Loss", "{:.2f}".format(experiment["results"]["target_val_label_loss"])],

        ["Source Test Label Accuracy", "{:.2f}".format(experiment["results"]["source_test_label_accuracy"])],
        ["Source Test Label Loss", "{:.2f}".format(experiment["results"]["source_test_label_loss"])],
        ["Target Test Label Accuracy", "{:.2f}".format(experiment["results"]["target_test_label_accuracy"])],
        ["Target Test Label Loss", "{:.2f}".format(experiment["results"]["target_test_label_loss"])],
        ["Total Epochs Trained", "{:.2f}".format(experiment["results"]["total_epochs_trained"])],
        ["Total Experiment Time", "{:.2f}".format(experiment["results"]["total_experiment_time_secs"])],
    ],
    loc="best",
)
t.auto_set_font_size(False)
t.set_fontsize(20)
t.scale(1.5, 2)

ax = fig.add_subplot(gs[0,0])
ax.set_axis_off() 
ax.set_title("Parameters")

t = ax.table(
    [
        ["Experiment Name", experiment_name],
        ["Learning Rate", lr],
        ["Num Epochs", n_epoch],
        ["Batch Size", batch_size],
        ["patience", patience],
        ["seed", seed],
        ["device", device],
        ["source_snrs", str(source_snrs)],
        ["target_snrs", str(target_snrs)],
    ],
    loc="best"
)
t.auto_set_font_size(False)
t.set_fontsize(20)
t.scale(1.5, 2)

#
# Build a damn pandas dataframe and plot it
# 
ax = fig.add_subplot(gs[0,2])
df = pds.DataFrame.from_dict(accuracies_by_domain)
df = df.sort_values("domain")
df = df.pivot(index="domain", columns="source", values="accuracy")
df.plot(kind="bar", ax=ax)


if not (len(sys.argv) > 1 and sys.argv[1] == "-"):
    plt.savefig(LOSS_CURVE_PATH)
    plt.show()
plt.savefig(LOSS_CURVE_PATH)