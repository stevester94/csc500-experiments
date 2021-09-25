#! /usr/bin/env python3
from cida_images_cnn import CIDA_Images_CNN_Model
from steves_utils.cida_train_eval_test_jig import  CIDA_Train_Eval_Test_Jig
from steves_utils.cida_mnist_dataset import CIDA_MNIST_DS
import torch
import numpy as np
import os
import sys
import json
import time

# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
NUM_CLASSES = -1337
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
    fake_args["n_epoch"] = 3
    fake_args["batch_size"] = 128
    fake_args["patience"] = 10
    fake_args["seed"] = 1337
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
# Build the dataset
###################################
source_ds = CIDA_MNIST_DS(
    seed, 
    1, 
    0,
    0,
    1000
)

target_ds = CIDA_MNIST_DS(
    seed, 
    1, 
    0,
    0,
    1000
)

source_dl = torch.utils.data.DataLoader(
    source_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)

target_dl = torch.utils.data.DataLoader(
    target_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)

alpha_func = lambda epoch, total_epochs: epoch/total_epochs

###################################
# Build the model
###################################
model = CIDA_Images_CNN_Model(NUM_CLASSES)


###################################
# Build the tet jig, train
###################################
cida_tet_jig = CIDA_Train_Eval_Test_Jig(
    model=model,
    label_loss_object=torch.nn.NLLLoss(),
    domain_loss_object=torch.nn.L1Loss(),
    path_to_best_model=BEST_MODEL_PATH,
    device=torch.device(device)
)

cida_tet_jig.train(
    source_train_iterable=source_dl,
    source_val_iterable=source_dl,
    target_train_iterable=target_dl,
    target_val_iterable=target_dl,
    patience=patience,
    learning_rate=lr,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
    alpha_func=alpha_func
)


###################################
# Colate experiment results
###################################
source_test_label_accuracy, source_test_label_loss, source_test_domain_loss = cida_tet_jig.test(source_dl)
target_test_label_accuracy, target_test_label_loss, target_test_domain_loss = cida_tet_jig.test(target_dl)

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
        "total_epochs_trained": total_epochs_trained,
        "total_experiment_time_secs": total_experiment_time_secs,
    },
    "history": history,
}

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

# cida_tet_jig.save_loss_diagram(LOSS_CURVE_PATH)