#! /usr/bin/env python3

import numpy as np

from torch.optim import Adam
import torch

from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.dummy_cida_dataset import Dummy_CIDA_Dataset
from steves_utils.lazy_map import Lazy_Map

from steves_fsl import Steves_Prototypical_Network, split_ds_into_episodes

seed = 420

import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

torch.use_deterministic_algorithms(True) 



def build_Dummy_episodic_iterable(
    num_classes,
    num_examples_per_class,
    n_shot,
    n_query,
    n_train_tasks,
    n_val_tasks,
    n_test_tasks,
    seed,
):

    ds = Dummy_CIDA_Dataset(x_shape=[2,128], domains=[0], num_classes=num_classes, num_unique_examples_per_class=num_examples_per_class)
    ds = Lazy_Map(ds, lam=lambda ex: (torch.from_numpy(ex[0]), ex[1]))

    return split_ds_into_episodes(
        ds=ds,
        n_way=num_classes,
        n_shot=n_shot,
        n_query=n_query,
        n_train_tasks=n_train_tasks,
        n_val_tasks=n_val_tasks,
        n_test_tasks=n_test_tasks,
        seed=seed,
    )


train_dl, val_dl, test_dl = build_Dummy_episodic_iterable(
    num_classes=4,
    num_examples_per_class=75000,
    n_shot=5,
    n_query=10,
    n_train_tasks=2500,
    n_val_tasks=500,
    n_test_tasks=500,
    seed=seed,
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






NUM_EPOCHS = 25

epoch_train_loss = model.fit(train_dl, optimizer, val_loader=val_dl, validation_frequency=500)


# for epoch in range(NUM_EPOCHS):
#     epoch_train_loss = model.fit(train_dl, optimizer, val_loader=val_dl, validation_frequency=500)
#     accuracy = model.evaluate(test_dl)

#     print(epoch_train_loss)

#     print(f"Average Val Accuracy : {(100 * accuracy):.2f}")
#     print(f"Average Loss: {(epoch_train_loss):.2f}")