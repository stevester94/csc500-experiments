#! /usr/bin/env python3
from numpy.lib.utils import source
from steves_utils.ORACLE.ORACLE_sequence import ORACLE_Sequence
import steves_utils.ORACLE.torch as ORACLE_Torch
from steves_utils.lazy_map import Lazy_Map

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

from math import floor

import torch
from torch import nn, optim

from task_sampler import TaskSampler

from steves_utils.torch_sequential_builder import build_sequential

import numpy as np
torch.set_default_dtype(torch.float32)

###############################################
# BEGIN FUCKERY
###############################################

# import math
# import torch

# from steves_utils.ORACLE.ORACLE_sequence import ORACLE_Sequence
# from steves_utils.lazy_map import Lazy_Map

# class ORACLE_Torch_Dataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         desired_serial_numbers,
#         desired_runs,
#         desired_distances,
#         window_length,
#         window_stride,
#         num_examples_per_device,
#         seed,
#         max_cache_size=1e6,
#         transform_func=None,
#         prime_cache=False,
#     ) -> None:
#         super().__init__()

#         self.os = ORACLE_Sequence(
#             desired_serial_numbers,
#             desired_runs,
#             desired_distances,
#             window_length,
#             window_stride,
#             num_examples_per_device,
#             seed,
#             max_cache_size,
#             prime_cache=prime_cache
#         )

#         self.transform_func = transform_func

#     def __len__(self):
#         return len(self.os)
    
#     def __getitem__(self, idx):
#         print("fug")
#         if self.transform_func != None:
#             return self.transform_func(self.os[idx])
#         else:
#             return self.os[idx]

###############################################
# END FUCKERY
###############################################




def build_episodic_iterable(
    desired_serial_numbers,
    desired_distances,
    desired_runs,
    window_length,
    window_stride,
    num_examples_per_device,
    seed,
    max_cache_size,
):
    HACK_NUM_CLASSES = 5

    # ds = ORACLE_Torch.ORACLE_Torch_Dataset(
    #                 desired_serial_numbers=desired_serial_numbers,
    #                 desired_distances=desired_distances,
    #                 desired_runs=desired_runs,
    #                 window_length=window_length,
    #                 window_stride=window_stride,
    #                 num_examples_per_device=num_examples_per_device,
    #                 seed=seed,  
    #                 max_cache_size=max_cache_size,
    #                 # transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]),
    #                 transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), ), # Just (x,y)
    #                 prime_cache=False
    # )

    from steves_utils.dummy_cida_dataset import Dummy_CIDA_Dataset
    ds = Dummy_CIDA_Dataset(x_shape=[2,128], domains=[0], num_classes=HACK_NUM_CLASSES, num_unique_examples_per_class=num_examples_per_device)
    ds = Lazy_Map(ds, lam=lambda ex: ex[:2])

    print(next(iter(ds)))

    train_len = floor(len(ds)*0.7)
    val_len   = floor(len(ds)*0.15)
    test_len  = len(ds) - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

    train_ds.labels = [ex[1] for ex in train_ds]
    val_ds.labels   = [ex[1] for ex in val_ds]
    test_ds.labels  = [ex[1] for ex in test_ds]

    def wrap_in_dataloader(ds):
        sampler = TaskSampler(
                ds,
                n_way=HACK_NUM_CLASSES,
                n_shot=5,
                n_query=10,
                n_tasks=40000
            )

        return torch.utils.data.DataLoader(
            ds,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=50,
            # pin_memory=True,
            batch_sampler=sampler,
            collate_fn=sampler.episodic_collate_fn
        )
    
    return (
        wrap_in_dataloader(train_ds),
        wrap_in_dataloader(val_ds),
        wrap_in_dataloader(test_ds),
    )
    


train_dl, val_dl, test_dl = build_episodic_iterable(
    # desired_serial_numbers=ALL_SERIAL_NUMBERS,
    desired_serial_numbers=[
        "3123D52",
        "3123D65",
        "3123D79",
        "3123D80",
    ],
    # desired_distances=[50,32,8],
    desired_distances=[2],
    desired_runs=[1],
    window_length=256,
    window_stride=50,
    num_examples_per_device=75000,
    seed=1337,
    max_cache_size=200000*len(ALL_SERIAL_NUMBERS)*1000,
)

train_loader = train_dl

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


x_net = [
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0, "groups":2 },},
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

# 0 Loss like right off the bat
x_net = [
        {"class": "Flatten", "kargs": {}},
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 512}},
]

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



x_net           = build_sequential(x_net)

model = PrototypicalNetworks(x_net).cuda()



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()

    return loss.item()


# Train the model yourself with this cell
from tqdm import tqdm

# SM: Rip from easyfsl
from typing import List, Tuple
import numpy as np
def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.
    Returns:
        average of the last window instances in value_list
    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()

log_update_frequency = 10

all_loss = []
model.train()
with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        loss_value = fit(support_images, support_labels, query_images, query_labels)
        all_loss.append(loss_value)

        if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))


print("DONE")