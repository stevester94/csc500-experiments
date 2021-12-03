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

from task_sampler import TaskSampler

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

    ds = ORACLE_Torch.ORACLE_Torch_Dataset(
                    desired_serial_numbers=desired_serial_numbers,
                    desired_distances=desired_distances,
                    desired_runs=desired_runs,
                    window_length=window_length,
                    window_stride=window_stride,
                    num_examples_per_device=num_examples_per_device,
                    seed=seed,  
                    max_cache_size=max_cache_size,
                    # transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]),
                    transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), ), # Just (x,y)
                    prime_cache=False
    )


    train_len = floor(len(ds)*0.7)
    val_len   = floor(len(ds)*0.15)
    test_len  = len(ds) - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

    train_ds.labels = [ex[1] for ex in train_ds]
    val_ds.labels   = [ex[1] for ex in val_ds]
    test_ds.labels  = [ex[1] for ex in test_ds]

    def wrap_in_dataloader(ds):
        return torch.utils.data.DataLoader(
            ds,
            num_workers=1,
            persistent_workers=True,
            prefetch_factor=50,
            # pin_memory=True,
            batch_sampler=TaskSampler(
                ds,
                n_way=len(desired_serial_numbers),
                n_shot=2,
                n_query=1,
                n_tasks=2
            )
        )
    
    return (
        wrap_in_dataloader(train_ds),
        wrap_in_dataloader(val_ds),
        wrap_in_dataloader(test_ds),
    )
    






# def wrap_in_dataloader(ds):
#     return torch.utils.data.DataLoader(
#         ds,
#         batch_size=128,
#         shuffle=True,
#         num_workers=1,
#         persistent_workers=True,
#         prefetch_factor=50,
#         pin_memory=True
#     )
# transform_lambda = lambda ex: ex[:2] # Strip the tuple to just (x,y)

# # CIDA combines source and target training sets into a single dataloader, that's why this one is just called train_dl
# train_dl = wrap_in_dataloader(
#     Lazy_Map(source_train_ds, transform_lambda)
# )

# print(type(source_train_ds))
# print(type(train_dl))

# print(train_dl[20])

train_dl, val_dl, test_dl = build_episodic_iterable(
    desired_serial_numbers=ALL_SERIAL_NUMBERS,
    desired_distances=[50,32,8],
    desired_runs=[1],
    window_length=256,
    window_stride=50,
    num_examples_per_device=1000,
    seed=1337,
    max_cache_size=200000*len(ALL_SERIAL_NUMBERS)*1000,
)

i = iter(train_dl)
print(next(i)[1])
for _ in i: pass
i = iter(train_dl)
print(next(i)[1])

print("DONE")