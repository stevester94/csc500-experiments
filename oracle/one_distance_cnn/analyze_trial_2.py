#! /usr/bin/env python3

import pandas as pds

from steves_utils.utils_v2 import (
	get_experiments_from_path
)


group_2_x = [
    {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0, "groups":2 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 5800, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
]

group_1_x = [
    {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0},},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 5800, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},

    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
]

experiments = get_experiments_from_path("results/each_distance_each_run_stride_1/trial_2/")

"""
We only varied alpha and seed in this trial
"""
experiments = [
    {
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
        "seed": e["parameters"]["seed"],
        "x_net": "group_1_x" if e["parameters"]["x_net"] == group_1_x else "group_2_x",
        "desired_runs": e["parameters"]["desired_runs"],
        "window_stride": e["parameters"]["window_stride"],
        "source_domains": e["parameters"]["source_domains"],
        "target_domains": e["parameters"]["target_domains"],
        "num_examples_per_device": e["parameters"]["num_examples_per_device"],
    }
    for e in experiments
]


df = pds.DataFrame(experiments)
print(df["x_net"].unique())

pds.set_option("display.max_rows", None, "display.max_columns", None, "display.width", 160)
print(df.sort_values("source_val_label_accuracy", ascending=False))

# grouped = df.groupby("alpha")[["source_val_label_accuracy","target_val_label_accuracy"]].mean()
# grouped["count"] = df.groupby("alpha")[["source_val_label_accuracy","target_val_label_accuracy"]].size()
# print("Averages")
# print(grouped)