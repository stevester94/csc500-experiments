#! /usr/bin/env python3

import pandas as pds

from steves_utils.utils_v2 import (
	get_experiments_from_path
)


experiments = get_experiments_from_path("results/trial_1")


"""
We only varied alpha and seed in this trial
"""
experiments = [
    {
        "alpha":  e["parameters"]["alpha"], "seed": e["parameters"]["seed"], 
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
    }
    for e in experiments
]


df = pds.DataFrame(experiments)
grouped = df.groupby("alpha")[["source_val_label_accuracy","target_val_label_accuracy"]].mean()
grouped["count"] = df.groupby("alpha")[["source_val_label_accuracy","target_val_label_accuracy"]].size()
print("Averages")
print(grouped)

