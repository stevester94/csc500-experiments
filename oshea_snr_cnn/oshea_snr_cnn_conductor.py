#! /usr/bin/env python3
import os
import shutil
import subprocess
import json

###########################################
# Globals unlikely to change
###########################################
PAST_RUNS_DIR="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-past-runs/"
DRIVER_NAME="run.sh"
LOGS_NAME="logs.txt"
BEST_MODEL_NAME="results/best_model.pth"

###########################################
# Organization params (not experiment params)
###########################################
TRIALS_DIR=os.path.join(PAST_RUNS_DIR, "chapter3/cnn/oshea_snr_2")
EXPERIMENT_PATH="./oshea_snr"
KEEP_MODEL=False



###########################################
# Helper Functions
###########################################
def ensure_path_dir_exists(path):
    os.system("mkdir -p {}".format(path))

    if not os.path.isdir(path):
        raise Exception("Error creating "+path)


def get_next_trial_name(trials_path):
    ints = []

    for d in os.listdir(trials_path):
        try:
            ints.append(int(d))
        except:
            pass
    
    ints.sort()

    if len(ints) == 0:
        return str(1)
    else:
        return str(ints[-1]+1)

# Gets overridden 
def _print_and_log(log_path, s):
    with open(log_path, "a") as f:
        print(s, end="")
        f.write(s)

def _debug_print_and_log(log_path, s):
    s = "[CONDUCTOR]: " + s + "\n"
    _print_and_log(log_path, s)
    
def run_experiment(trial_dir, driver_name, logs_name, json):
    import time
    from queue import Queue
    from threading import Thread
    
    def enqueue_output(stream, queue):
        while True:
            s = stream.readline()
            if len(s) == 0: # Empty line indicates end of stream
                break
            queue.put(s)
        stream.close()
    debug_print_and_log("Begin experiment")
    proc = subprocess.Popen([os.path.join(trial_dir, driver_name), "-"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=trial_dir, text=True)
    q = Queue()
    stdout_thread = Thread(target=enqueue_output, args=(proc.stdout, q))
    stderr_thread = Thread(target=enqueue_output, args=(proc.stderr, q))

    stdout_thread.daemon = True
    stdout_thread.start()
    stderr_thread.daemon = True
    stderr_thread.start()

    proc.stdin.write(json)
    proc.stdin.close()

    while True:
        while q.qsize() > 0:
            try:  
                line = q.get_nowait() # or q.get(timeout=.1)
            except:
                pass
            else:
                print_and_log(line)

        try:
            proc.wait(1)
        except:
            pass # Proc still alive
        else:
            debug_print_and_log("Experiment proc ended")
            break # Proc dead

    # Flush the remaining stdout and stderr
    debug_print_and_log("Flush the output buffer")
    while q.qsize() > 0:
        try:  
            line = q.get_nowait() # or q.get(timeout=.1)
        except:
            pass
        else:
            print_and_log(line)
    debug_print_and_log("Done flushing")

    
    if proc.returncode != 0:
        debug_print_and_log("[ERROR] Experiment exited with non-zero code: "+str(proc.returncode))



###########################################
# Form the experiment parameters
###########################################
experiment_jsons = []

base_parameters = {}
base_parameters["experiment_name"] = "Manual Experiment"
base_parameters["lr"] = 0.001
base_parameters["n_epoch"] = 100
base_parameters["batch_size"] = 1024
base_parameters["patience"] = 10
base_parameters["seed"] = 1337
base_parameters["device"] = "cuda"

base_parameters["source_snrs"] = [0, 2, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
base_parameters["target_snrs"] = [4]
base_parameters["alpha"] = "sigmoid"


base_parameters["x_net"] = [
    {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Flatten", "kargs": {}},
]
base_parameters["u_net"] = [
    {"class": "Identity", "kargs": {}},
]
base_parameters["merge_net"] = [
    {"class": "Linear", "kargs": {"in_features": 50*58+1, "out_features": 256}},
]
base_parameters["class_net"] = [
    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Dropout", "kargs": {"p": 0.5}},
    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
]
base_parameters["domain_net"] = [
    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
    {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
    {"class": "nnClamp", "kargs": {"min": -20, "max": 20}},
]

base_parameters["device"] = "cuda"


# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
custom_parameters = [
    {"source_snrs":[0, 2, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2], "target_snrs":[4], "alpha":"sigmoid"},
    {"source_snrs":[0, 2, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2], "target_snrs":[4], "alpha":"null"},
    # {"source_snrs":[-6, -4, -2], "target_snrs":[14, 16, 18], "alpha":"sigmoid"},
    # {"source_snrs":[14, 16, 18], "target_snrs":[-6, -4, -2], "alpha":"null"},
    # {"source_snrs":[-6, -4, -2], "target_snrs":[14, 16, 18], "alpha":"null"},
]

seeds = [1337, 82, 1234, 9393, 1984]
seeds = [1337]

import copy
for s in seeds:
    for p in custom_parameters:
        parameters = copy.deepcopy(base_parameters)
        for key,val in p.items():
            parameters[key] = val
        parameters["seed"] = s

        j = json.dumps(parameters)
        experiment_jsons.append(j)

print("[Pre-Flight Conductor] Have a total of {} experiments".format(len(experiment_jsons)))


for j in experiment_jsons:
    ###########################################
    # Create the trial dir and copy our experiment into it
    ###########################################
    ensure_path_dir_exists(TRIALS_DIR)
    trial_name = get_next_trial_name(TRIALS_DIR)
    trial_dir = os.path.join(TRIALS_DIR, trial_name)

    # shutil will create the dir if it doesn't exist
    shutil.copytree(EXPERIMENT_PATH, trial_dir)

    print_and_log = lambda s: _print_and_log(os.path.join(trial_dir, LOGS_NAME), s)
    debug_print_and_log = lambda s: _debug_print_and_log(os.path.join(trial_dir, LOGS_NAME), s)

    ###########################################
    # Run the experiment
    ###########################################
    run_experiment(trial_dir, DRIVER_NAME, LOGS_NAME, j)


    ###########################################
    # Perform any cleanup
    ###########################################
    if not KEEP_MODEL:
        os.system("rm "+os.path.join(trial_dir, BEST_MODEL_NAME))
    os.system("rm -rf "+os.path.join(trial_dir, "__pycache__"))
    os.system("rm "+os.path.join(trial_dir, ".gitignore"))
    os.system("mv "+os.path.join(trial_dir, "logs.txt") + " " + os.path.join(trial_dir, "results"))









