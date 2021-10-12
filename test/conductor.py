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
TRIALS_DIR=os.path.join(PAST_RUNS_DIR, "chapter3/conductor_testing")
EXPERIMENT_PATH="./configurable_cida"
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
for n_epoch in [3,5]:
    parameters = {}
    parameters["experiment_name"] = "Manual Experiment"
    parameters["lr"] = 0.0001
    parameters["n_epoch"] = n_epoch
    parameters["batch_size"] = 128
    parameters["patience"] = 10
    parameters["seed"] = 1337
    parameters["device"] = "cuda"

    parameters[""] = "cuda"

    parameters["x_net"] = [
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":2, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},
    ]
    parameters["u_net"] = [
        {"class": "Identity", "kargs": {}},
    ]
    parameters["merge_net"] = [
        {"class": "Linear", "kargs": {"in_features": 50*58+1, "out_features": 256}},
    ]
    parameters["class_net"] = [
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 80}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
    ]
    parameters["domain_net"] = [
        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 100}},
        {"class": "BatchNorm1d", "kargs": {"num_features": 100}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Linear", "kargs": {"in_features": 100, "out_features": 1}},
    ]

    parameters["device"] = "cuda"

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








