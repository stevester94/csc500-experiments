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
REPLAY_SCRIPT_NAME="replay.sh"
REPLAY_PYTHON_PATH="/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.6/dist-packages"

###########################################
# Organization params (not experiment params)
###########################################
TRIALS_DIR=os.path.join(PAST_RUNS_DIR, "chapter3/reproduce_oshea_snr_/cnn_1")
EXPERIMENT_PATH="./cnn_experiment"
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

def prep_experiment(trial_dir, driver_name, json):
    with open(os.path.join(trial_dir, REPLAY_SCRIPT_NAME), "w") as f:
        f.write("#! /bin/sh\n")
        f.write("export PYTHONPATH={}\n".format(REPLAY_PYTHON_PATH))
        f.write("cat << EOF | ./{} -\n".format(driver_name))
        f.write(json)
        f.write("\nEOF")
        f.close()
    
    while not os.path.exists(os.path.join(trial_dir, REPLAY_SCRIPT_NAME)):
        debug_print_and_log("Waiting for replay script to be written")
    os.system("chmod +x {}".format(os.path.join(trial_dir, REPLAY_SCRIPT_NAME)))

    import inspect
    import steves_utils.dummy_cida_dataset
    import steves_models.configurable_vanilla

    steves_utils_path = os.path.dirname(inspect.getfile(steves_utils.dummy_cida_dataset))
    steves_models_path = os.path.dirname(inspect.getfile(steves_models.configurable_vanilla))

    os.system("rm -rf {}".format(os.path.join(trial_dir, "results")))
    os.mkdir(os.path.join(trial_dir, "results"))

    os.system("cp -R {} {}".format(steves_utils_path, trial_dir))
    os.system("cp -R {} {}".format(steves_models_path, trial_dir))



def run_experiment(trial_dir, replay_script_name):
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
    proc = subprocess.Popen([os.path.join(trial_dir, replay_script_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=trial_dir, text=True)
    q = Queue()
    stdout_thread = Thread(target=enqueue_output, args=(proc.stdout, q))
    stderr_thread = Thread(target=enqueue_output, args=(proc.stderr, q))

    stdout_thread.daemon = True
    stdout_thread.start()
    stderr_thread.daemon = True
    stderr_thread.start()

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
base_parameters["experiment_name"] = "OShea SNR CNN"
base_parameters["lr"] = 0.001
base_parameters["n_epoch"] = 200
base_parameters["batch_size"] = 128
base_parameters["patience"] = 10
base_parameters["seed"] = 1337
base_parameters["device"] = "cuda"

base_parameters["source_snrs"] = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
base_parameters["target_snrs"] = [0]

base_parameters["x_net"] = [
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
    {"class": "Linear", "kargs": {"in_features": 80, "out_features": 16}},
]

seeds = [1337, 82, 1234, 9393, 1984, 2017, 1445, 511, 
    16044, 16432, 1792, 4323, 6801, 13309, 3517, 12140,
    5961, 19872, 7250, 16276, 16267, 17534, 6114, 16017
]
seeds = [1337]

custom_parameters = [
    {"device": "cuda"} # Quick little hack so we have one experiment
]

import copy
for s in seeds:
    for p in custom_parameters:
        parameters = copy.deepcopy(base_parameters)
        for key,val in p.items():
            parameters[key] = val
        parameters["seed"] = s

        j = json.dumps(parameters, indent=2)
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

    prep_experiment(trial_dir, DRIVER_NAME, j)

    ###########################################
    # Run the experiment
    ###########################################
    run_experiment(trial_dir, REPLAY_SCRIPT_NAME)


    ###########################################
    # Perform any cleanup
    ###########################################
    if not KEEP_MODEL:
        os.system("rm "+os.path.join(trial_dir, BEST_MODEL_NAME))
    os.system("find {} | grep __pycache__ | xargs rm -rf".format(trial_dir))
    os.system("rm "+os.path.join(trial_dir, ".gitignore"))
    os.system("mv "+os.path.join(trial_dir, "logs.txt") + " " + os.path.join(trial_dir, "results"))