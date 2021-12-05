#! /usr/bin/env python3
import os
import shutil
import subprocess
import json


###########################################
# Helper Functions
###########################################
def ensure_path_dir_exists(path):
    os.system("mkdir -p {}".format(path))

    if not os.path.isdir(path):
        raise Exception("Error creating "+path)

# Gets wrapped in a lambda
def _print_and_log(log_path, s):
    with open(log_path, "a") as f:
        print(s, end="")
        f.write(s)

# Gets wrapped in a lambda
def _debug_print_and_log(log_path, s):
    s = "[CONDUCTOR]: " + s + "\n"
    _print_and_log(log_path, s)

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




class Conductor:
    def __init__(self,
        TRIALS_BASE_PATH:str,
        EXPERIMENT_PATH:str,
        DRIVER_NAME="run.sh",
        LOGS_NAME="logs.txt",
        BEST_MODEL_NAME="results/best_model.pth",
        REPLAY_SCRIPT_NAME="replay.sh",
        REPLAY_PYTHON_PATH="/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.6/dist-packages",
        KEEP_MODEL=False,
        ) -> None:
        ###########################################
        # Globals unlikely to change
        ###########################################
        self.DRIVER_NAME=DRIVER_NAME
        self.LOGS_NAME=LOGS_NAME
        self.BEST_MODEL_NAME=BEST_MODEL_NAME
        self.REPLAY_SCRIPT_NAME=REPLAY_SCRIPT_NAME
        self.REPLAY_PYTHON_PATH=REPLAY_PYTHON_PATH

        ###########################################
        # Organization params (not experiment params)
        ###########################################
        self.EXPERIMENT_PATH=EXPERIMENT_PATH
        self.KEEP_MODEL=KEEP_MODEL

        self.TRIALS_BASE_PATH = TRIALS_BASE_PATH

    
    def prep_experiment(self, trial_dir, driver_name, json):
        with open(os.path.join(trial_dir, self.REPLAY_SCRIPT_NAME), "w") as f:
            f.write("#! /bin/sh\n")
            f.write("export PYTHONPATH={}\n".format(self.REPLAY_PYTHON_PATH))
            f.write("cat << EOF | ./{} -\n".format(driver_name))
            f.write(json)
            f.write("\nEOF")
            f.close()
        
        while not os.path.exists(os.path.join(trial_dir, self.REPLAY_SCRIPT_NAME)):
            self.experiment_debug_print_and_log("Waiting for replay script to be written")
        os.system("chmod +x {}".format(os.path.join(trial_dir, self.REPLAY_SCRIPT_NAME)))

        import inspect
        import steves_utils.dummy_cida_dataset
        import steves_models.configurable_vanilla

        steves_utils_path = os.path.dirname(inspect.getfile(steves_utils.dummy_cida_dataset))
        steves_models_path = os.path.dirname(inspect.getfile(steves_models.configurable_vanilla))

        os.system("rm -rf {}".format(os.path.join(trial_dir, "results")))
        os.mkdir(os.path.join(trial_dir, "results"))

        os.system("cp -R {} {}".format(steves_utils_path, trial_dir))
        os.system("cp -R {} {}".format(steves_models_path, trial_dir))

    def run_experiment(self, trial_dir, replay_script_name):
        from queue import Queue
        from threading import Thread
        
        def enqueue_output(stream, queue):
            while True:
                s = stream.readline()
                if len(s) == 0: # Empty line indicates end of stream
                    break
                queue.put(s)
            stream.close()


        self.experiment_debug_print_and_log(f"Begin experiment at {trial_dir}")
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
                    self.experiment_print_and_log(line)

            try:
                proc.wait(1)
            except:
                pass # Proc still alive
            else:
                self.experiment_debug_print_and_log("Experiment proc ended")
                break # Proc dead

        # Flush the remaining stdout and stderr
        self.experiment_debug_print_and_log("Flush the output buffer")
        while q.qsize() > 0:
            try:  
                line = q.get_nowait() # or q.get(timeout=.1)
            except:
                pass
            else:
                self.experiment_print_and_log(line)
        self.experiment_debug_print_and_log("Done flushing")

        
        if proc.returncode != 0:
            self.experiment_debug_print_and_log("[ERROR] Experiment exited with non-zero code: "+str(proc.returncode))

    def conduct_experiments(self, json_experiment_parameters:list):
        print("[Pre-Flight Conductor] Have a total of {} experiments".format(len(json_experiment_parameters)))

        for idx, j in enumerate(json_experiment_parameters):
            ###########################################
            # Create the trial dir and copy our experiment into it
            ###########################################
            ensure_path_dir_exists(self.TRIALS_BASE_PATH)
            trial_name = get_next_trial_name(self.TRIALS_BASE_PATH)

            if int(trial_name) > idx+1:
                print(f"Trial {idx+1} exists, skipping")
                continue

            trial_dir = os.path.join(self.TRIALS_BASE_PATH, trial_name)

            # shutil will create the dir if it doesn't exist
            shutil.copytree(self.EXPERIMENT_PATH, trial_dir)

            self.experiment_print_and_log = lambda s: _print_and_log(os.path.join(trial_dir, self.LOGS_NAME), s)
            self.experiment_debug_print_and_log = lambda s: _debug_print_and_log(os.path.join(trial_dir, self.LOGS_NAME), s)

            self.prep_experiment(trial_dir, self.DRIVER_NAME, j)

            ###########################################
            # Run the experiment
            ###########################################
            self.run_experiment(trial_dir, self.REPLAY_SCRIPT_NAME)


            ###########################################
            # Perform any cleanup
            ###########################################
            if not self.KEEP_MODEL:
                os.system("rm "+os.path.join(trial_dir, self.BEST_MODEL_NAME))
            os.system("find {} | grep __pycache__ | xargs rm -rf".format(trial_dir))
            os.system("rm "+os.path.join(trial_dir, ".gitignore"))
            os.system("mv "+os.path.join(trial_dir, "logs.txt") + " " + os.path.join(trial_dir, "results"))