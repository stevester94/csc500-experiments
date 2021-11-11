#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.gridspec
import json
from steves_utils.vanilla_train_eval_test_jig import Vanilla_Train_Eval_Test_Jig
import pandas as pds

def do_report(experiment_json_path, results_dir):

    with open(experiment_json_path) as f:
        experiment = json.load(f)

    fig, axes = plt.subplots(2, 2)
    plt.tight_layout()

    fig.suptitle("Experiment Summary")
    fig.set_size_inches(30, 15)

    plt.subplots_adjust(hspace=0.4)
    plt.rcParams['figure.dpi'] = 163

    ###
    # Get Loss Curve
    ###
    Vanilla_Train_Eval_Test_Jig.do_diagram(experiment["history"], axes[0][0])

    ###
    # Get Results Table
    ###
    ax = axes[0][1]
    ax.set_axis_off() 
    ax.set_title("Results")
    t = ax.table(
        [
            ["Source Val Label Accuracy", "{:.2f}".format(experiment["results"]["source_val_label_accuracy"])],
            ["Source Val Label Loss", "{:.2f}".format(experiment["results"]["source_val_label_loss"])],
            ["Target Val Label Accuracy", "{:.2f}".format(experiment["results"]["target_val_label_accuracy"])],
            ["Target Val Label Loss", "{:.2f}".format(experiment["results"]["target_val_label_loss"])],

            ["Source Test Label Accuracy", "{:.2f}".format(experiment["results"]["source_test_label_accuracy"])],
            ["Source Test Label Loss", "{:.2f}".format(experiment["results"]["source_test_label_loss"])],
            ["Target Test Label Accuracy", "{:.2f}".format(experiment["results"]["target_test_label_accuracy"])],
            ["Target Test Label Loss", "{:.2f}".format(experiment["results"]["target_test_label_loss"])],
            ["Total Epochs Trained", "{:.2f}".format(experiment["results"]["total_epochs_trained"])],
            ["Total Experiment Time", "{:.2f}".format(experiment["results"]["total_experiment_time_secs"])],
        ],
        loc="best",
        cellLoc='left',
        colWidths=[0.3,0.4],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)


    ###
    # Get Parameters Table
    ###
    ax = axes[1][0]
    ax.set_axis_off() 
    ax.set_title("Parameters")

    t = ax.table(
        [
            ["Experiment Name", experiment["parameters"]["experiment_name"]],
            ["Learning Rate", experiment["parameters"]["lr"]],
            ["Num Epochs", experiment["parameters"]["n_epoch"]],
            ["Batch Size", experiment["parameters"]["batch_size"]],
            ["patience", experiment["parameters"]["patience"]],
            ["seed", experiment["parameters"]["seed"]],
            ["device", experiment["parameters"]["device"]],
            ["Source Distances", str(experiment["parameters"]["source_distances"])],
            ["Target Distances", str(experiment["parameters"]["target_distances"])],
        ],
        loc="best",
        cellLoc='left',
        colWidths=[0.2,0.55],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)



    #
    # Build a damn pandas dataframe and plot it
    # 


    # print(experiment["results"]["per_domain_accuracy"])

    # ax = axes[1][1]
    # df = pds.DataFrame(experiment["results"]["per_domain_accuracy"], index=[0])
    # df = df.sort_values("domain")
    # df = df.pivot(index="domain", columns="source", values="accuracy")
    # df.plot(kind="bar", ax=ax)

    plt.show()


    # if not (len(sys.argv) > 1 and sys.argv[1] == "-"):
    #     plt.savefig(LOSS_CURVE_PATH)
    #     plt.show()
    # plt.savefig(LOSS_CURVE_PATH)


if __name__ == "__main__":
    import sys
    do_report(sys.argv[1], "/tmp")