#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.gridspec
import json
from steves_utils.cida_train_eval_test_jig import CIDA_Train_Eval_Test_Jig
import pandas as pds
import matplotlib.patches as mpatches
import sys


def do_report(experiment_json_path, training_curve_path, results_path, show_only=False):
    with open(experiment_json_path) as f:
        experiment = json.load(f)

    fig, axes = plt.subplots(2, 2)
    plt.tight_layout()

    fig.suptitle("Training Curves")
    fig.set_size_inches(30, 15)

    plt.subplots_adjust(hspace=0.4)
    plt.rcParams['figure.dpi'] = 163

    ###
    # Get Training Curves
    ###
    jig_axes = axes
    CIDA_Train_Eval_Test_Jig.do_diagram(experiment["history"], jig_axes)


    if show_only:
        plt.show()
    else:
        plt.savefig(training_curve_path)

    fig, axes = plt.subplots(2, 2)
    plt.tight_layout()

    fig.suptitle("Parameters and Results")
    fig.set_size_inches(30, 15)

    plt.subplots_adjust(hspace=0.4)
    plt.rcParams['figure.dpi'] = 163

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
            ["Total Experiment Time Secs", "{:.2f}".format(experiment["results"]["total_experiment_time_secs"])],
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
            ["Source Domains", str(experiment["parameters"]["source_domains"])],
            ["Target Domains", str(experiment["parameters"]["target_domains"])],
        ],
        loc="best",
        cellLoc='left',
        colWidths=[0.2,0.55],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)



    #
    # Build a damn pandas dataframe for the per domain accuracies and plot it
    # 

    ax = axes[1][1]
    ax.set_title("Per Domain Accuracy")

    # Convert the dict to a list of tuples
    per_domain_accuracy = experiment["results"]["per_domain_accuracy"]
    per_domain_accuracy = [(domain, v["accuracy"], v["source?"]) for domain,v in per_domain_accuracy.items()]


    df = pds.DataFrame(per_domain_accuracy, columns=["domain", "accuracy", "source?"])
    df.domain = df.domain.astype(float)
    df = df.set_index("domain")
    df = df.sort_values("domain")

    domain_colors = {True: 'r', False: 'b'}
    df['accuracy'].plot(kind='bar', color=[domain_colors[i] for i in df['source?']], ax=ax)

    source_patch = mpatches.Patch(color=domain_colors[True], label='Source Domain')
    target_patch = mpatches.Patch(color=domain_colors[False], label='Target Domain')
    ax.legend(handles=[source_patch, target_patch])

    if show_only:
        plt.show()
    else:
        plt.savefig(results_path)

    sys.exit(0)
    # We hijack the original loss curves diagram for our own nefarious purposes
    plt.rcParams.update({'font.size': 15})
    fig, axis = cida_tet_jig._do_diagram()

    fig.suptitle("Experiment Summary")
    fig.set_size_inches(30, 15)


    # https://stackoverflow.com/questions/52480756/change-subplot-dimension-of-existing-subplots-in-matplotlib
    #
    # The original loss curves use indices [:4]
    alpha_curve, train_label_loss_vs_train_domain_loss, source_val_label_loss_vs_target_val_label_loss, source_train_label_loss_vs_source_val_label_loss = fig.axes

    gs = matplotlib.gridspec.GridSpec(3,3)

    alpha_curve.set_position(gs[1].get_position(fig))
    train_label_loss_vs_train_domain_loss.set_position(gs[2].get_position(fig))
    source_val_label_loss_vs_target_val_label_loss.set_position(gs[4].get_position(fig))
    source_train_label_loss_vs_source_val_label_loss.set_position(gs[5].get_position(fig))


    ax = fig.add_subplot(gs[1,0])
    ax.set_axis_off() 
    ax.set_title("Results")
    t = ax.table(
        [
            ["Source Val Label Accuracy", "{:.2f}".format(experiment["results"]["source_val_label_accuracy"])],
            ["Target Val Label Accuracy", "{:.2f}".format(experiment["results"]["target_val_label_accuracy"])],

            ["Source Test Label Accuracy", "{:.2f}".format(experiment["results"]["source_test_label_accuracy"])],
            ["Target Test Label Accuracy", "{:.2f}".format(experiment["results"]["target_test_label_accuracy"])],

            ["Total Epochs Trained", "{:.2f}".format(experiment["results"]["total_epochs_trained"])],
            ["Total Experiment Time", "{:.2f}".format(experiment["results"]["total_experiment_time_secs"])],

            ["Source Test Label Loss", "{:.2f}".format(experiment["results"]["source_test_label_loss"])],
            ["Target Test Label Loss", "{:.2f}".format(experiment["results"]["target_test_label_loss"])],
            ["Source Test Domain Loss", "{:.2f}".format(experiment["results"]["source_test_domain_loss"])],
            ["Target Test Domain Loss", "{:.2f}".format(experiment["results"]["target_test_domain_loss"])],

            ["Source Val Label Loss", "{:.2f}".format(experiment["results"]["source_val_label_loss"])],
            ["Target Val Label Loss", "{:.2f}".format(experiment["results"]["target_val_label_loss"])],
            ["Source Val Domain Loss", "{:.2f}".format(experiment["results"]["source_val_domain_loss"])],
            ["Target Val Domain Loss", "{:.2f}".format(experiment["results"]["target_val_domain_loss"])],

        ],
        loc="best",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)

    ax = fig.add_subplot(gs[0,0])
    ax.set_axis_off() 
    ax.set_title("Parameters")

    t = ax.table(
        [
            ["Experiment Name", experiment_name],
            ["Learning Rate", lr],
            ["Num Epochs", n_epoch],
            ["Batch Size", batch_size],
            ["patience", patience],
            ["seed", seed],
            ["device", device],
            ["alpha", alpha],
            ["source_domains", str(source_domains)],
            ["target_domains", str(target_domains)],
            # ["x_net", str(x_net)],
            # ["u_net", str(u_net)],
            # ["merge_net", str(merge_net)],
            # ["class_net", str(class_net)],
            # ["domain_net", str(domain_net)],
        ],
        loc="best"
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)


    #
    # Build a damn pandas dataframe and plot it
    #
    ax = fig.add_subplot(gs[2,2])
    df = pds.DataFrame.from_dict(accuracies_by_domain)
    df = df.sort_values("domain")
    df = df.pivot(index="domain", columns="source", values="accuracy")
    df.plot(kind="bar", ax=ax)

    if not (len(sys.argv) > 1 and sys.argv[1] == "-"):
        plt.savefig(LOSS_CURVE_PATH)
        plt.show()
    else:
        plt.savefig(LOSS_CURVE_PATH)


if __name__ == "__main__":
    do_report("results/experiment.json", "/tmp/fug.jpg", "/tmp/fug.jpg", show_only=True)