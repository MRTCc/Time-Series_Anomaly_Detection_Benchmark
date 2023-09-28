import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix

from compute_metrics import get_y_pred
from compute_metrics import get_y_pred_adjusted


def build_outfile_path(file_path, custom_name):
    file_extension = file_path.split('.')[-1]
    outfile_path = f"{file_path.replace('.' + file_extension, '')}_{custom_name}.{file_extension}"

    return outfile_path


def plot_sequences(seq_dict, file_path, plot_name, annotation=""):
    n_subplots = len(seq_dict)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(8, 3 * n_subplots))

    for i, (title, seq) in enumerate(seq_dict.items()):
        ax = axes[i]
        ax.plot(seq)
        ax.set_title(f"{title}")
        ax.set_xlabel("Time")
        # TODO: customizzare i nomi per gli assi y (feature di bassa priorità)
        ax.set_ylabel("Value")

    plt.tight_layout()

    if annotation != "":
        plt.annotate(annotation, xy=(3, 9), xytext=(2, 15),
                     arrowprops=dict(arrowstyle='->', linewidth=1.5),
                     fontsize=12, color='red')

    outfile_path = build_outfile_path(file_path=file_path, custom_name=plot_name)
    plt.savefig(outfile_path)

    plt.show()


def plot_data(data, channel_list, file_path):
    n_channels = len(channel_list)

    fig, axes = plt.subplots(n_channels, 1, figsize=(8, 3 * n_channels))
    for i in channel_list:
        ax = axes[i]
        ax.plot(data[:, i])
        ax.set_title(f"Channel {i + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

    plt.tight_layout()

    outfile_path = build_outfile_path(file_path=file_path, custom_name="data_channels")
    plt.savefig(outfile_path)

    plt.show()


def plot_labels_estimations(labels, estimations, outfile_path):
    seq_dict = {
        "labels": labels,
        "estimations": estimations
    }
    plot_sequences(seq_dict=seq_dict, file_path=outfile_path, plot_name="labels_vs_estimations")


def plot_labels_estimations_predictions(labels, estimations, predictions, file_path, anomaly_rate):
    seq_dict = {
        "labels": labels,
        "estimations": estimations,
        "predictions": predictions
    }
    plot_sequences(seq_dict=seq_dict, file_path=file_path, plot_name="preds", annotation=str(anomaly_rate))


def plot_xy(ax, ay, label, xlabel, ylabel, plot_title, file_path):
    plt.plot(ax, ay, label=label, marker='o', linestyle='-')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()

    outfile_path = build_outfile_path(file_path=file_path, custom_name=plot_title)
    plt.savefig(outfile_path)

    plt.show()


def plot_confusion_matrix(conf_matr, file_path, plot_title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matr, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(plot_title)

    outfile_path = build_outfile_path(file_path=file_path, custom_name=plot_title)
    plt.savefig(outfile_path)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='plot_evaluation.py',
        description='Plot metric results and data (times-series)',
        epilog='')
    parser.add_argument("config_file")
    args = parser.parse_args()

    print("[INFO] Loading configuration file...")
    with open(args.config_file, 'r') as cfg_file:
        cfg = json.load(cfg_file)

        print("[INFO] Loading data...")
        data = np.load(cfg['data_path'])
        labels = np.load(cfg['label_path']).copy().astype(np.int32)
        outfile_path = cfg["outfile_path"]

        print("[INFO] Plotting...")
        if cfg["plot_data"]:
            channels = cfg["channels_to_plot"]
            plot_data(data, channel_list=channels, file_path=outfile_path)

        if cfg["plot_labels_and_estimations"]:
            estimations = np.load(cfg["estimation_path"])
            plot_labels_estimations(labels, estimations, outfile_path)

        if cfg["plot_labels_estimations_predictions"]:
            # TODO: aggiungere la feature di predictions con point adjustment (priorità media)
            estimations = np.load(cfg["estimation_path"])
            predictions = get_y_pred(estimations=estimations, anomaly_rate=cfg["anomaly_rate"])
            plot_labels_estimations_predictions(labels, estimations, predictions, outfile_path, cfg["anomaly_rate"])

        if cfg["plot_f1_anomaly_rate"]:
            data = np.load(cfg["metric_results_path"])["f1_score"]
            plot_xy(ax=data[:, 0], ay=data[:, 3], label="anomaly_rate-f1_score", xlabel="anomaly_rate",
                    ylabel="f1_score", plot_title="anomaly_rate-f1_score", file_path=outfile_path)

        if cfg["plot_f1_pa_anomaly_rate"]:
            data = np.load(cfg["metric_results_path"])["f1_pa_score"]
            plot_xy(ax=data[:, 0], ay=data[:, 3], label="anomaly_rate-f1_pa_score", xlabel="anomaly_rate",
                    ylabel="f1_pa_score", plot_title="anomaly_rate-f1_pa_score", file_path=outfile_path)

        if cfg["plot_precision_anomaly_rate"]:
            # TODO: nota che, per ora, precision e recall si riferiscono alla f1-score (NON alla f1-pa-score)
            data = np.load(cfg["metric_results_path"])["f1_score"]
            plot_xy(ax=data[:, 0], ay=data[:, 1], label="anomaly_rate-precision", xlabel="anomaly_rate",
                    ylabel="precision", plot_title="anomaly_rate-precision", file_path=outfile_path)

        if cfg["plot_recall_anomaly_rate"]:
            data = np.load(cfg["metric_results_path"])["f1_score"]
            plot_xy(ax=data[:, 0], ay=data[:, 1], label="anomaly_rate-recall", xlabel="anomaly_rate",
                    ylabel="recall", plot_title="anomaly_rate-recall", file_path=outfile_path)

        if cfg["plot_confusion_matrix"]:
            estimations = np.load(cfg["estimation_path"])
            predictions = get_y_pred(estimations=estimations, anomaly_rate=cfg["anomaly_rate"])
            conf_matr = confusion_matrix(labels, predictions)
            plot_confusion_matrix(conf_matr, outfile_path, "confusion_matrix")

print("[INFO] Done!")
