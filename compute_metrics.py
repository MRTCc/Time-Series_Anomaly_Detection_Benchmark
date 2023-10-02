import json
import numpy as np
import sklearn.metrics as sk_metr
import argparse


def ewma(series, weighting_factor=0.9):
    """
    Exponential weighted moving average
    :param series:
    :param weighting_factor:
    :return:
    """
    current_factor = 1 - weighting_factor
    _ewma = series.copy()
    for i in range(1, len(_ewma)):
        _ewma[i] = _ewma[i - 1] * weighting_factor + _ewma[i] * current_factor
    return _ewma


def get_y_pred(estimations, anomaly_rate):
    anomalies_point = estimations.copy()
    q = np.quantile(anomalies_point, 1 - anomaly_rate)
    anomalies_point = (anomalies_point > q).astype(np.int32)

    return anomalies_point


def get_y_pred_adjusted(test_labels, estimations, anomaly_rate):
    anomalies_point = estimations.copy()
    q = np.quantile(anomalies_point, 1 - anomaly_rate)
    anomalies_point = (anomalies_point > q).astype(np.int32)

    padded_test_data = np.concatenate([np.zeros(1), test_labels, np.zeros(1)]).astype(np.int32)
    diff_signal = padded_test_data[1:] - padded_test_data[:-1]

    starting = np.where(diff_signal == 1)[0]
    ending = np.where(diff_signal == -1)[0]
    intervals = np.stack([starting, ending], axis=1)

    for start, end in intervals:
        interval = slice(start, end)
        if anomalies_point[interval].sum() > 0:
            anomalies_point[interval] = 1

    return anomalies_point


def compute_f1_score(test_labels, estimations, divisions, min_anomaly_rate, max_anomaly_rate,
                     step_anomaly_rate, adjustment: bool = False):
    for i, (idx_start, idx_end) in enumerate(divisions):
        y_true = test_labels[idx_start:idx_end]
        _estimations = estimations[idx_start:idx_end]

        if len(divisions) < 2:
            # evaluation format: anomaly_rate | precision | recall | f1-score | support
            rates = np.arange(min_anomaly_rate, max_anomaly_rate + step_anomaly_rate, step_anomaly_rate)
            evaluation = np.empty((rates.shape[0], 5), dtype=np.float32)
            confusion_matrices = np.empty((rates.shape[0], 5), dtype=np.float32)

            for idx in range(rates.shape[0]):
                if adjustment:
                    y_pred = get_y_pred_adjusted(test_labels, _estimations, rates[idx])
                else:
                    y_pred = get_y_pred(_estimations, rates[idx])
                evaluation[idx, 0] = rates[idx]
                scores = sk_metr.precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                 average='binary', pos_label=1,
                                                                 labels=1)
                scores = np.array(scores, dtype=np.float32)
                evaluation[idx, 1:] = scores

                confusion_matrix = sk_metr.confusion_matrix(y_true=y_true, y_pred=y_pred).flatten()
                confusion_matrices[idx, 0] = rates[idx]
                confusion_matrices[idx, 1:] = confusion_matrix

            return evaluation, confusion_matrices
        else:
            # TODO: see lines 190,...,205 -> mi sembra un po' insensata... (ma può essere che non abbia capito bene)
            raise SystemError("Not yet implemented!")


def compute_metrics(cfg_file):
    print("Loading configuration data...")
    with open(cfg_file, 'r') as cfg_file:
        cfg = json.load(cfg_file)

        test_data = np.load(cfg['test_data_path'])
        test_labels = np.load(cfg['test_label_path']).copy().astype(np.int32)

        if cfg['divisions_path'] == "":
            divisions = [[0, len(test_data)]]
        else:
            with open(cfg['divisions_path'], 'r') as divisions_file:
                divisions = json.load(divisions_file)

        estimations = np.load(cfg["estimation_path"])

        print("Computing f1_score...")
        evaluation_f1, confusion_matrices = compute_f1_score(test_labels, estimations, divisions,
                                                             cfg["min_anomaly_rate"],
                                                             cfg["max_anomaly_rate"], cfg["step_anomaly_rate"])

        evaluation_f1_pa = None
        if cfg["compute_f1_pa"] is True:
            print("Computing f1_pa_score...")
            evaluation_f1_pa, confusion_matrices_pa = compute_f1_score(test_labels, estimations, divisions,
                                                                       cfg["min_anomaly_rate"], cfg["max_anomaly_rate"],
                                                                       cfg["step_anomaly_rate"], adjustment=True)

        if evaluation_f1_pa is not None:
            np.savez(cfg["outfile_path"], f1_score=evaluation_f1, f1_pa_score=evaluation_f1_pa,
                     confusion_matrices=confusion_matrices, confusion_matrices_pa=confusion_matrices_pa)
        else:
            np.savez(cfg["outfile_path"], f1_score=evaluation_f1, confusion_matrices=confusion_matrices)
    print("Computation done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='compute_metrics.py',
        description='Compute metrics given the estimations and the ground truth labels',
        epilog='')
    parser.add_argument("config_file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as cfg_file:
        cfg = json.load(cfg_file)

        test_data = np.load(cfg['test_data_path'])
        test_labels = np.load(cfg['test_label_path']).copy().astype(np.int32)

        if cfg['divisions_path'] == "":
            divisions = [[0, len(test_data)]]
        else:
            with open(cfg['divisions_path'], 'r') as divisions_file:
                divisions = json.load(divisions_file)

        estimations = np.load(cfg["estimation_path"])

        print("Computing f1_score...")
        evaluation_f1, confusion_matrices = compute_f1_score(test_labels, estimations, divisions,
                                                             cfg["min_anomaly_rate"],
                                                             cfg["max_anomaly_rate"], cfg["step_anomaly_rate"])

        evaluation_f1_pa = None
        if cfg["compute_f1_pa"] is True:
            print("Computing f1_pa_score...")
            evaluation_f1_pa, confusion_matrices_pa = compute_f1_score(test_labels, estimations, divisions,
                                                                       cfg["min_anomaly_rate"], cfg["max_anomaly_rate"],
                                                                       cfg["step_anomaly_rate"], adjustment=True)

        if evaluation_f1_pa is not None:
            np.savez(cfg["outfile_path"], f1_score=evaluation_f1, f1_pa_score=evaluation_f1_pa,
                     confusion_matrices=confusion_matrices, confusion_matrices_pa=confusion_matrices_pa)
        else:
            np.savez(cfg["outfile_path"], f1_score=evaluation_f1, confusion_matrices=confusion_matrices)
