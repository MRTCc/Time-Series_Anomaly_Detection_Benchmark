import argparse

from compute_metrics import compute_metrics
from decode_evaluation import decode
from plot_evaluation import plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='evaluate_metrics.py',
        description='Metrics evaluation pipeline of given score of anomaly on a time-series',
        epilog='')

    parser.add_argument("cfg_compute", help=".json configuration file for compute_metrics.py script")
    parser.add_argument("cfg_decode", help=".json configuration file for decode_evaluation.py script")
    parser.add_argument("cfg_plot", help=".json configuration file for plot_evaluation.py script")

    args = parser.parse_args()

    compute_metrics(cfg_file=args.cfg_compute)

    decode(cfg_file=args.cfg_decode)

    plot(cfg_file=args.cfg_plot)
