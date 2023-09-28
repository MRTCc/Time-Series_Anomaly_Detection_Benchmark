import argparse
import numpy as np


def print_score_line(score, out_file):
    print(f"anomaly rate: {float(score[0]):.5f} | precision: {float(score[1]):.5f}"
          f" | recall: {float(score[2]):.5f} | f1-score: {float(score[3]):.5f}", file=out_file)


def write_txt_data(data, out_file, head_line: str = "F1-Score"):
    """
    assuming data shape is (n, 5), where the columns are:
    - 0 -> anomaly_rate
    - 1 -> precision
    - 2 -> recall
    - 3 -> f1-score
    - 4 -> support

    :param head_line:
    :param data:
    :param out_file:
    :return:
    """
    print(f"<{head_line}>", file=out_file)

    # find best score in data (According to f1-score)
    best_idx = np.argmax(data[:, 3])

    print("!!! Best evaluation:", file=out_file)
    print_score_line(score=data[best_idx], out_file=out_file)
    print("\n\n\n", file=out_file)

    for score in data:
        print_score_line(score=score, out_file=out_file)

    print("\n\n\n", file=out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='decode_evaluation.py',
        description='Decode results of compute_metrics.py from an .npz file to a .txt file (human-readable)',
        epilog='')

    parser.add_argument("in_file", help=".npz file to be decoded")
    parser.add_argument("out_file", help=".txt or .csv where to save decoded results")

    args = parser.parse_args()

    print("Loading data...")
    with np.load(args.in_file) as data_file:
        print("Parsing data...")
        if args.out_file.split(".")[-1] == 'txt':
            with open(args.out_file, 'w') as out_file:
                for key in data_file:
                    if key in 'confusion_matrices_pa':
                        continue
                    write_txt_data(data_file[key], out_file, head_line=key)
        elif args.out_file.split(".")[-1] == 'csv':
            for key in data_file:
                with open(key + "_" + args.out_file, 'w') as out_file:
                    np.savetxt(out_file, data_file[key], delimiter=',', fmt="%.5f")
        else:
            raise ValueError(f"Format not supported for out_file parameter {args.out_file}")
    print("Parsing completed!")
