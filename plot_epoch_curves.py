'''
the purpose of this script is to generate per-epoch plots of loss, IoU, and PQ
'''

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# matplotlib plot parameters
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

def main(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    log_csv = run_dir / "epoch_metrics.csv"

    epochs, mean_iou, median_iou, mean_pq = [], [], [], []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            mean_iou.append(float(row["mean_iou"]))
            median_iou.append(float(row["median_iou"]))
            mean_pq.append(float(row["mean_pq"]))
    plt.figure()
    plt.plot(epochs, mean_iou, label="mean IoU")
    plt.plot(epochs, median_iou, label="median IoU")
    plt.xlabel("epoch")
    plt.ylabel("IoU")
    plt.title(f"IoU over epochs — {run_dir.name}")
    plt.legend()
    plt.savefig(run_dir / "iou_over_epochs.png")

    plt.figure()
    plt.plot(epochs, mean_pq, label="mean PQ")
    plt.xlabel("epoch")
    plt.ylabel("PQ")
    plt.title(f"PQ over epochs — {run_dir.name}")
    plt.legend()
    plt.savefig(run_dir / "pq_over_epochs.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="e.g. ~/runs/cme_yolo/yolo26s_50pct")
    args = parser.parse_args()
    args = parser.parse_args()
    main(args)
