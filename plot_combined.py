import argparse
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

""" this is the script for plotting loss & 1Q, 3Q, and median IoU simultaneously against epochs """
# matplotlib plot parameters
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# default loss gains used by Ultralytics YOLO26
DEFAULT_GAINS = {
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "seg": 7.5,
}

# obtain the loss gain terms (quantities each component of the loss is multiplied by to calculate total loss) used by the model
def get_gains(run_dir):
    gains = dict(DEFAULT_GAINS)
    args_path = run_dir / "args.yaml"
    if args_path.exists():
        with open(args_path) as f:
            saved = yaml.safe_load(f)
        for key in gains:
            if key in saved:
                gains[key] = saved[key]
    return gains


def weighted_total(df, cols, gains):
    total = None
    for col in cols:
        key = col.split("/")[-1].replace("_loss", "")
        gain = gains.get(key)
        if gain is None:
            continue
        term = df[col] * gain
        total = term if total is None else total + term
    return total

# make plots
def main(args):
    run_dir = Path(args.run_dir).expanduser().resolve()

    loss_df = pd.read_csv(run_dir / "results.csv")
    loss_df.columns = [c.strip() for c in loss_df.columns]

    iou_csv = run_dir / "epoch_metrics.csv"
    if not iou_csv.exists():
        raise FileNotFoundError()
    iou_df = pd.read_csv(iou_csv)

    loss_cols_train = [c for c in loss_df.columns if c.startswith("train/") and "loss" in c]
    loss_cols_val = [c for c in loss_df.columns if c.startswith("val/") and "loss" in c]

    gains = get_gains(run_dir)
    train_total = weighted_total(loss_df, loss_cols_train, gains)
    val_total = weighted_total(loss_df, loss_cols_val, gains)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # only total loss lines, not individual components
    if train_total is not None:
        ax1.plot(loss_df["epoch"], train_total, linestyle="--", color="tab:red", linewidth=2, label="total_loss (train)")
    if val_total is not None:
        ax1.plot(loss_df["epoch"], val_total, linestyle="-", color="tab:red", linewidth=2, label="total_loss (val)")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.fill_between(iou_df["epoch"], iou_df["q1_iou"], iou_df["q3_iou"],
                      color="0.6", alpha=0.5, label="CME IoU IQR (Q1-Q3)")
    ax2.plot(iou_df["epoch"], iou_df["median_iou"], color="black", linewidth=2.5, label="CME median IoU")

    # autoscale IoU axis to actual data range instead of forcing 0-1,
    iou_min = min(iou_df["q1_iou"].min(), iou_df["median_iou"].min())
    iou_max = max(iou_df["q3_iou"].max(), iou_df["median_iou"].max())
    padding = (iou_max - iou_min) * 0.1 if iou_max > iou_min else 0.05
    ax2.set_ylim(max(0, iou_min - padding), min(1, iou_max + padding))

    ax2.set_ylabel("CME IoU")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=8)

    plt.title(f"Total loss and CME IoU over epochs — {run_dir.name}")
    fig.tight_layout()
    out_path = run_dir / "combined_loss_iou.png"
    plt.savefig(out_path, bbox_inches="tight")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    main(args)
