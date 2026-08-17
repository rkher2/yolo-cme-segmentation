import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})
from ultralytics import YOLO
from ultralytics.utils import ops as ultralytics_ops

from seg_metrics import process_mask_raw_uncropped

BIN_EDGES = [1.01, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.0]
DENSE_LEVELS = np.arange(0.05, 1.0, 0.05)  # probabilities for contour maps spaced evenly in units of 0.05. e.g. [0.05, 0.10, 0.15, 0.20,....]

def get_raw_probability_maps(model_path, image_paths, imgsz=512, device="0"):
    # run inference on small number of example images in val dataset
    ultralytics_ops.process_mask = process_mask_raw_uncropped

    model = YOLO(model_path)
    prob_maps = {}

    for img_path in image_paths:
        results = model(source=str(img_path), imgsz=imgsz, device=device, verbose=False)
        r = results[0]
        if r.masks is None:
            prob_maps[Path(img_path).stem] = None
            continue

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        # obtain probability maps
        cme_masks = [m for m, c in zip(masks, classes) if int(c) == 1]
        prob_maps[Path(img_path).stem] = cme_masks[0] if cme_masks else None

    return prob_maps


def draw_plain_masks(ax, image_path, gt_mask_path, pred_mask_path, iou_val, title_extra=""):
    """code to create the images that draw predicted vs true masks. Green = GT only, red = predicted only,
    yellow = overlap"""
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_UNCHANGED)
    gt_mask = gt_mask[:, :, 0] if gt_mask.ndim == 3 else gt_mask
    gt_binary = (gt_mask > 0)

    color_layer = np.zeros_like(img_rgb)
    color_layer[gt_binary] = [0, 255, 0]  # GT = solid green

    if pred_mask_path and Path(pred_mask_path).exists():
        pred_mask = cv2.imread(str(pred_mask_path), cv2.IMREAD_UNCHANGED)
        pred_binary = (pred_mask > 0)
        color_layer[pred_binary] = [255, 0, 0]      # pred-only = solid red
        overlap = gt_binary & pred_binary
        color_layer[overlap] = [255, 255, 0]         # overlap = solid yellow


    mask_any = color_layer.any(axis=-1)
    blended = img_rgb.copy()
    blended[mask_any] = cv2.addWeighted(
        img_rgb[mask_any], 0.35, color_layer[mask_any], 0.65, 0
    )

    ax.imshow(blended)
    ax.set_title(f"IoU={iou_val:.4f}{title_extra}\ngreen=GT only | red=pred only | yellow=overlap", fontsize=8)
    ax.axis("off")


def draw_dense_contour_map(ax, image_path, prob_map, iou_val, title_extra=""):
    """code to create probability contour map across the whole image"""
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    ax.imshow(img_rgb)

    if prob_map is None:
        ax.set_title(f"IoU={iou_val:.4f}{title_extra} [NO PREDICTION]", fontsize=8)
        ax.axis("off")
        return

    prob_resized = cv2.resize(prob_map, (w, h))
    cs = ax.contour(prob_resized, levels=DENSE_LEVELS, cmap="turbo", linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%.2f")
    ax.set_title(f"Dense probability contour map (IoU={iou_val:.4f}){title_extra}", fontsize=8)
    ax.axis("off")


def make_bin_overview(valid_df, outdir):
    """make images to display an example of pred vs true masks, for one representative example per IoU bin, 0.5 to 1.0"""
    picked_rows = []

    fig, axes = plt.subplots(len(BIN_EDGES) - 1, 1, figsize=(6, 5 * (len(BIN_EDGES) - 1)))
    if len(BIN_EDGES) - 1 == 1:
        axes = [axes]

    for i in range(len(BIN_EDGES) - 1):
        upper, lower = BIN_EDGES[i], BIN_EDGES[i + 1]
        bin_label = f"< {upper:.2f}" if lower == 0.0 else f"{lower:.2f}-{upper:.2f}"

        in_bin = valid_df[(valid_df["cme_iou"] >= lower) & (valid_df["cme_iou"] < upper)]
        if in_bin.empty:
            axes[i].axis("off")
            axes[i].set_title(f"[{bin_label}] -- no examples in this bin", fontsize=8)
            picked_rows.append(None)
            continue

        bin_center = (upper + lower) / 2 if lower > 0 else upper * 0.95
        closest_idx = (in_bin["cme_iou"] - bin_center).abs().idxmin()
        row = in_bin.loc[closest_idx]
        picked_rows.append((row, bin_label))

        draw_plain_masks(
            axes[i], row["orig_image_path"], row["cme_gt_mask_path"], row["cme_pred_mask_path"],
            row["cme_iou"], title_extra=f"  [bin: {bin_label}]"
        )

    plt.tight_layout()
    out_path = outdir / "iou_bin_overview.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"saved -> {out_path}")

    return picked_rows


def make_dense_contour_map_for_bins(picked_rows, model_path, outdir, imgsz, device):
    """make dense probability contour map images for the same examples used in the IoU bin overview""" 
    valid_rows = [(row, label) for item in picked_rows if item is not None for row, label in [item]]

    if not valid_rows:
        return

    image_paths = [row["orig_image_path"] for row, _ in valid_rows]
    prob_maps = get_raw_probability_maps(model_path, image_paths, imgsz, device)

    fig, axes = plt.subplots(len(valid_rows), 1, figsize=(6, 5 * len(valid_rows)))
    if len(valid_rows) == 1:
        axes = [axes]

    for ax, (row, bin_label) in zip(axes, valid_rows):
        stem = Path(row["orig_image_path"]).stem
        draw_dense_contour_map(ax, row["orig_image_path"], prob_maps.get(stem),
                               row["cme_iou"], title_extra=f"  [bin: {bin_label}]")

    plt.tight_layout()
    out_path = outdir / "iou_bin_overview_contours.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved -> {out_path}")


def make_worst_gallery(df, outdir, n_worst):
    """ make compilation of bad examples where the model fails (half cases where the model fails to make a prediction,
 half cases where the model successfully predicts something but the IoU is extremely low (lowest IoU cases such that IoU > 0)
    """
    n_half = n_worst // 2

    misses = df[df["cme_iou"] == 0]
    detected = df[df["cme_iou"] > 0]

    worst_misses = misses.nsmallest(n_half, "cme_iou")
    worst_detected = detected.nsmallest(n_worst - n_half, "cme_iou")

    combined = pd.concat([worst_misses, worst_detected])
    ncols = 4
    nrows = int(np.ceil(len(combined) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, combined.iterrows()):
        pred_path = row["cme_pred_mask_path"] if row["cme_pred_mask_path"] and not pd.isna(row["cme_pred_mask_path"]) else None
        category = "MISS" if row["cme_iou"] == 0 else "worst-detected"
        draw_plain_masks(ax, row["orig_image_path"], row["cme_gt_mask_path"], pred_path,
                          row["cme_iou"], title_extra=f"  [{category}]")

    for ax in axes[len(combined):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = outdir / f"worst_{n_worst}_examples.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"saved -> {out_path}")

    csv_path = outdir / f"worst_{n_worst}_examples.csv"
    combined[["image", "cme_iou", "cme_num_preds", "orig_image_path", "cme_gt_mask_path", "cme_pred_mask_path"]].to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")


def main(args):
    csv_path = Path(args.run_dir) / "eval_outputs" / "per_image_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- run eval_final.py first")

    df = pd.read_csv(csv_path)
    outdir = Path(args.run_dir) / "eval_outputs" / "iou_examples"
    outdir.mkdir(parents=True, exist_ok=True)

    valid = df[df["cme_pred_mask_path"].notna() & (df["cme_pred_mask_path"] != "")].copy()
   
    picked_rows = make_bin_overview(valid, outdir)
    make_dense_contour_map_for_bins(picked_rows, args.model, outdir, args.imgsz, args.device)

    make_worst_gallery(df, outdir, args.n_worst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model", required=True, help="Path to best.pt, needed only for the dense contour map")
    parser.add_argument("--n_worst", default=20, type=int)
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    main(args)
