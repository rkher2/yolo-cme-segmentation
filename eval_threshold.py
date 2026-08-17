import argparse
from pathlib import Path

import numpy as np
import cv2
import yaml
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import ops as ultralytics_ops

from seg_metrics import get_gt_mask, iou

# this file handles evaluation of the pixel probability threshold for creating masks and creation of the graph of IoU vs pixel probability

CLASS_MASK_SUFFIX = {0: "_mask_0.png", 1: "_mask_2.png"}
CLASS_NAMES = {0: "occulter", 1: "cme"}

THRESHOLDS = np.arange(0.05, 1.00, 0.05)

_original_process_mask = ultralytics_ops.process_mask


def process_mask_raw(protos, masks_in, bboxes, shape, upsample: bool = False):
    """
    Reimplementation of ultralytics.utils.ops.process_mask, without the final
    `.gt_(0.0).byte()` binarization step. returns continuous sigmoid
    probabilities [0, 1] instead, thus reconstructing the probabilities that YOLO predicts
    """
    c, mh, mw = protos.shape
    if masks_in.shape[0] == 0:
        return torch.zeros((0, *(shape if upsample else (mh, mw))), dtype=torch.float32, device=masks_in.device)

    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # raw logits

    if upsample:
        masks = torch.nn.functional.interpolate(masks[None], shape, mode="bilinear")[0]

    masks = torch.sigmoid(masks)  # sigmoid squeezes logits into output probabilities
    return ultralytics_ops.crop_mask(masks, bboxes)


def find_original_folder(image_stem, source_root, cache={}):
    if not cache:
        for folder in source_root.iterdir():
            if folder.is_dir():
                for img in folder.glob("*_btot.png"):
                    cache[img.stem] = folder
    return cache.get(image_stem)


def verify_raw_probabilities(model, val_dir, imgsz, device):
    """sanity check"""
    results = model(source=str(val_dir), imgsz=imgsz, device=device, verbose=False, stream=True)
    for r in results:
        if r.masks is not None:
            m = r.masks.data.cpu().numpy()
            frac_continuous = (9)).mean()
            print(f"[verify] mask value range: min={min():.4f} max={m.max():.4f} mean={m.mean():.4f}")
            print(f"[verify] fraction with continuous 

def main(args):
    # monkeypatch BEFORE loading/running the model, so every inference call uses the raw version
    ultralytics_ops.process_mask = process_mask_raw

    model = YOLO(args.model)

    with open(args.data) as f:
        data = yaml.safe_load(f)

    val_dir = Path(data["path"]) / data["val"]
    source_root = Path(args.original)

    print("Running verification check on raw probability extraction...")
    verify_raw_probabilities(model, val_dir, args.imgsz, args.device)

    # stream=True: process one image's results at a time instead of holding all
    # 17k+ images' masks in memory simultaneously
    results = model(source=str(val_dir), imgsz=args.imgsz, device=args.device, verbose=False, stream=True)

    # per class, per threshold: list of scalar IoU values (NOT raw masks). The
    # previous version stored every image's raw float32 mask for the WHOLE
    # dataset simultaneously (tens of GB in system RAM), which was very likely
    # being silently SIGKILL'd by the Linux OOM killer -- that kills a process
    # with no Python traceback at all, which is exactly why every crash so far
    # showed no error message. This version only ever keeps small scalar
    # floats per threshold, computed immediately per image.
    per_class_per_threshold_ious = {
        cls: {t: [] for t in THRESHOLDS} for cls in CLASS_NAMES
    }

    n_processed = 0
    for r in results:
        stem = Path(r.path).stem
        original = find_original_folder(stem, source_root)
        if original is None:
            del r
            continue

        mask_dir = original / "mask"

        raw_masks_by_class = {0: [], 1: []}
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            h, w = cv2.imread(str(next(original.glob("*_btot.png")))).shape[:2]
            for mask, cls in zip(masks, classes):
                cls = int(cls)
                if cls not in raw_masks_by_class:
                    continue
                resized = cv2.resize(mask, (w, h))
                raw_masks_by_class[cls].append(resized)

        for cls, suffix in CLASS_MASK_SUFFIX.items():
            gt_files = list(mask_dir.glob(f"*{suffix}"))
            if not gt_files:
                continue
            gt_mask = get_gt_mask(gt_files[0])
            raw_preds = raw_masks_by_class[cls]

            # compute IoU at every threshold RIGHT NOW, for this image only,
            # then let raw_preds get garbage collected -- nothing large is
            # ever held across images
            for t in THRESHOLDS:
                if not raw_preds:
                    per_class_per_threshold_ious[cls][t].append(0.0)
                    continue
                binarized_preds = [p > t for p in raw_preds]
                best_iou = max(iou(gt_mask, p) for p in binarized_preds)
                per_class_per_threshold_ious[cls][t].append(best_iou)

        n_processed += 1
        del r, raw_masks_by_class

    sweep_results = {name: {"median": [], "q1": [], "q3": []} for name in CLASS_NAMES.values()}

    for t in THRESHOLDS:
        for cls, name in CLASS_NAMES.items():
            vals = per_class_per_threshold_ious[cls][t]
            sweep_results[name]["median"].append(np.median(vals) if vals else 0.0)
            sweep_results[name]["q1"].append(np.percentile(vals, 25) if vals else 0.0)
            sweep_results[name]["q3"].append(np.percentile(vals, 75) if vals else 0.0)

    medians = sweep_results["cme"]["median"]
    best_idx = int(np.argmax(medians))
    best_threshold = THRESHOLDS[best_idx]
    best_median_iou = medians[best_idx]

    plt.figure(figsize=(10, 6))
    plt.fill_between(THRESHOLDS, sweep_results["cme"]["q1"], sweep_results["cme"]["q3"],
                      color="0.6", alpha=0.5, label="CME IoU IQR (Q1-Q3)")
    plt.plot(THRESHOLDS, sweep_results["cme"]["median"], color="black",
              linewidth=2.5, label="CME median IoU")
    plt.axvline(best_threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"best threshold = {best_threshold:.2f} (IoU = {best_median_iou:.4f})")

    plt.xlabel("Mask pixel-probability threshold")
    plt.ylabel("CME IoU")
    plt.title("CME IoU (median, Q1, Q3) vs mask threshold [raw probabilities]")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    out_path = Path(args.outdir) / "iou_vs_mask_threshold.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)

    out_path = Path(args.outdir) / "iou_vs_mask_threshold.png"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--original", default="~/synthetic_images/cme_seg_20250320_full_size")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--outdir", default="~/eval_outputs")
    args = parser.parse_args()
    args.original = str(Path(args.original).expanduser())
    args.outdir = str(Path(args.outdir).expanduser())
    main(args)
