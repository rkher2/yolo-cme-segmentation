import csv
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
from seg_metrics import get_gt_mask, iou, dice, find_original_folder
from polar_utils import get_polar_params, from_polar

"""This file is the version of eval_final corresponding to polar runs, ensuring that polar masks are converted to cartesian before 
evaluation so that an accurate comparison to cartesian methods and easier visualization can be established."""
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
})

CLASS_MASK_SUFFIX = {0: "_mask_0.png", 1: "_mask_2.png"}
CLASS_NAMES = {0: "occulter", 1: "cme"}
PQ_IOU_THRESHOLD = 0.5


def main(args):
    model = YOLO(args.model)
    with open(args.polar_data) as f:
        data = yaml.safe_load(f)
    val_dir = Path(data["path"]) / data["val"]
    cartesian_source_root = Path(args.cartesian_original)

    pred_mask_dir = Path(args.outdir) / "pred_masks_cartesian"
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    results = model(source=str(val_dir), imgsz=args.imgsz, device=args.device, verbose=False, stream=True)

    records = {0: [], 1: []}
    per_image_rows = []
    n_processed = 0

    for r in results:
        stem = Path(r.path).stem
        original = find_original_folder(stem, cartesian_source_root)
        if original is None:
            continue

        orig_image_path = next(original.glob("*_btot.png"))
        mask_dir = original / "mask"

        cart_img = cv2.imread(str(orig_image_path))
        h, w = cart_img.shape[:2]
        center, max_radius = get_polar_params(cart_img.shape)

        pred_masks_by_class = {0: [], 1: []}
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy() 
            classes = r.boxes.cls.cpu().numpy()
            for mask, cls in zip(masks, classes):
                cls = int(cls)
                if cls not in pred_masks_by_class:
                    continue
                # resize to the polar dsize used during training (should already match)
                mask_resized = cv2.resize(mask, (args.imgsz, args.imgsz))
                mask_binary_u8 = (mask_resized > 0.5).astype(np.uint8) * 255
                # transform this binary mask back to Cartesian space
                mask_cartesian = from_polar(mask_binary_u8, center, max_radius, (w, h), is_mask=True)
                pred_masks_by_class[cls].append(mask_cartesian > 0)

        row = {"image": stem, "orig_image_path": str(orig_image_path)}

        for cls, suffix in CLASS_MASK_SUFFIX.items():
            gt_files = list(mask_dir.glob(f"*{suffix}"))
            name = CLASS_NAMES[cls]

            if not gt_files:
                row[f"{name}_gt_mask_path"] = ""
                row[f"{name}_pred_mask_path"] = ""
                row[f"{name}_iou"] = ""
                row[f"{name}_f1_dice"] = ""
                row[f"{name}_pq"] = ""
                row[f"{name}_num_preds"] = ""
                continue

            gt_mask = get_gt_mask(gt_files[0])  # real Cartesian GT
            preds = pred_masks_by_class[cls]

            if preds:
                ious_this_image = [iou(gt_mask, p) for p in preds]
                best_idx = int(np.argmax(ious_this_image))
                best_iou = ious_this_image[best_idx]
                best_mask = preds[best_idx]
                best_dice = dice(gt_mask, best_mask)

                pred_mask_path = pred_mask_dir / f"{stem}_{name}_pred_cartesian.png"
                cv2.imwrite(str(pred_mask_path), (best_mask.astype(np.uint8)) * 255)
            else:
                best_iou = 0.0
                best_dice = 0.0
                pred_mask_path = None

            records[cls].append({"best_iou": best_iou, "num_preds": len(preds)})

            is_match = best_iou > PQ_IOU_THRESHOLD
            row[f"{name}_gt_mask_path"] = str(gt_files[0])
            row[f"{name}_pred_mask_path"] = str(pred_mask_path) if pred_mask_path else ""
            row[f"{name}_iou"] = round(best_iou, 4)
            row[f"{name}_f1_dice"] = round(best_dice, 4)
            row[f"{name}_pq"] = round(best_iou, 4) if is_match else 0.0
            row[f"{name}_num_preds"] = len(preds)

        per_image_rows.append(row)
        n_processed += 1
        if n_processed % 1000 == 0:
            print(f"  processed {n_processed} images...")

    pq_per_class = {}
    pq_values = []

    for cls, name in CLASS_NAMES.items():
        recs = records[cls]
        tp_ious = [r_["best_iou"] for r_ in recs if r_["best_iou"] > PQ_IOU_THRESHOLD]
        tp = len(tp_ious)
        fn = sum(1 for r_ in recs if r_["num_preds"] == 0 or r_["best_iou"] <= PQ_IOU_THRESHOLD)
        fp = sum(max(r_["num_preds"] - 1, 0) for r_ in recs)

        sq = np.mean(tp_ious) if tp_ious else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq
        pq_per_class[name] = pq
        pq_values.append(pq)

        print(f"\n[{name}]")
        print(f"  TP={tp}  FP={fp}  FN={fn}")
        print(f"  SQ: {sq:.4f}")
        print(f"  RQ: {rq:.4f}")
        print(f"  PQ: {pq:.4f}")

        all_best_ious = [r_["best_iou"] for r_ in recs]
        if name == "cme":
            in_range = sum(1 for v in all_best_ious if 0.5 <= v <= 1.0)
            plt.figure()
            plt.hist(all_best_ious, bins=50, range=(0.5, 1.0), color="0.6", edgecolor="black")
            plt.axvline(PQ_IOU_THRESHOLD, color="red", linestyle="--")
            plt.yscale("log")
            plt.title(f"CME IoU (polar model, Cartesian eval) (n={in_range} of {len(all_best_ious)})")
            plt.xlabel("IoU")
            plt.ylabel("count (log scale)")
            out_path = Path(args.outdir) / f"iou_hist_{name}_cartesian.png"
            plt.savefig(out_path)
            print(f"  saved histogram -> {out_path}")

    csv_path = Path(args.outdir) / "per_image_metrics.csv"
    fieldnames = ["image", "orig_image_path"]
    for name in CLASS_NAMES.values():
        fieldnames += [f"{name}_gt_mask_path", f"{name}_pred_mask_path",
                       f"{name}_iou", f"{name}_f1_dice", f"{name}_pq", f"{name}_num_preds"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a polar-trained best.pt")
    parser.add_argument("--polar_data", required=True, help="Polar dataset's data.yaml")
    parser.add_argument("--cartesian_original", default="~/synthetic_images/cme_seg_20250320_full_size")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.cartesian_original = str(Path(args.cartesian_original).expanduser())
    args.outdir = str(Path(args.outdir).expanduser())
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    main(args)

