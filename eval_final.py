# eval_final.py
import csv
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml
from ultralytics import YOLO
from seg_metrics import get_gt_mask, iou, dice, find_original_folder
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})
# dicts defining classes
CLASS_MASK_SUFFIX = {0: "_mask_0.png", 1: "_mask_2.png"}
CLASS_NAMES = {0: "occulter", 1: "cme"}

# definition from paper about PQ
PQ_IOU_THRESHOLD = 0.5  



def main(args):
    # load trained model + validation images
    model = YOLO(args.model)
    # read data.yaml config file to reach data
    with open(args.data) as f:
        data = yaml.safe_load(f)
    val_dir = Path(data["path"]) / data["val"]
    source_root = Path(args.original)

    # folder for saving predicted mask images, so the CSV can point to real files on disk
    pred_mask_dir = Path(args.outdir) / "pred_masks"
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    # run inference to get results
    results = model(source=str(val_dir), imgsz=args.imgsz, device=args.device, verbose=False)

    records = {0: [], 1: []}
    per_image_rows = []

    # loop thru every image 
    for r in results:
        stem = Path(r.path).stem
        # recover its original folder
        original = find_original_folder(stem, source_root)
        if original is None:
            print("Missing original for:", stem)
            continue

        orig_image_path = next(original.glob("*_btot.png"))

        # find the mask files corresponding to the image 
        mask_dir = original / "mask"
        pred_masks_by_class = {0: [], 1: []}
        if r.masks is not None:
            # get masks, their corresponding classes
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            h, w = cv2.imread(str(orig_image_path)).shape[:2]
            for mask, cls in zip(masks, classes):
                cls = int(cls)
                if cls not in pred_masks_by_class:
                    continue
                # convert values to True/False 
                resized = cv2.resize(mask, (w, h)) > 0.5
                pred_masks_by_class[cls].append(resized)

        row = {"image": stem, "orig_image_path": str(orig_image_path)}

        # process occulter first then CME
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

            # load binary mask for this object (whole mask = one instance,
            # even if it renders as disconnected regions -- that's still one object)
            gt_mask = get_gt_mask(gt_files[0])

            preds = pred_masks_by_class[cls]

            if preds:
                # pick the highest-matching IoU. for example, if the model predicts 3 CME, pick the one with highest IoU to evaluate overlap with
                # the ground-truth CME (with PQ it will get penalized for this by the other term anyway)
                ious_this_image = [iou(gt_mask, p) for p in preds]
                best_idx = int(np.argmax(ious_this_image))
                best_iou = ious_this_image[best_idx]
                best_mask = preds[best_idx]
                best_dice = dice(gt_mask, best_mask)

                # save the best-matching predicted mask to disk so the CSV has a real file to point to
                pred_mask_path = pred_mask_dir / f"{stem}_{name}_pred.png"
                cv2.imwrite(str(pred_mask_path), (best_mask.astype(np.uint8)) * 255)
            else:
                best_iou = 0.0
                best_dice = 0.0
                pred_mask_path = None

            # save records (the number of predicted masks and the best IoU) 
            records[cls].append({"best_iou": best_iou, "num_preds": len(preds)})

            # per-image pseudo-PQ: IoU if matched above threshold, else 0 for a miss
            is_match = best_iou > PQ_IOU_THRESHOLD
            row[f"{name}_gt_mask_path"] = str(gt_files[0])
            row[f"{name}_pred_mask_path"] = str(pred_mask_path) if pred_mask_path else ""
            row[f"{name}_iou"] = round(best_iou, 4)
            row[f"{name}_f1_dice"] = round(best_dice, 4)
            row[f"{name}_pq"] = round(best_iou, 4) if is_match else 0.0
            row[f"{name}_num_preds"] = len(preds)

        per_image_rows.append(row)

    print(f"panoptic quality")
    pq_per_class = {}
    pq_values = []

    for cls, name in CLASS_NAMES.items():
        recs = records[cls]

        # true positives
        tp_ious = [r_["best_iou"] for r_ in recs if r_["best_iou"] > PQ_IOU_THRESHOLD]
        tp = len(tp_ious)

        # false negatives
        fn = sum(1 for r_ in recs if r_["num_preds"] == 0 or r_["best_iou"] <= PQ_IOU_THRESHOLD)

        # false positives
        fp = sum(max(r_["num_preds"] - 1, 0) for r_ in recs)

        # calculate PQ 
        sq = np.mean(tp_ious) if tp_ious else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq
        pq_per_class[name] = pq

        pq_values.append(pq)

        print(f"\n[{name}]")
        print(f"  TP={tp}  FP={fp}  FN={fn}")
        print(f"  SQ: {sq:.4f}")
        print(f"  RQ:                   {rq:.4f}")
        print(f"  PQ:                   {pq:.4f}")

        all_best_ious = [r_["best_iou"] for r_ in recs]

        # histogram only generated for CME (per earlier "CME only graphs" request)
        if name == "cme":
            plt.figure()
            plt.hist(all_best_ious, bins=50, range=(0.5, 1.0), color="0.6", edgecolor="black")
            plt.axvline(PQ_IOU_THRESHOLD, color="red", linestyle="--")
            plt.yscale("log")
            plt.title(f"CME best-match IoU per image (n={len(all_best_ious)})")
            plt.xlabel("IoU")
            plt.ylabel("count (log scale)")
            out_path = Path(args.outdir) / f"iou_hist_{name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            print(f"  saved histogram -> {out_path}")
        print(f"  mean IoU: {np.mean(all_best_ious):.4f}")
        print(f"  median IoU: {np.median(all_best_ious):.4f}")

        pct_above_95 = 100 * np.mean(np.array(all_best_ious) >= 0.95)
        print(f"  % of images with IoU >= 0.95: {pct_above_95:.2f}%  (n={len(all_best_ious)})")

    print(f"\n mean PQ across classes: {np.mean(pq_values):.4f}")
    print(f"PQ of CMEs: {pq_per_class['cme']:.4f}")

    # write per-image metrics CSV
    csv_path = Path(args.outdir) / "per_image_metrics.csv"
    fieldnames = ["image", "orig_image_path"]
    for name in CLASS_NAMES.values():
        fieldnames += [
            f"{name}_gt_mask_path", f"{name}_pred_mask_path",
            f"{name}_iou", f"{name}_f1_dice", f"{name}_pq", f"{name}_num_preds"
        ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(f"\nsaved per-image metrics -> {csv_path}")
    print(f"saved predicted mask images -> {pred_mask_dir}")


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
