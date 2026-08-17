"""This is a custom script for training runs, necessary due to certain specifications of this project"""

import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

from seg_metrics import get_gt_mask, iou

PQ_IOU_THRESHOLD = 0.5  # this follows the textbook definition/paper definition of PQ
CLASS_MASK_SUFFIX = {0: "_mask_0.png", 1: "_mask_2.png"}
CME_CLASS = 1

_folder_cache = {}

def disable_builtin_albumentations():
    """
    Ultralytics calls albumentations (automatic data augmentations) internally. This code neutralizes this, ensuring that augmentation is off by default,
    and individual ones must be turned on manually with flags (see below). Note that the flag --disable_builtin_albumentations should not be used with the flag
    to enable the gaussian noise augmentation, --gaussian_noise_p 
    """
    try:
        from ultralytics.data.augment import Albumentations

        def disabled_init(self, p=1.0, transforms=None, **kwargs):
            self.p = 0.0
            self.transform = None
            self.contains_spatial = False

        Albumentations.__init__ = disabled_init
    except Exception as e:
        print(f"[patch]note:failed to disable built-in augmentations")

# stores the folder corresponding to each image 
def find_original_folder(stem, source_root):
    if not _folder_cache:
        for folder in source_root.iterdir():
            if folder.is_dir():
                for img in folder.glob("*_btot.png"):
                    _folder_cache[img.stem] = folder
    return _folder_cache.get(stem)

# takes a model, yaml, dataset location, and returns mean/median/Q1/Q3 IoU and PQ
def run_eval(model, data_yaml, original_root, imgsz, device):
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    # locate validation data
    val_dir = Path(data["path"]) / data["val"]
    source_root = Path(original_root).expanduser()
    # run inference with the trained YOLO model
    results = model(source=str(val_dir), imgsz=imgsz, device=device, verbose=False)

    records = {0: [], 1: []}
    # process validation data (for every validation image)
    for r in results:
        stem = Path(r.path).stem
        original = find_original_folder(stem, source_root)
        if original is None:
            continue
        mask_dir = original / "mask"
        # produce predicted masks
        pred_masks_by_class = {0: [], 1: []}
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            h, w = cv2.imread(str(next(original.glob("*_btot.png")))).shape[:2]
            for mask, cls in zip(masks, classes):
                cls = int(cls)
                if cls in pred_masks_by_class:
                    pred_masks_by_class[cls].append(cv2.resize(mask, (w, h)) > 0.5)

        for cls, suffix in CLASS_MASK_SUFFIX.items():
            gt_files = list(mask_dir.glob(f"*{suffix}"))
            if not gt_files:
                continue
            gt_mask = get_gt_mask(gt_files[0])
            preds = pred_masks_by_class[cls]
            # store per-image validation result     
            best_iou = max((iou(gt_mask, p) for p in preds), default=0.0)
            records[cls].append({"best_iou": best_iou, "num_preds": len(preds)})

    # CME-only metrics (class 1). Occulter metrics are still computed but not used in the returned/logged values below.
    cme_recs = records[CME_CLASS]
    # calculate IoUs and PQ
    tp_ious = [r_["best_iou"] for r_ in cme_recs if r_["best_iou"] > PQ_IOU_THRESHOLD]
    tp = len(tp_ious)
    fn = sum(1 for r_ in cme_recs if r_["num_preds"] == 0 or r_["best_iou"] <= PQ_IOU_THRESHOLD)
    fp = sum(max(r_["num_preds"] - 1, 0) for r_ in cme_recs)

    sq = np.mean(tp_ious) if tp_ious else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    cme_pq = sq * rq

    cme_ious = [r_["best_iou"] for r_ in cme_recs]
    mean_iou = np.mean(cme_ious) if cme_ious else 0.0
    median_iou = np.median(cme_ious) if cme_ious else 0.0
    q1_iou = np.percentile(cme_ious, 25) if cme_ious else 0.0
    q3_iou = np.percentile(cme_ious, 75) if cme_ious else 0.0

    return mean_iou, median_iou, q1_iou, q3_iou, cme_pq

# creates a callback function that Ultralytics can use during training, to run my custom evaluation and calculate my custom metrics every epoch,
# without needing to halt training
def make_callback(data_yaml, original_root, eval_every):
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        if epoch % eval_every != 0:
            return

        import torch
        import gc

        # free memory before eval
        torch.cuda.empty_cache()
	
	# save weights
        weights_path = trainer.save_dir / "weights" / "last.pt"
        model = YOLO(str(weights_path))

        mean_iou, median_iou, q1_iou, q3_iou, mean_pq = run_eval(
            model, data_yaml, original_root,
            imgsz=trainer.args.imgsz, device=trainer.args.device
        )

        log_path = trainer.save_dir / "epoch_metrics.csv"
        write_header = not log_path.exists()

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "q1_iou", "median_iou", "q3_iou", "mean_iou", "mean_pq"])
            writer.writerow([epoch, q1_iou, median_iou, q3_iou, mean_iou, mean_pq])

        print(f"[epoch {epoch}] q1_iou={q1_iou:.4f} median_iou={median_iou:.4f} "
              f"q3_iou={q3_iou:.4f} mean_iou={mean_iou:.4f} mean_pq={mean_pq:.4f}")

        # release the temporary eval model before returning to the training loop
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return on_fit_epoch_end

def main(args):
    if args.disable_builtin_albumentations:
        disable_builtin_albumentations()

    data_yaml = Path(args.data).expanduser().resolve()
    original_root = Path(args.original).expanduser().resolve()
    project = Path(args.project).expanduser()

    if args.polar:
        from create_polar_dataset import build_polar_dataset

        cache_name = data_yaml.parent.name  # e.g. cme_yolo_seg_100
        polar_data_out = Path(f"~/datasets_polar_cache/{cache_name}_polar").expanduser()
        polar_original_out = Path(f"~/datasets_polar_cache/{cache_name}_polar_original").expanduser()

        data_yaml, original_root = build_polar_dataset(
            cartesian_data=data_yaml.parent,
            original=original_root,
            polar_data_out=polar_data_out,
            polar_original_out=polar_original_out,
            polar_width=args.polar_width,
            polar_height=args.polar_height,
            force=args.force_polar_regen,
        )
        print(f"training on polar dataset: {data_yaml}")
        print(f"eval scripts for this run should use --original {original_root}")

    custom_transforms = []
    if args.gaussian_noise_p > 0.0:
        import albumentations as A
        custom_transforms.append(
            A.GaussNoise(var_limit=(0.0, args.gaussian_noise_std), p=args.gaussian_noise_p)
        )

    model = YOLO(args.model)
    model.add_callback(
        "on_fit_epoch_end",
        make_callback(data_yaml, original_root, args.eval_every)
    )

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        workers=args.workers,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        bgr=args.bgr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        auto_augment=args.auto_augment,
        crop_fraction=args.crop_fraction,
        augmentations=custom_transforms if custom_transforms else None,
    )

# note all the flag that can be used to modify training below
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to data.yaml, e.g. ~/datasets/cme_yolo_seg_50/data.yaml")
    parser.add_argument("--original", default="~/synthetic_images/cme_seg_20250320_full_size", help="Path to original source dataset (for GT masks)")

    parser.add_argument("--model", default="yolo26s-seg.pt")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="~/runs/cme_yolo")
    parser.add_argument("--name", required=True, help="Run name, e.g. yolo26s_50pct")
    parser.add_argument("--eval_every", default=10, type=int)

    parser.add_argument("--hsv_h", default=0.0, type=float)
    parser.add_argument("--hsv_s", default=0.0, type=float)
    parser.add_argument("--hsv_v", default=0.0, type=float)
    parser.add_argument("--degrees", default=0.0, type=float)
    parser.add_argument("--translate", default=0.0, type=float)
    parser.add_argument("--scale", default=0.0, type=float)
    parser.add_argument("--shear", default=0.0, type=float)
    parser.add_argument("--perspective", default=0.0, type=float)
    parser.add_argument("--flipud", default=0.0, type=float)
    parser.add_argument("--fliplr", default=0.0, type=float)
    parser.add_argument("--bgr", default=0.0, type=float)
    parser.add_argument("--mosaic", default=0.0, type=float)
    parser.add_argument("--mixup", default=0.0, type=float)
    parser.add_argument("--copy_paste", default=0.0, type=float)
    parser.add_argument("--erasing", default=0.0, type=float)
    parser.add_argument("--auto_augment", default=None, type=str, help="e.g. 'randaugment' — None disables it")
    parser.add_argument("--crop_fraction", default=1.0, type=float, help="1.0 = no crop augmentation")
    parser.add_argument("--disable_builtin_albumentations", action="store_true",
                     help="Neutralize Ultralytics' automatic default Albumentations pipeline (Blur/MedianBlur/ToGray/CLAHE)")
    parser.add_argument("--gaussian_noise_std", default=0.0, type=float,
                         help="Upper bound of variance for Gaussian noise (Albumentations var_limit). 0 = disabled.")
    parser.add_argument("--gaussian_noise_p", default=0.0, type=float,
                         help="Probability of applying Gaussian noise per image. 0 = disabled.")

    parser.add_argument("--polar", action="store_true",
                         help="Train on a polar-coordinate transform of the dataset (auto-generated, cached, original data untouched)")
    parser.add_argument("--polar_width", default=512, type=int)
    parser.add_argument("--polar_height", default=512, type=int)
    parser.add_argument("--force_polar_regen", action="store_true",
                         help="Rebuild the cached polar dataset even if it already exists")
    parser.add_argument("--workers", default=4, type=int, help="Number of dataloader workers")

    args = parser.parse_args()
    main(args)
