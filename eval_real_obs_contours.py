# this script runs inference on real CME observations, generating predicted masks and probability contour maps 

import argparse
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils import ops as ultralytics_ops
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
from seg_metrics import process_mask_raw_uncropped

# occulter (0) = cyan. cme (1) = red.
CLASS_COLORS = {0: (0, 255, 255), 1: (255, 0, 0)}  
CLASS_NAMES = {0: "occulter", 1: "cme"}
DENSE_LEVELS = np.arange(0.05, 1.0, 0.05)


def draw_predictions(img_rgb, result):
    """ This function creates images that overlay the predicted masks, bounding boxes, and confidence onto the image YOLO saw"""
    overlay = img_rgb.copy()

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        h, w = overlay.shape[:2]

        for mask, box, cls, conf in zip(masks, boxes, classes, confs):
            cls = int(cls)
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            name = CLASS_NAMES.get(cls, f"class{cls}")

            mask_resized = cv2.resize(mask, (w, h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

            label = f"{name} {conf:.2f}"
            cv2.putText(overlay, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return overlay


def get_cme_probability_map(model, img_path, imgsz, device):
    """This function pulls out the CME probability map for each pixel."""
    original_process_mask = ultralytics_ops.process_mask
    ultralytics_ops.process_mask = process_mask_raw_uncropped
    try:
        results = model(source=str(img_path), imgsz=imgsz, device=device, verbose=False)
    finally:
        ultralytics_ops.process_mask = original_process_mask
    r = results[0]

    if r.masks is None:
        return None

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    cme_masks = [m for m, c in zip(masks, classes) if int(c) == 1]
    return cme_masks[0] if cme_masks else None


def main(args):
    image_dir = Path(args.image_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_indices = list(range(args.img_start, args.img_end + 1))

    model_pred = YOLO(args.model)   # standard predictions
    model_prob = YOLO(args.model)   # separate instance for the probability-contour plot

    for idx in image_indices:
        img_path = image_dir / f"img_{idx}.png"


        orig = cv2.imread(str(img_path))
        fed_img = cv2.resize(orig, (args.imgsz, args.imgsz))
        fed_rgb = cv2.cvtColor(fed_img, cv2.COLOR_BGR2RGB)

        # predicted mask,box,label,confidence plot
        results = model_pred(source=str(img_path), imgsz=args.imgsz, device=args.device, verbose=False)
        r = results[0]
        annotated = draw_predictions(fed_rgb, r)

        n_detections = 0 if r.masks is None else len(r.masks.data)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(annotated)
        ax.set_title(f"{args.model_name} -- img_{idx} -- predictions (n={n_detections})", fontsize=9)
        ax.axis("off")
        plt.tight_layout()
        pred_out = outdir / f"{args.model_name}_img_{idx}_predictions.png"
        plt.savefig(pred_out, dpi=100)
        plt.close()

        # probability contour map
        prob_map = get_cme_probability_map(model_prob, img_path, args.imgsz, args.device)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(fed_rgb)
        if prob_map is not None:
            prob_resized = cv2.resize(prob_map, (args.imgsz, args.imgsz))
            cs = ax.contour(prob_resized, levels=DENSE_LEVELS, cmap="turbo", linewidths=0.9)
            ax.clabel(cs, inline=True, fontsize=5, fmt="%.2f")
            ax.set_title(f"{args.model_name} -- img_{idx} -- CME probability contours", fontsize=9)
        else:
            ax.set_title(f"{args.model_name} -- img_{idx} -- [no prediction]", fontsize=9)
        ax.axis("off")
        plt.tight_layout()
        contour_out = outdir / f"{args.model_name}_img_{idx}_contours.png"
        plt.savefig(contour_out, dpi=100)
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--model_name", required=True, help="Short label used in output filenames, e.g. yolo26n")
    parser.add_argument("--image_dir", default="~/real_obs_SPD")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--img_start", default=69, type=int)
    parser.add_argument("--img_end", default=99, type=int)
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.image_dir = str(Path(args.image_dir).expanduser())
    args.outdir = str(Path(args.outdir).expanduser())
    main(args)
