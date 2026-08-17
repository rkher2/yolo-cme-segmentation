import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

""" This file runs inference with the trained model on real CME observations, drawing predicted masks and class labels. Note that
since these are real, non-synthetic images there is no ground truth to compare to.
"""
CLASS_COLORS = {0: (0, 255, 255), 1: (255, 0, 0)}  # occulter=cyan, cme=red
CLASS_NAMES = {0: "occulter", 1: "cme"}


def draw_predictions(img_rgb, result):

    """Draw predicted masks, bounding boxes, class labels, and confidence
    scores directly onto the image"""
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


def main(args):
    model = YOLO(args.model)
    image_paths = sorted(Path(args.image_dir).glob("*.png"))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    for img_path in image_paths:
        orig = cv2.imread(str(img_path))
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        results = model(source=str(img_path), imgsz=args.imgsz, device=args.device, verbose=False)
        r = results[0]

        # get image exactly as YOLO processed it
        fed_img = cv2.resize(orig, (args.imgsz, args.imgsz))
        fed_rgb = cv2.cvtColor(fed_img, cv2.COLOR_BGR2RGB)
        annotated = draw_predictions(fed_rgb, r)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"Original observation\n{img_path.name}", fontsize=9)
        axes[0].axis("off")

        n_detections = 0 if r.masks is None else len(r.masks.data)
        axes[1].imshow(annotated)
        axes[1].set_title(f"YOLO input + predictions (n={n_detections})", fontsize=9)
        axes[1].axis("off")

        plt.tight_layout()
        out_path = outdir / f"{img_path.stem}_pred.png"
        plt.savefig(out_path, dpi=100)
        plt.close()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained best.pt")
    parser.add_argument("--image_dir", default="~/real_obs_testing")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.image_dir = str(Path(args.image_dir).expanduser())
    args.outdir = str(Path(args.outdir).expanduser())
    main(args)
