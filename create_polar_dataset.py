import argparse
from pathlib import Path
import cv2
import numpy as np
import shutil

from polar_utils import get_polar_params, to_polar


def mask_to_polygons(mask):
    mask = (mask > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 10:
            continue
        c = c.squeeze()
        if len(c.shape) != 2:
            continue
        polys.append(c)
    return polys


def process_folder(folder, polar_source_out, dsize):
    image_files = list(folder.glob("*_btot.png"))
    if len(image_files) != 1:
        return None
    image_path = image_files[0]
    prefix = image_path.stem.replace("_btot", "")

    img = cv2.imread(str(image_path))
    center, max_radius = get_polar_params(img.shape)
    img_polar = to_polar(img, center, max_radius, dsize, is_mask=False)

    out_folder = polar_source_out / folder.name
    (out_folder / "mask").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_folder / image_path.name), img_polar)

    mask_paths = {
        0: folder / "mask" / f"{prefix}_mask_0.png",
        2: folder / "mask" / f"{prefix}_mask_2.png",
    }
    polar_masks = {}
    for suffix, mpath in mask_paths.items():
        if not mpath.exists():
            continue
        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        m = m[:, :, 0] if m.ndim == 3 else m
        m_polar = to_polar(m, center, max_radius, dsize, is_mask=True)
        polar_masks[suffix] = m_polar
        cv2.imwrite(str(out_folder / "mask" / mpath.name), m_polar)

    return prefix, img_polar.shape[:2], polar_masks


def build_yolo_split(image_stem, img_shape, polar_masks, yolo_out, split_name, img_path):
    h, w = img_shape
    shutil.copy(img_path, yolo_out / "images" / split_name / img_path.name)

    lines = []
    class_map = {0: 0, 2: 1}  # mask_0 corresponds to class 0 (occulter), mask_2 corresponds to class 1 (cme)
    for suffix, cls in class_map.items():
        if suffix not in polar_masks:
            continue
        for poly in mask_to_polygons(polar_masks[suffix]):
            coords = []
            for x, y in poly:
                coords.append(str(x / w))
                coords.append(str(y / h))
            lines.append(str(cls) + " " + " ".join(coords))

    label_path = yolo_out / "labels" / split_name / (img_path.stem + ".txt")
    label_path.write_text("\n".join(lines))


def build_polar_dataset(cartesian_data, original, polar_data_out, polar_original_out,
                         polar_width=512, polar_height=512, force=False):
    yolo_out = Path(polar_data_out).expanduser()
    polar_source_out = Path(polar_original_out).expanduser()

    if (yolo_out / "data.yaml").exists() and not force:
        return yolo_out / "data.yaml", polar_source_out

    cartesian_yolo_dir = Path(cartesian_data).expanduser()
    source_root = Path(original).expanduser()
    dsize = (polar_width, polar_height)

    for split_name in ["train", "val"]:
        (yolo_out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (yolo_out / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        existing_images = (cartesian_yolo_dir / "images" / split_name).glob("*_btot.png")
        for img_path in existing_images:
            prefix = img_path.stem.replace("_btot", "")
            src_folder = None
            for folder in source_root.iterdir():
                if folder.is_dir() and (folder / img_path.name).exists():
                    src_folder = folder
                    break
            if src_folder is None:
                continue

            result = process_folder(src_folder, polar_source_out, dsize)
            if result is None:
                continue
            _, shape, polar_masks = result
            polar_img_path = polar_source_out / src_folder.name / img_path.name
            build_yolo_split(prefix, shape, polar_masks, yolo_out, split_name, polar_img_path)

    yaml_text = f"""

train: images/train
val: images/val

names:
  0: occulter
  1: cme
"""
    (yolo_out / "data.yaml").write_text(yaml_text)
    print(f"Polar dataset written -> {yolo_out}")
    return yolo_out / "data.yaml", polar_source_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartesian_data", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--polar_data_out", required=True)
    parser.add_argument("--polar_original_out", required=True)
    parser.add_argument("--polar_width", default=512, type=int)
    parser.add_argument("--polar_height", default=512, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_polar_dataset(
        args.cartesian_data, args.original, args.polar_data_out, args.polar_original_out,
        args.polar_width, args.polar_height, args.force
    )
