import cv2
import numpy as np
import json
from pathlib import Path
import torch
from ultralytics.utils import ops as ultralytics_ops
_folder_index = None
def process_mask_raw(protos, masks_in, bboxes, shape, upsample: bool = False):
    """This is a reimplementation of ultralytics.utils.ops.process_mask WITHOUT the final
    .gt_(0.0).byte() binarization step. This way we can return continuous sigmoid probabilities
    instead of a binary mask."""
    c, mh, mw = protos.shape
    if masks_in.shape[0] == 0:
        return torch.zeros((0, *(shape if upsample else (mh, mw))), dtype=torch.float32, device=masks_in.device)

    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)

    if upsample:
        masks = torch.nn.functional.interpolate(masks[None], shape, mode="bilinear")[0]

    masks = torch.sigmoid(masks)
    return ultralytics_ops.crop_mask(masks, bboxes) # probabilistic object mask rather than binary mask

def process_mask_raw_uncropped(protos, masks_in, bboxes, shape, upsample: bool = False):
    """Reimplementation that doesn't crop masks. this way we get a probability field for the entire image/probabilities for every pixel"""
    c, mh, mw = protos.shape
    if masks_in.shape[0] == 0:
        return torch.zeros((0, *(shape if upsample else (mh, mw))), dtype=torch.float32, device=masks_in.device)

    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)

    if upsample:
        masks = torch.nn.functional.interpolate(masks[None], shape, mode="bilinear")[0]

    return torch.sigmoid(masks)  # full probability field
_folder_index = None

_folder_index_cache = {}  

def find_original_folder(image_stem, source_root, cache=None):
    source_root = Path(source_root)
    key = str(source_root)

    if key not in _folder_index_cache:
        # try cached index file specific to this source_root first
        index_path = Path(f"~/folder_index_{source_root.name}.json").expanduser()
        if index_path.exists():
            with open(index_path) as f:
                _folder_index_cache[key] = json.load(f)
        else:
            # if not then build it fresh
            index = {}
            for folder in source_root.iterdir():
                if folder.is_dir():
                    for img in folder.glob("*_btot.png"):
                        index[img.stem] = str(folder)
            _folder_index_cache[key] = index
            with open(index_path, "w") as f:
                json.dump(index, f)


    path = _folder_index_cache[key].get(image_stem)
    return Path(path) if path else None

# convert mask image to numpy binary mask
def get_gt_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 0

# calculate IoU between a ground truth mask and a predicted mask
def iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return intersection / union
def dice(a, b):
    #F1 score for binary masks
    intersection = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2 * intersection / denom
