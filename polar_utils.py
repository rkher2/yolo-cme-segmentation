import cv2
import numpy as np


def get_polar_params(img_shape):
    # parameters for polar transformation (assumes occulter is perfectly centered)
    h, w = img_shape[:2]
    center = (w / 2, h / 2)
    max_radius = min(w, h) / 2
    return center, max_radius


def to_polar(img, center, max_radius, dsize, is_mask=False):
    # cartesian to polar transformation for an image
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpPolar(
        img, dsize, center, max_radius,
        flags=interp + cv2.WARP_FILL_OUTLIERS
    )
