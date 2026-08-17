import os
import cv2
import shutil
import random
from pathlib import Path
'''
the purpose of this code file is to convert the synthetic dataset of images into YOLO-ready datasets with the right format for training data
and masks, the right folder structure, etc.
'''
# location of the synthetic data in the folder
import argparse
SOURCE = Path("~/synthetic_images/cme_seg_20250320_full_size").expanduser()
OUT_BASE = Path("~/datasets").expanduser()

# this is the random seed used to randomly shuffle the data (for reproducibility) 
random.seed(42)

# i chose to make different dataset folders corresponding to different fractions of the dataset; one with 100% of the data, one with 50% of the data, etc
fractions = {
    "100": 1.0,
    "50": 0.5,
    "30": 0.3,
    "10": 0.1,
}

'''
this function converts binary masks into the format YOLO expects for ground truth masks, which is lists of points that make up the edges of
the polygonal shape that comprises the mask. this is done using OpenCV's open-source algorithm for obtaining polygonal edges from binary 
masks
'''
def mask_to_polygons(mask):
    mask = mask.astype("uint8")
    # open-source OpenCV function for finding distinct figures within the mask that are connected/polygonal, and converting them 
    # into lists of polygonal edges
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    polygons_list = []
    # for each polygon, 
    for c in contours:
        # filter out random noise/extremely small figures
        if cv2.contourArea(c) < 10:
            continue
        # remove extra dimensions
        c = c.squeeze()
        if len(c.shape) != 2:
            continue
        polygons_list.append(c)
        # returns final list of polygons. each polygon is itself a list of edges.
        # the final output looks something like: [ [[x1,y1],[x2,y2],[x3,y3]], [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ]
    return polygons_list

# make a list of folders
folders = [
    f for f in SOURCE.iterdir()
    # ignore folders whose names don't have numerals 
    if f.is_dir() and f.name.isdigit()
]
random.shuffle(folders)

# random train-test split
split = int(0.85 * len(folders))
train_all = folders[:split]
val_all = folders[split:]

'''
this function converts binary masks into the format YOLO expects for ground truth masks, which is lists of points that make up the edges of
the polygonal shape that comprises the mask. this is done using OpenCV's open-source algorithm for obtaining polygonal edges from binary 
masks
'''
def process_folder(folder, outdir, split_name):
    # extract the image from the folder (always the one that ends with _btot.png)
    image = list(folder.glob("*_btot.png"))
    if len(image) != 1:
        return
    image = image[0]
    shutil.copy(
        image,
        # cme_seg/images/train/xxx.btot.png, for example
        outdir / "images" / split_name / image.name
    )
    # create the path for labels. 
    # cme_seg/labels/train/xxx.btot.png, for example
    label_path = (
        outdir /
        "labels" /
        split_name /
        (image.stem + ".txt")
    )
    # normalize coordinates
    img = cv2.imread(str(image))
    height, width = img.shape[:2]
    lines=[]
    # clean up image names
    prefix=image.stem.replace("_btot","")
    # define classes (0.png is always the occulter image, 2.png is always the CME image)
    masks = [
        (0, folder/"mask"/f"{prefix}_mask_0.png"),
        (1, folder/"mask"/f"{prefix}_mask_2.png"),
    ]

    # loop thru all masks and their corresponding classes
    for cls, maskfile in masks:
        if not maskfile.exists():
            continue
        # load the mask
        mask=cv2.imread(
            str(maskfile),
            cv2.IMREAD_UNCHANGED
        )
        mask=mask[:,:,0]

        # loop through the polygons in each mask. write files containing masks in the format YOLO expects using masks_to_polygons() above 
        # this is because YOLO requires the following format: class x1 y1 x2 y2 x3 y3 ..., where x1, y1, etc are normalized to [0,1]
        # for example, 1 0.12 0.44 0.15 0.50 0.20 0.55 0.18 0.60 is a mask for a class-1 object with the polygon borders (0.12,0.44),
        # (0.15,0.50), (0.20,0.55), etc.
        for poly in mask_to_polygons(mask):
            coords=[]
            for x,y in poly:
                coords.append(str(x/width))
                coords.append(str(y/height))
            lines.append(
                str(cls)+" "+" ".join(coords)
            )
        
    label_path.write_text(
        "\n".join(lines)
    )
# loop for each dataset i wished to create (10% of full data, 30% of full data, 50% of full data, etc)
for name, frac in fractions.items():
    # create the root directory for this dataset
    outdir = OUT_BASE / f"cme_yolo_seg_{name}"
    # get the correct amount of training and validation data by multiplying by the required fraction 
    selected_train = train_all[:int(len(train_all)*frac)]
    selected_val = val_all
    for split_name, folderset in [
        ("train", selected_train),
        ("val", selected_val)
    ]:
        # create output directories (images/train, labels/train, images/val, labels/val)
        for d in [
            outdir/"images"/split_name,
            outdir/"labels"/split_name
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # use process_folder() to find the images and masks in each image folder, copy them to the right directory, and convert 
        # segmentation masks to the required format for YOLO
        for folder in folderset:
            process_folder(
                folder,
                outdir,
                split_name
            )
    # creates the file data.yaml. this is a config file required for YOLO, that directs YOLO on where each part of the data (train vs test 
    # and images vs masks) are found in the config file, and what the class labels are
    yaml = f"""
path: {outdir}

train: images/train
val: images/val

names:
  0: occulter
  1: cme
"""


    (outdir/"data.yaml").write_text(yaml)
    print(
        name,
        "created:",
        len(selected_train),
        "train folders",
        len(selected_val),
        "val folders"
    )
