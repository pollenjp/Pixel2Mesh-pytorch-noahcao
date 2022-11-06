# Standard Library
from pathlib import Path

# dataset root
DATASET_ROOT = Path("datasets") / "data"
SHAPENET_ROOT = DATASET_ROOT / "shapenet"
SHAPENET_WITH_TEMPLATE_ROOT = DATASET_ROOT / "shapenet_with_template"
IMAGENET_ROOT = DATASET_ROOT / "imagenet"

# ellipsoid path
ELLIPSOID_PATH = DATASET_ROOT / "ellipsoid/info_ellipsoid.dat"

# pretrained weights path
PRETRAINED_WEIGHTS_PATH = {
    "vgg16": DATASET_ROOT / "pretrained/vgg16-397923af.pth",
    "resnet50": DATASET_ROOT / "pretrained/resnet50-19c8e357.pth",
    "vgg16p2m": DATASET_ROOT / "pretrained/vgg16-p2m.pth",
}

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
