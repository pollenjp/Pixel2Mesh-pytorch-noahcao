# Third Party Library
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize

# First Party Library
from p2m import config


class BaseDataset(Dataset):
    def __init__(self):
        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
