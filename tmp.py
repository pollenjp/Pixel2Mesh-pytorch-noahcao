# Standard Library
import logging
from logging import getLogger
from pathlib import Path

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
    level=logging.WARNING,
)
logger = getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Third Party Library
    import torch

    # First Party Library
    from p2m.datasets.shapenet_with_template import extract_coords_from_obj_file

    fpath = Path("/home/hiroakisugisaki/workdir/github/pollenjp/pixel2mesh-pytorch-noahcao/datasets/data/shapenet/data_with_template/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/00_depth0001.obj")

    coords = torch.tensor(extract_coords_from_obj_file(fpath))
    logger.info(coords)
