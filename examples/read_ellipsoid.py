# Standard Library
import logging
from logging import getLogger
from pathlib import Path
from typing import List

# First Party Library
from p2m.utils.mesh import Ellipsoid

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
    level=logging.WARNING,
)
logger = getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    output_filepath: Path = Path("ellipsoid.obj")

    mesh_pos: List[float] = [-0.0, -0.0, -0.8]
    ellipsoid: Ellipsoid = Ellipsoid(mesh_pos=mesh_pos)

    logger.info(f"coord         =\n{ellipsoid.coord}")
    logger.info(f"edges         =\n{ellipsoid.edges}")
    logger.info(f"laplace_idx   =\n{ellipsoid.laplace_idx}")
    logger.info(f"unpool_idx    =\n{ellipsoid.unpool_idx}")
    logger.info(f"adj_mat       =\n{ellipsoid.adj_mat}")
    # logger.info(f"obj_fmt_faces =\n{ellipsoid.obj_fmt_faces}")
    logger.info(f"faces         =\n{ellipsoid.faces}")

    with open(output_filepath, mode="wt") as f:
        for coord in ellipsoid.coord:
            f.write(f"v {coord[0]} {coord[1]} {coord[2]}\n")

        ellipsoid.faces[0] = ellipsoid.faces[0] + 1  # convert 0 origin to 1 origin
        for face in ellipsoid.faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


if __name__ == "__main__":
    main()
