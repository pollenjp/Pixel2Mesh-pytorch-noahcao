# Standard Library
import logging
import os
import re
import shutil
from logging import getLogger
from pathlib import Path
from typing import Iterator
from typing import TypeVar

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
    level=logging.WARNING,
)
logger = getLogger(__name__)
logger.setLevel(logging.INFO)

_PathLike = TypeVar("_PathLike", Path, str)


def get_search_file_iterator(dirpath: _PathLike, pattern) -> Iterator[Path]:
    count: int = 0
    for _dirpath, _dirnames, _filenames in os.walk(str(dirpath)):
        for _filename in _filenames:
            if count % 1000 == 0:
                logger.info(f"count: {count}, {_filename}, {pattern.match(string=_filename)}")
            count += 1

            if pattern.match(string=_filename) is not None:
                yield Path(_dirpath) / _filename

def myfunc():
    pass

def myfunc_wrapper(args_dict: Dict[Any, Any]):
    return myfunc(**args_dict)

class TmpIterator:
    def __init__(self, dirpath: _PathLike, pattern):
        self.dirpath: _PathLike = dirpath
        self.pattern = pattern

    def __iter__(self):
        for data_pair_idx, data_pair in enumerate(self.data_list):
            yield dict(
                data_pair=data_pair,
                model_input_image_size_hw=model_input_image_size_hw,
                out_dir_path=out_dir_path,
                semaseg_id_to_bbox_id=semaseg_id_to_bbox_id,
                is_train=is_train,
                area_threashold=0,
                debug_index=data_pair_idx,
            )
    def get_search_file_iterator() -> Iterator[Path]:
        count: int = 0
        for _dirpath, _dirnames, _filenames in os.walk(str(self.dirpath)):
            for _filename in _filenames:
                if count % 1000 == 0:
                    logger.info(f"count: {count}, {_filename}, {self.pattern.match(string=_filename)}")
                count += 1

                if self.pattern.match(string=_filename) is not None:
                    yield Path(_dirpath) / _filename

# pool: multiprocessing.Pool
with multiprocessing.Pool(num_workers) as pool:
    pool.map(func=myfunc_wrapper, iterable=TmpIterator(data_list))

def main():
    input_root_dirpath: Path = Path("datasets/data/shapenet/data_tf")
    output_root_dirpath: Path = Path("datasets/data/shapenet/data_tf_2021-12-17")
    pattern = re.compile(
        pattern=r"rendering_metadata.txt",
        # flags=re.IGNORECASE,
    )
    for filepath in get_search_file_iterator(dirpath=input_root_dirpath, pattern=pattern):
        relative_path = filepath.relative_to(input_root_dirpath)
        save_filepath: Path = output_root_dirpath / relative_path.parents[1] / relative_path.name
        assert save_filepath.parent.exists(), save_filepath
        shutil.copyfile(filepath, save_filepath)

if __name__ == "__main__":
    main()
