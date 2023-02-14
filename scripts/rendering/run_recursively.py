# Standard Library
import argparse
import concurrent.futures
import subprocess
import typing as t
from dataclasses import dataclass
from logging import NullHandler
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--out_dir", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--num_workers", default=2, type=int)
    args = parser.parse_args()
    return args


name2index = {
    # "category_id/object_id/num": (1, 2),
    #
    # "02691156/d3b9114df1d8a3388e415c6cf89025f0/00": (1, 0),
    # "02691156/d3b9114df1d8a3388e415c6cf89025f0/02": (3, 1),
    # "02691156/d4dac019726e980e203936772104a82d/02": (3, 3),
    # "02691156/d54ca25127a15d2b937ae00fead8910d/00": (1, 0),
    # "02691156/d59d75f52ac9b241ae0d772a1c85134a/02": (5, 2),
    # "02691156/d63daa9d1fd2ff5d575bf8a4b14be4f4/03": (2, 1),
    # "02691156/d605a53c0917acada80799ffaf21ea7d/00": (3, 2),
    #
    "02691156/d3dcf83f03c7ad2bbc0909d98a1ff2b4/00": (3, 2),
    "02691156/d4aec2680be68c813a116bc3efac4e3b/02": (2, 2),
    "02691156/d6bf9fb6149faabe36f3a2343a8221f2/04": (2, 3),
}


def search_file_iter(dir: Path) -> t.Iterator[Path]:
    for p in dir.glob("**/*_2.obj"):
        yield p
        # category_id: str = p.parents[2].name
        # object_id: str = p.parents[1].name
        # i_label: str = p.parents[0].name
        # if f"{category_id}/{object_id}/{i_label}" in name2index:
        #     yield p


@dataclass
class Cmd:
    cmd: t.List[str]
    env: t.Optional[t.Dict[str, str]] = None
    stdout: t.Optional[t.Union[t.TextIO, int]] = None
    stderr: t.Optional[t.Union[t.TextIO, int]] = None
    debug_msg: t.Optional[str] = None


def run_cmd(cmd: Cmd) -> None:
    subprocess.run(cmd.cmd, stdout=cmd.stdout, stderr=cmd.stderr, env=cmd.env)


def main() -> None:
    args = get_args()

    blender_cmd: Path = (Path(__file__).parent / ".local/blender/blender").resolve()
    assert blender_cmd.exists(), f"{blender_cmd} does not exists"

    py_file: Path = (Path(__file__).parent / "main.py").resolve()
    assert py_file.exists(), f"{py_file} does not exists"

    output_base_dir: Path = args.out_dir
    output_base_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_fpath: t.Dict[concurrent.futures.Future[None], Cmd] = {}
        cmd: Cmd
        for i, filepath in enumerate(search_file_iter(args.data_dir)):
            logger.info(f"{i:>5}: {filepath}")
            if not filepath.exists():
                logger.error(f"{filepath} is not exists")
                continue

            output_file_parent: Path = output_base_dir / filepath.parent.relative_to(args.data_dir)
            output_file_parent.mkdir(parents=True, exist_ok=True)
            cmd = Cmd(
                cmd=[
                    str(blender_cmd),
                    "--background",
                    "--python",
                    f"{py_file}",
                    "--",
                    f"obj_filepath='{filepath}'",
                    f"output_dirpath='{output_file_parent}'",
                    f"output_name='rendering_{filepath.stem}'",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                debug_msg=f"{filepath}",
            )
            future_to_fpath[executor.submit(run_cmd, cmd)] = cmd

        future: concurrent.futures.Future[None]
        for future in concurrent.futures.as_completed(future_to_fpath):
            cmd = future_to_fpath[future]
            try:
                future.result()
            except Exception as exc:
                raise exc
            else:
                logger.info(f"Completed: {cmd.debug_msg}")


if __name__ == "__main__":
    # Standard Library
    import logging

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
        level=logging.INFO,
    )

    main()
