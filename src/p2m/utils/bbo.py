# Standard Library
import re
from logging import NullHandler
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class NoStdoutException(Exception):
    pass


class NotFoundError(Exception):
    pass


class ScoreSender:
    score_path_line_prefix = r"Best score is saved in "
    score_filename = "score.txt"

    @classmethod
    def save_score(cls, dirpath: Path, score: float) -> None:
        filepath = dirpath / cls.score_filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wt") as f:
            f.write(f"{score}")
        print(f"{cls.score_path_line_prefix}{filepath.absolute()}")

    @classmethod
    def extract_path_from_stdout(cls, stdout: bytes) -> Path:

        key_name = "path"
        pattern = re.compile(
            "{}{}{}{}{}".format(
                r"^",
                cls.score_path_line_prefix,
                r"(?P<",
                key_name,
                r">.*)$",
            )
        )

        for _, line in enumerate(bytes.decode(stdout).split("\n")):
            line = line.strip()
            if (m := pattern.match(line)) is not None:
                return Path(m.group(f"{key_name}"))

        raise NotFoundError("Target string does not found in stdout")

    @classmethod
    def extract_score(cls, filepath: Path) -> float:
        with open(filepath, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    return float(line)

        raise NotFoundError(f"Score in {filepath}")
