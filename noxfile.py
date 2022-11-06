# Standard Library
import uuid
from pathlib import Path
from typing import Dict
from typing import List

# Third Party Library
import nox
from nox.sessions import Session

src_dir: Path = Path(__file__).parent / "src"
python_code_path_list: List[str] = [
    f"{src_dir}",
    "noxfile.py",
]
assert all(isinstance(path, str) for path in python_code_path_list)
env_common: Dict[str, str] = {
    "PYTHONPATH": f"{src_dir}",
}
nox_tmp_dir: Path = Path(__file__).parent / ".nox_tmp"
python_version_list: List[str] = ["3.7"]


def install_package(session: Session, dev: bool = False) -> None:
    session.install("--upgrade", "pip")
    session.run("pip", "-V")
    requirements_txt_path: Path = nox_tmp_dir / f"poetry-requirements-{str(uuid.uuid4())}.txt"
    requirements_txt_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        cmd = [
            "poetry",
            "export",
            "--format",
            "requirements.txt",
            "--output",
            f"{requirements_txt_path}",
        ] + (["--with", "dev"] if dev else [])
        session.run(*cmd, external=True)
        session.install("-r", f"{requirements_txt_path}")
    except Exception as e:
        raise e
    else:
        if requirements_txt_path.exists():
            requirements_txt_path.unlink()


@nox.session(python=python_version_list)
def format(session: Session) -> None:
    env: Dict[str, str] = {}
    env.update(env_common)
    kwargs = dict(env=env, success_codes=[0, 1])

    install_package(session, dev=True)
    session.run(
        "autoflake8",
        "--in-place",
        "--recursive",
        "--remove-unused-variables",
        "--in-place",
        "--exit-zero-even-if-changed",
        *python_code_path_list,
        **kwargs,
    )
    session.run("isort", *python_code_path_list, **kwargs)
    session.run("black", *python_code_path_list, **kwargs)


@nox.session(python=python_version_list)
def lint(session: Session) -> None:
    env: Dict[str, str] = {}
    env.update(env_common)
    kwargs = dict(env=env)

    install_package(session, dev=True)
    session.run("flake8", "--statistics", "--count", "--show-source", *python_code_path_list, **kwargs)
    session.run("autoflake8", "--check", "--recursive", "--remove-unused-variables", *python_code_path_list, **kwargs)
    session.run("isort", "--check", *python_code_path_list, **kwargs)
    session.run("black", "--check", *python_code_path_list, **kwargs)
    session.run("mypy", "--check", "--no-incremental", *python_code_path_list, **kwargs)


@nox.session(python=python_version_list)
def test(session: Session) -> None:
    env: Dict[str, str] = {}
    env.update(env_common)
    kwargs = dict(env=env)

    install_package(session, dev=True)
    session.run("pytest", **kwargs)
