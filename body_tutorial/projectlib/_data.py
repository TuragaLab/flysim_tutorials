"""
Definitions related to downloading data.
"""

from pathlib import Path
from shutil import move, unpack_archive
from subprocess import run
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

__all__ = [
    "download_body_model_if_missing",
    "download_controller_if_missing",
    "download_pose_dataset_if_missing",
]


def download_body_model_if_missing() -> None:
    """
    Download the body model to `_inbox/flybody`, if it isn't there already.
    """
    output_path = Path("_inbox/flybody")
    repo_url = "https://github.com/TuragaLab/flybody.git"
    commit_hash = "e1a6135c310c39291f4fb68d682f2fd0b05e0555"
    subdir = "flybody/fruitfly/assets"

    if not output_path.exists():
        with TemporaryDirectory() as repo:
            run(["git", "clone", repo_url, repo])
            run(["git", "-C", repo, "checkout", commit_hash, "--quiet"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            move(Path(repo) / subdir, output_path)


def download_pose_dataset_if_missing() -> None:
    """
    Download the pose dataset to `_inbox/fly-grooming-poses.pkl`, if it isn't
    there already.
    """
    output_path = Path("_inbox/fly-grooming-poses.pkl")
    archive_url = "https://janelia.figshare.com/ndownloader/files/52159823"
    file_name_in_archive = "fly-grooming-poses.pkl"

    if not output_path.exists():
        with TemporaryDirectory() as temp:
            urlretrieve(archive_url, Path(temp, "archive.zip"))
            unpack_archive(Path(temp, "archive.zip"), temp)
            move(Path(temp, file_name_in_archive), output_path)


def download_controller_if_missing() -> None:
    """
    Download the walking controller to `_inbox/controller.onnx`, if it isn't
    there already.
    """
    output_path = Path("_inbox/controller.onnx")
    file_id = "1ydFwmy_4wFyQ00SoZxPZUMvLRiCnCFSk"
    url = f"https://drive.usercontent.google.com/u/0/uc?id={file_id}&export=download"

    if not output_path.exists():
        urlretrieve(url, output_path)
