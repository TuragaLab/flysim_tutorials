from itertools import product
from math import ceil
from os import PathLike
from pathlib import Path
from shutil import move, rmtree, unpack_archive
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Callable, Final
from urllib.request import urlretrieve
from weakref import WeakKeyDictionary

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "VideoWriter",
    "add_keypoint_sites",
    "add_target_position_sites",
    "caption",
    "download_body_model_if_missing",
    "download_controller_if_missing",
    "download_pose_dataset_if_missing",
    "render",
]


_renderers = WeakKeyDictionary[object, mj.Renderer]()


class VideoWriter:
    def __init__(
        self,
        path: str | PathLike[str],
        duration: float,
        play_speed: float = 1.0,
        framerate: float = 25.0,
        verbose: bool = True,
    ) -> None:
        self.path: Final = Path(path)
        self.duration = duration
        self.play_speed: Final = play_speed
        self.framerate: Final = framerate
        self.verbose: Final = verbose
        self._n_frames_written = 0

    def is_writing(self) -> bool:
        return self._next_frame_time() < self.duration

    def send(self, timestamp: float, render_fn: Callable[[], np.ndarray]) -> None:
        if not self.is_writing():
            return

        frame_duration = self.play_speed / self.framerate
        next_frame_time = self._n_frames_written * frame_duration

        if timestamp >= next_frame_time:
            self._write_frame(render_fn())

        if not self.is_writing():
            self._encode_video_and_delete_frames()

    def _frame_dir(self) -> Path:
        return self.path.with_name(f"{self.path.stem}-frames")

    def _next_frame_time(self) -> float:
        return self._n_frames_written * self.play_speed / self.framerate

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._n_frames_written == 0:
            rmtree(self._frame_dir(), ignore_errors=True)
            self._frame_dir().mkdir(parents=True, exist_ok=True)

        if self.verbose:
            total_n_frames = ceil(self.duration / self.play_speed * self.framerate)
            progress_desc = f"{self._n_frames_written + 1}/{total_n_frames}"
            print(f"\rGenerating frames... ({progress_desc})", end="", flush=True)

        frame_path = self._frame_dir() / f"{self._n_frames_written:06d}.png"
        Image.fromarray(frame).save(frame_path)
        self._n_frames_written += 1

    def _encode_video_and_delete_frames(self) -> None:
        if self.verbose:
            print("\nEncoding video...")

        run([
            "ffmpeg",
            *("-y", "-hide_banner", "-loglevel", "error"),
            *("-framerate", str(self.framerate)),
            *("-i", self._frame_dir() / r"%06d.png"),
            *("-pix_fmt", "yuv420p"),
            self.path,
        ])
        rmtree(self._frame_dir())


def download_body_model_if_missing() -> None:
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
    output_path = Path("_inbox/fly-grooming-poses.pkl")
    archive_url = "https://janelia.figshare.com/ndownloader/files/52159823"
    file_name_in_archive = "fly-grooming-poses.pkl"

    if not output_path.exists():
        with TemporaryDirectory() as temp:
            urlretrieve(archive_url, Path(temp, "archive.zip"))
            unpack_archive(Path(temp, "archive.zip"), temp)
            move(Path(temp, file_name_in_archive), output_path)


def download_controller_if_missing() -> None:
    output_path = Path("_inbox/controller.onnx")
    file_id = "1W3xYdbPoARkV27EQCLIKV19sl7IiN18g"
    url = f"https://drive.usercontent.google.com/u/0/uc?id={file_id}&export=download"

    if not output_path.exists():
        urlretrieve(url, output_path)


def render(
    model: object,
    sim_state: object,
    camera_name: str,
    height: int = 480,
    width: int = 640,
) -> np.ndarray:
    camera_query: int = mj.mjtObj.mjOBJ_CAMERA  # type: ignore
    camera_id: int = mj.mj_name2id(model, camera_query, camera_name)  # type: ignore
    renderer = _renderers.get(model, None)

    if renderer is None or renderer.height != height or renderer.width != width:
        renderer = mj.Renderer(model, height, width)
        _renderers[model] = renderer

    renderer.update_scene(sim_state, camera_id)
    return renderer.render()


def caption(image: np.ndarray, text: str) -> np.ndarray:
    pil_image = Image.fromarray(image)
    font = ImageFont.load_default(size=16)
    draw_obj = ImageDraw.Draw(pil_image)
    draw_obj.text((0, 0), text, "white", font)
    return np.array(pil_image)


def add_keypoint_sites(spec: object) -> None:
    """
    Add sites corresponding to tracked keypoints to a model.

    The sites are named "site_0", "site_1", ..., "site_35". Sites 0 through 29
    are on the legs, and sites 30 through 35 are on the body.
    """
    leg_names = ["T1_left", "T2_left", "T3_left", "T3_right", "T2_right", "T1_right"]
    leg_part_names = ["coxa", "femur", "tibia", "tarsus", "claw"]

    body_site_specs = [
        ("head", "site_30", [0.0, 0.04375, 0.00875]),  # Head
        ("abdomen_7", "site_31", [0.0, 0.039375, -0.004375]),  # Tail
        ("thorax", "site_32", [-0.0455, 0.02625, -0.00875]),  # Left haltere
        ("thorax", "site_33", [-0.0455, -0.02625, -0.00875]),  # Right haltere
        ("head", "site_34", [-0.04375, 0.016625, 0.0035]),  # Left eye
        ("head", "site_35", [0.04375, 0.016625, 0.0035]),  # Right eye
    ]

    size = (0.006, 0.006, 0.006)
    color = (0, 1, 0, 0.8)

    for i, (leg_name, part_name) in enumerate(product(leg_names, leg_part_names)):
        body = spec.body(f"{part_name}_{leg_name}")  # type: ignore
        part_is_claw = part_name == "claw"
        pos = spec.site(f"claw_{leg_name}").fromto[-3:] if part_is_claw else [0, 0, 0]  # type: ignore
        body.add_site(name=f"site_{i}", pos=pos, size=size, rgba=color, group=0)

    for body_name, site_name, pos in body_site_specs:
        body = spec.body(body_name)  # type: ignore
        body.add_site(name=site_name, pos=pos, size=size, rgba=color, group=0)


def add_target_position_sites(spec: object, target_positions: np.ndarray) -> None:
    """ """
    for pos in target_positions:
        size = (0.006, 0.006, 0.006)
        color = (1.0, 0.0, 0.0, 0.8)
        spec.worldbody.add_site(pos=pos, size=size, rgba=color)  # type: ignore
