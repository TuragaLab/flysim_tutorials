from math import floor
from os import PathLike
from pathlib import Path
from shutil import move, rmtree
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Callable, Final, Iterable, Self
from weakref import WeakKeyDictionary

import mujoco as mj
import numpy as np
from PIL import Image

__all__ = ["VideoWriter", "advance", "download_body_model_if_missing", "render"]


_renderers = WeakKeyDictionary[object, mj.Renderer]()


# class VideoWriter:
#     def __init__(
#         self,
#         path: str | PathLike[str],
#         n_frames: int,
#         framerate: float = 25.0,
#         verbose: bool = True,
#     ) -> None:
#         self.path: Final = Path(path)
#         self.n_frames: Final = n_frames
#         self.framerate: Final = framerate
#         self.verbose: Final = verbose
#         self._n_frames_written = 0

#     def frame_times(self) -> Iterable[float]:
#         for i in range(self.n_frames):
#             yield i / self.framerate

#     def is_writing(self) -> bool:
#         return self._n_frames_written < self.n_frames

#     def write(self, frame: np.ndarray) -> None:
#         frame_dir = self.path.with_name(f"{self.path.stem}-frames")
#         frame_path = frame_dir / f"{self._n_frames_written:06d}.png"

#         if self._n_frames_written == 0:
#             rmtree(frame_dir, ignore_errors=True)
#             frame_dir.mkdir(parents=True, exist_ok=True)

#         if self.verbose:
#             progress_desc = f"{self._n_frames_written + 1}/{self.n_frames}"
#             print(f"\rGenerating frames... ({progress_desc})", end="", flush=True)

#         Image.fromarray(frame).save(frame_path)
#         self._n_frames_written += 1

#         if self._n_frames_written == self.n_frames:
#             if self.verbose:
#                 print("\nEncoding video...")
#             run([
#                 "ffmpeg",
#                 *("-y", "-hide_banner", "-loglevel", "error"),
#                 *("-framerate", str(self.framerate)),
#                 *("-i", frame_dir / r"%06d.png"),
#                 *("-pix_fmt", "yuv420p"),
#                 self.path,
#             ])
#             rmtree(frame_dir)


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
            raise ValueError(f"`{self.path}` has already been written")

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
            total_n_frames = floor(self.duration / self.play_speed * self.framerate)
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


def advance(model: object, sim_state: object, t_min: float) -> None:
    if sim_state.time >= t_min:  # type: ignore
        mj.mj_forward(model, sim_state)  # type: ignore
    else:
        while sim_state.time < t_min:  # type: ignore
            mj.mj_step(model, sim_state)  # type: ignore


def render(
    model: object,
    sim_state: object,
    camera_name: str,
    height: int,
    width: int,
) -> np.ndarray:
    camera_query: int = mj.mjtObj.mjOBJ_CAMERA  # type: ignore
    camera_id: int = mj.mj_name2id(model, camera_query, camera_name)  # type: ignore
    renderer = _renderers.get(model, None)

    if renderer is None or renderer.height != height or renderer.width != width:
        renderer = mj.Renderer(model, height, width)
        _renderers[model] = renderer

    renderer.update_scene(sim_state, camera_id)
    return renderer.render()
