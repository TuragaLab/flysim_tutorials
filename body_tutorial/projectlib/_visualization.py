"""
Definitions that can be used to generate images and videos from Mujoco
simulations.
"""

from math import ceil
from os import PathLike
from pathlib import Path
from shutil import rmtree
from subprocess import run
from typing import Callable, Final
from weakref import WeakKeyDictionary, proxy

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mujoco import MjData, MjModel  # type: ignore

__all__ = ["VideoWriter", "caption", "render"]


_renderers = WeakKeyDictionary[object, mj.Renderer]()


class VideoWriter:
    """
    An object that writes a single video file.
    """

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
        """
        Return `True` if the video writer is currently writing a video, and
        `False` otherwise.
        """
        return self._next_frame_time() < self.duration

    def send(self, timestamp: float, render_fn: Callable[[], np.ndarray]) -> None:
        """
        Send a (timestamp, render function) pair to the video writer.

        The video writer may call the render function and store the result if it
        needs to generate a frame. If the writer has generated its last frame,
        it will convert the stored frames to a video file.
        """
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


def render(
    model: MjModel,
    sim_state: MjData,
    camera_name: str,
    height: int = 480,
    width: int = 640,
) -> np.ndarray:
    """
    Render a snapshot from a Mujoco simulation.

    The result will be a `height` x `width` x 3 array.
    """
    camera_query: int = mj.mjtObj.mjOBJ_CAMERA  # type: ignore
    camera_id: int = mj.mj_name2id(model, camera_query, camera_name)  # type: ignore
    renderer = _renderers.get(model, None)

    if renderer is None or renderer.height != height or renderer.width != width:
        renderer = mj.Renderer(proxy(model), height, width)
        _renderers[model] = renderer

    renderer.update_scene(sim_state, camera_id)
    return renderer.render()


def caption(image: np.ndarray, text: str) -> np.ndarray:
    """
    Return a version of a given image with an added caption.
    """
    pil_image = Image.fromarray(image)
    font = ImageFont.load_default(size=16)
    draw_obj = ImageDraw.Draw(pil_image)
    draw_obj.text((0, 0), text, "white", font)
    return np.array(pil_image)
