# Fruit fly body model tutorial

## Local setup (Linux):

Clone this repository, install [Pixi](https://pixi.sh/latest/installation/) if
it isn't installed already, and then run

```shell
> pixi shell
```

to start a shell with the necessary dependencies installed.

## Colab setup (Ubuntu 22.04 runtime, the default as of June 2025):

Open `main.ipynb` in Colab, and then run the following in the terminal, or in a
notebook cell with a preceeding "!":

```shell
> apt update && \
  apt install -y ffmpeg git && \
  pip install h5py matplotlib mujoco numpy pillow && \
  git clone https://github.com/TuragaLab/flysim_tutorials.git /tmp/tutorial_repository && \
  mv /tmp/tutorial_repository/body_tutorial/projectlib . && \
  rm -rf /tmp/tutorial
```
