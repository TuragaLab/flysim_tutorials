# Fruit fly body model tutorial

## Local setup (Linux):

1. Clone this repository.
2. Install the [Pixi](https://pixi.sh/latest/installation/) package manager.
3. Run `pixi install` in the `body_tutorial` directory.
4. Point your notebook editor of choice to the Python interpreter installed in
   `.pixi/envs/default/bin` and open `tutorial.ipynb`.

If you don't have a notebook editor installed, you can run

```shell
> pixi run jupyter notebook tutorial.ipynb
```

to open the tutorial in the Jupyter editor (in which case you can actually skip
step 3, since `pixi run` will install dependencies as needed).

## Colab setup (Ubuntu 22.04 runtime, the default as of June 2025):

Open `tutorial.ipynb` in Colab, and then run the following in the terminal, or
in a notebook cell with a preceeding "!":

```shell
> apt update && \
  apt install -y ffmpeg git && \
  pip install h5py matplotlib mujoco==3.3.3 numpy onnx onnxruntime pillow scipy && \
  git clone https://github.com/TuragaLab/flysim_tutorials.git /tmp/tutorial_repo && \
  mv /tmp/tutorial_repo/body_tutorial/projectlib . && \
  rm -rf /tmp/tutorial_repo
```
