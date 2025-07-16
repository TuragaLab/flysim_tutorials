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

## Paperspace Gradient setup (Ubuntu 22.04 runtime, the default as of July 2025):

Open `tutorial.ipynb` [in
Gradient](https://console.paperspace.com/github/TuragaLab/flysim_tutorials?clone=true)
and then run the following in the terminal:

```shell
> pip uninstall -y tensorboard tensorflow wandb && \
  pip install h5py matplotlib mujoco==3.3.3 numpy onnx onnxruntime pillow rich scipy
```

## Colab setup (Ubuntu 22.04 runtime, the default as of July 2025):

Open `tutorial.ipynb` [in
Colab](https://colab.research.google.com/github/TuragaLab/flysim_tutorials/blob/main/body_tutorial/tutorial.ipynb),
and then run the following in the terminal, or in a notebook cell with a
preceeding "!":

```shell
> pip install h5py matplotlib mujoco==3.3.3 numpy onnx onnxruntime pillow rich scipy && \
  git clone https://github.com/TuragaLab/flysim_tutorials.git /tmp/tutorial_repo && \
  mv /tmp/tutorial_repo/body_tutorial/projectlib . && \
  rm -rf /tmp/tutorial_repo
```

## Project ideas

- **State space mapping:** How complicated of a behavior is six-legged walking
  on a flat plane? One might imagine that the positions of some joints and/or
  the measurements collected from some sensors are highly correlated. How many
  dimensions are needed to faithfully characterize the fly's instantaneous
  position at some point in a straight walking trajectory on a flat plane? What
  about if the fly is moving in a winding trajectory, or walking over obstacles?

- **Controller distillation:** Is it possible to make the walking controller
  simpler and easier to understand while approximately preserving its function?
  If you train simpler networks to mimic the controller's behavior, how does the
  control quality fall off as the network size decreases? In addition to
  simplifying the controller internally, it might also be possible to simplify
  or ignore some of its inputs.

- **Controller stress testing:** Mujoco allows us to simulate flies in arbitrary
  environments. In which environments does the walking controller work well, and
  in which environments does it fail? (_e.g._, how uneven can the terrain be
  before the fly falls over?) You can consider other deviations from the
  training conditions as well. For example, many animals gradually change size
  over the course of their life, and maintain their ability to walk as they
  grow. How robust is the walking controller to changes in the size of some or
  all of the fly's body parts?

- **Controller training:** The imitation learning procedure used to optimize the
  walking controller takes about a week to run, so we don't suggest trying to
  replicate it exactly. But it could still be interesting to see whether it's
  possible to approximate it (or improve upon it) using a simpler training
  objective. _e.g._, optimizing for some combination of walking speed, energy
  consumption, head stability, joint stress, and/or trajectory smoothness.
