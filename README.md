# Welcome to flysim tutorials

This repository contains tutorials and ideas for working with models of the fly.

Currently covering

```
├── notebooks
│   ├── 01_flyvis_tutorial_deep_stimulus_design.ipynb
│   ├── 02_flyvis_tutorial_mechanism_discovery.ipynb
│   ├── 03_flyvis_project_optimal_deep_stimulus_design.ipynb
│   ├── 04_flyvis_project_deep_illusion_design.ipynb
│   ├── 05_flyvis_project_deep_mechanistic_interpretation.ipynb
│   └── 06_flyvis_project_deep_mechanistic_auxiliary_targets.ipynb
├── LICENSE
├── pyproject.toml
└── README.md
```

# Install requirements via

## Local

```
uv venv
uv pip install -e .  
source .venv/bin/activate
flyvis download-pretrained
```

## From colab

```
pip install git+https://github.com/TuragaLab/flysim_tutorials.git
flyvis download-pretrained
```

# Refs

- [flyvis](https://github.com/turagalab/flyvis)