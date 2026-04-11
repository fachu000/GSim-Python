# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GSim-Python is a simulation/experimentation framework designed to be used as a **git submodule** in research projects. It provides:
- Numbered experiment management with automatic figure persistence (no re-running to re-plot)
- `GFigure`: a MATLAB-style plotting abstraction over matplotlib
- `NeuralNet`: a PyTorch pipeline wrapper with normalizers and LR schedulers

## Commands

```bash
# Run tests (from this directory)
python -m pytest tests

# Run a single test file
python -m pytest tests/test_normalizers.py -v

# Run a single test
python -m pytest tests/test_normalizers.py::TestStdFeatNormalizer::test_fit -v
```

The repo has no build step or linter configured. Tests use pytest ≥ 7.0 with `-ra -q` defaults.

When installed as a submodule in a parent repo, experiments are run from the parent root:

```bash
python run_experiment.py <experiment_id>          # run and store figures
python run_experiment.py -p <experiment_id>       # plot stored figures only
python run_experiment.py -e <experiment_id>       # export PDF
python run_experiment.py -x <module> <id>         # select experiment module
python run_experiment.py -g <gpu_id> <id>         # select GPU
```

## Architecture

### Core Flow

1. User defines an `ExperimentSet` class inheriting from `gsim.AbstractExperimentSet`
2. Each experiment is a method named `experiment_<id>(l_args)` that returns `None`, a `GFigure`, or a list of `GFigure`s
3. `run_experiment.py` (in parent repo root) dispatches to the correct module and experiment ID
4. Results are pickled to `./output/{module_name}/experiment_{id}.pk`; `-p` replots without re-running

### Key Modules

**`experiment_set.py` — `AbstractExperimentSet`**
- `run_experiment(id, l_args, save_pdf, inspect)` — executes the `experiment_<id>` method, stores figures
- `plot_only(id, save_pdf, inspect)` — loads stored figures and replots
- `load_GFigures(id)` — public API to load another experiment's results from within an experiment

**`gfigure.py` — `GFigure`, `Subplot`, `Curve`**  
The largest file (≈1800 lines). `GFigure` is a list of `Subplot`s; each `Subplot` is a list of `Curve`s. Supports 2D/3D plots, shaded confidence intervals, histograms, PDF export, MATLAB-style style strings. All plotting code in experiments should use this abstraction.

**`include/neural_net/neural_net.py` — `NeuralNet`**  
PyTorch pipeline wrapper. Handles device management, history tracking, LR scheduling, live training plots, and save/load. Takes a `torch.utils.data.Dataset`.

**`include/neural_net/normalizers.py`**  
Normalizer hierarchy: `StdFeatNormalizer`, `IntervalFeatNormalizer`, `ScaleToUnitPowerFeatNormalizer`, `MultiFeatNormalizer`. All support incremental `.fit()` and `.normalize()`/`.unnormalize()`. Used to pre/post-process neural net inputs and outputs.

**`include/neural_net/lr_schedulers.py` — `WarmupCosineMinLRScheduler`**  
Custom LR scheduler: linear warmup then cosine decay to a minimum LR.

**`utils.py`**  
`instr()` for inspecting nested tensor/array structures, timer helpers, `xor()`.

### Installation (in a parent repo)

```bash
git submodule add https://github.com/fachu000/GSim-Python.git ./gsim
bash gsim/install.sh   # creates run_experiment.py, gsim_conf.py, experiments/ in parent root
```

`gsim_conf.py` in the parent root selects which experiment modules are active and configures logging.

### Test Layout

- `tests/test_utils.py` — `xor`, time formatting, timers
- `tests/test_normalizers.py` — all normalizer classes including incremental fit and save/load
- `tests/test_neuralnet.py` — `NeuralNet` training, data loading, LR schedulers, inference
- `tests/conftest.py` — adds parent directory to `sys.path` so `import gsim` works from `tests/`
