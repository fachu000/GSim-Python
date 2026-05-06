

All experiments are dispatched through `run_experiment.py` (must be invoked from the repo root — it asserts `./gsim` exists). By default, `run_experiment.py` looks for the experiment module specified in `gsim_conf.py` (e.g. `experiments.example_experiments`) but this can be overridden via the `-x` flag. 

For example, to run experiment 1001 in `experiments/example_experiments.py`, run
```bash
python run_experiment.py -x example_experiments 1001
```
If `gsim_conf.py` already specifies `module_name = "experiments.example_experiments"`, then one can simply run
```bash
python run_experiment.py 1001
```

The options are listed below. They can be combined. 

```bash
python run_experiment.py <experiment_id>          # run and store figures
python run_experiment.py -p <experiment_id>       # plot stored figures only
python run_experiment.py -e <experiment_id>       # export PDF
python run_experiment.py -n <experiment_id>       # run without plotting (useful for agents)
python run_experiment.py -x <module> <id>         # select experiment module (overrides gsim_conf.py)
python run_experiment.py -g <gpu_id> <id>         # select GPU
```

`module_name` resolution: `run_experiment.py` first looks for `<module>.py` relative to CWD, then under `experiments/`. From an IPython shell: `from run_experiment import run; run(<id>)`.


For agents, the most useful flags are `-n` (no plotting) and `-x` (to specify the experiment module, e.g. `-x path_planning_experiments`).
```bash
python run_experiment.py -x <module> -n <experiment_id> # same but overriding the module specified in gsim_conf.py
```