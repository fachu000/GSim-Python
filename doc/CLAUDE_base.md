This file provides coding guidance when working with code in this repository. The first part is common to all GSim-based repositories. Domain-specific guidelines are found below. 

# GSim-based research codebase

This repository contains a research codebase based on the GSim-Python submodule. The code is organized into experiment modules in the `experiments/` folder. Each experiment module defines a class `ExperimentSet` with methods named `experiment_<id>(l_args)`, where `<id>` is an integer identifier for the experiment. These methods return `GFigure` objects that are pickled for later plotting and analysis. 

Research codebase for simulating **aerial base station** placement, channel-gain map estimation, path planning, and related signal-processing tasks. Runs on top of **GSim-Python** (the `gsim/` submodule), which provides the experiment harness, `GFigure` plotting, and `NeuralNet` PyTorch wrapper. See `gsim/CLAUDE.md` for harness-level detail.

Agents can run experiments as follows:
```bash
source .venv/bin/activate  # Ask the user to create a link from `.venv` to their Python environment if not done already
python run_experiment.py -x <module> -n <experiment_id> # overrides the module specified in gsim_conf.py
```
- `<module>` is the full dotted path including subfolder and suffix, e.g. `Experiments.fundamentals_part_2_experiments` (not just `fundamentals_part_2`).
- `-n` skips `plt.show()`, so it never blocks in a headless shell. Combine it with `-e` (`-n -e <experiment_id>`) to also export a PDF to `output/<module>/experiment_<id>.pdf` without displaying it. `-n` alone (no `-e`) skips plotting entirely — use that just to confirm an experiment runs without errors.
- To inspect a figure, read the exported PDF directly with the Read tool — matplotlib's PDF backend embeds vector text, so the Read tool renders the figure and extracts axis/legend text without needing a PNG conversion step.

More details on running instructions can be found in `gsim/doc/running_instructions.md`.

For coding guidelines, see `gsim/doc/coding_guidelines.md`. 

For developing the GSim-Python submodule itself, see `gsim/CLAUDE.md` and `gsim/doc/dev.md`. 

# PR Reviews

When reviewing a pull request, please verify that:

- The functionality is as expected by the name, docstring, and place of the function/class in the codebase.  

- The rules in `coding_guidelines.md` are followed.

PRs are mostly created by students, so please provide feedback in a way that helps them learn. For example, you can explain why it is important to follow a certain rule.  