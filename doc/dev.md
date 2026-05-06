
This document describes the development workflow for the GSim-Python submodule itself.


## Running GSim-Python tests
The submodule has its own tests that are run separately from the project-level
tests. To run tests:

```bash
cd gsim                           # if not already in the gsim/ folder
python -m pytest tests            # or a single file: python -m pytest tests/test_normalizers.py -v
```

## Coding guidelines

Please see `coding_guidelines.md` for instructions. 