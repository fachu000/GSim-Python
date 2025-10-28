"""
Pytest configuration for gsim tests.

This file adds the gsim parent directory to the Python path
so that 'import gsim' works correctly in test files.
"""
import sys
from pathlib import Path

# Add the parent directory of gsim to the Python path
# This allows importing gsim as a top-level module
gsim_parent_dir = Path(__file__).resolve().parent.parent.parent
if str(gsim_parent_dir) not in sys.path:
    sys.path.insert(0, str(gsim_parent_dir))
