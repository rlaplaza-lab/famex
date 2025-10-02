"""
Shared utilities for backend handling in QME examples and benchmarks.

This module provides a centralized way to handle backend availability
and ensures consistent behavior across all example files.

Note: This module now imports from the shared qme.core.backend_utils
to avoid code duplication.
"""

import sys
from pathlib import Path

# Add parent directory to path to import qme
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    import qme
    from qme.core.backend_utils import *  # Import all shared utilities
except ImportError as e:
    print(f"Error importing QME: {e}")
    print("Make sure you're in the QME package directory or have it installed.")
    sys.exit(1)