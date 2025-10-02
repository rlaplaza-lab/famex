"""
Shared utilities for backend testing across all test modules.

This module provides a centralized way to handle backend availability
and ensures consistent behavior across all test files.

Note: This module now imports from the shared qme.core.backend_utils
to avoid code duplication.
"""

import qme
from qme.core.backend_utils import *  # Import all shared utilities
