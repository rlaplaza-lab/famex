#!/bin/bash
# 
# Test runner script for QME with MKL fix
#
# This script sets the necessary environment variables to avoid 
# Intel MKL threading conflicts in conda environments.
#
# Usage:
#   ./run_tests.sh                          # Run all tests  
#   ./run_tests.sh tests/test_aimnet2.py    # Run specific test file
#   ./run_tests.sh -k test_water            # Run tests matching pattern

# Set MKL threading layer to avoid symbol conflicts
export MKL_THREADING_LAYER=GNU

echo "Running QME tests with MKL threading fix..."
echo "MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
echo

# Run pytest with all arguments passed through
pytest "$@"
