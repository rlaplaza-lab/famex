#!/usr/bin/env python3
"""Test Option 3 implementation - validate_ts only passed via kwargs."""

import os
import subprocess
import tempfile

from ase import Atoms

# Create a simple test structure
atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0.48, 0.93, 0]])

# Write to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
    atoms.write(f.name, format='xyz')
    temp_file = f.name

try:
    print("Testing Option 3: validate_ts only passed via kwargs...")

    # Test 1: Minima optimization (should NOT receive validate_ts)
    print("\n1. Testing minima optimization (should NOT validate):")
    result1 = subprocess.run([
        'python', '-m', 'qme.cli.cli', 'opt',
        temp_file, '--backend', 'uma', '--fmax', '0.1', '--steps', '1', '--validate-ts'
    ], capture_output=True, text=True, cwd='/home/laplaza/Software/qme')

    print(f"Return code: {result1.returncode}")
    if "TS validation" in result1.stdout or "TS validation" in result1.stderr:
        print("❌ Validation ran for minima optimization (should not happen)")
    else:
        print("✅ No validation for minima optimization (correct)")

    # Test 2: TS optimization without --validate-ts (should NOT validate)
    print("\n2. Testing TS optimization WITHOUT --validate-ts:")
    result2 = subprocess.run([
        'python', '-m', 'qme.cli.cli', 'tsopt', 'local',
        temp_file, '--backend', 'uma', '--fmax', '0.1', '--steps', '1'
    ], capture_output=True, text=True, cwd='/home/laplaza/Software/qme')

    print(f"Return code: {result2.returncode}")
    if "TS validation" in result2.stdout or "TS validation" in result2.stderr:
        print("❌ Validation ran without --validate-ts flag")
    else:
        print("✅ No validation without --validate-ts flag (correct)")

    # Test 3: TS optimization with --validate-ts (should validate)
    print("\n3. Testing TS optimization WITH --validate-ts:")
    result3 = subprocess.run([
        'python', '-m', 'qme.cli.cli', 'tsopt', 'local',
        temp_file, '--backend', 'uma', '--fmax', '0.1', '--steps', '1', '--validate-ts'
    ], capture_output=True, text=True, cwd='/home/laplaza/Software/qme')

    print(f"Return code: {result3.returncode}")
    if "TS validation" in result3.stdout or "TS validation" in result3.stderr:
        print("✅ Validation ran with --validate-ts flag (correct)")
    else:
        print("❌ No validation with --validate-ts flag")

    # Test 4: Path optimization (should NOT receive validate_ts)
    print("\n4. Testing path optimization (should NOT validate):")
    # Create two structures for path optimization
    atoms2 = Atoms('H2O', positions=[[0, 0, 0], [1.0, 0, 0], [0.5, 0.9, 0]])
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f2:
        atoms2.write(f2.name, format='xyz')
        temp_file2 = f2.name

    try:
        result4 = subprocess.run([
            'python', '-m', 'qme.cli.cli', 'path', 'interpolate',
            temp_file, temp_file2, '--backend', 'uma', '--validate-ts'
        ], capture_output=True, text=True, cwd='/home/laplaza/Software/qme')

        print(f"Return code: {result4.returncode}")
        if "TS validation" in result4.stdout or "TS validation" in result4.stderr:
            print("❌ Validation ran for path optimization (should not happen)")
        else:
            print("✅ No validation for path optimization (correct)")
    finally:
        os.unlink(temp_file2)

finally:
    # Clean up
    os.unlink(temp_file)

