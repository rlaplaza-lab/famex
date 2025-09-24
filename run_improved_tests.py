#!/usr/bin/env python3
"""
Test runner script for the improved QME test suite.

This script provides convenient ways to run different categories of tests
and generate reports on test coverage and performance.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=None):
    """Run a command and display the result."""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\nSTDOUT:")
    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    print(f"\nReturn code: {result.returncode}")
    return result.returncode == 0


def run_basic_tests():
    """Run basic functionality tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_geometry_improved.py",
        "tests/test_integration_improved.py",
        "-v",
        "--tb=short",
    ]

    return run_command(cmd, "Running Basic Functionality Tests")


def run_cli_tests():
    """Run CLI tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_cli_improved.py",
        "-v",
        "--tb=short",
    ]

    return run_command(cmd, "Running CLI Tests")


def run_file_io_tests():
    """Run file I/O tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_file_io_improved.py",
        "-v",
        "--tb=short",
    ]

    return run_command(cmd, "Running File I/O Tests")


def run_error_handling_tests():
    """Run error handling tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_error_handling_improved.py",
        "-v",
        "--tb=short",
    ]

    return run_command(cmd, "Running Error Handling Tests")


def run_performance_tests():
    """Run performance tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_advanced_workflows.py",
        "-v",
        "--tb=short",
        "-m",
        "performance",
    ]

    return run_command(cmd, "Running Performance Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_advanced_workflows.py",
        "-v",
        "--tb=short",
        "-m",
        "integration",
    ]

    return run_command(cmd, "Running Integration Tests")


def run_all_improved_tests():
    """Run all improved tests."""
    test_files = [
        "tests/test_geometry_improved.py",
        "tests/test_integration_improved.py",
        "tests/test_cli_improved.py",
        "tests/test_file_io_improved.py",
        "tests/test_error_handling_improved.py",
        "tests/test_advanced_workflows.py",
    ]

    cmd = [sys.executable, "-m", "pytest"] + test_files + ["-v", "--tb=short"]

    return run_command(cmd, "Running All Improved Tests")


def run_fast_tests_only():
    """Run only fast tests (exclude slow/performance tests)."""
    test_files = [
        "tests/test_geometry_improved.py",
        "tests/test_integration_improved.py",
        "tests/test_cli_improved.py",
        "tests/test_file_io_improved.py",
        "tests/test_error_handling_improved.py",
        "tests/test_advanced_workflows.py",
    ]

    cmd = (
        [sys.executable, "-m", "pytest"]
        + test_files
        + ["-v", "--tb=short", "-m", "not slow"]
    )

    return run_command(cmd, "Running Fast Tests Only")


def run_with_coverage():
    """Run tests with coverage reporting."""
    test_files = [
        "tests/test_geometry_improved.py",
        "tests/test_integration_improved.py",
        "tests/test_cli_improved.py",
        "tests/test_file_io_improved.py",
        "tests/test_error_handling_improved.py",
    ]

    cmd = (
        [sys.executable, "-m", "pytest"]
        + test_files
        + ["--cov=qme", "--cov-report=html", "--cov-report=term", "-v", "--tb=short"]
    )

    return run_command(cmd, "Running Tests with Coverage")


def run_backend_specific_tests():
    """Run backend-specific tests."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_integration_improved.py",
        "tests/test_advanced_workflows.py",
        "-v",
        "--tb=short",
        "-m",
        "backend_specific",
    ]

    return run_command(cmd, "Running Backend-Specific Tests")


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")

    # Check if pytest is available
    try:
        import pytest

        print(f"✓ pytest {pytest.__version__} is available")
    except ImportError:
        print("✗ pytest is not available")
        return False

    # Check if QME is importable
    try:
        import qme

        print(f"✓ QME is importable")
    except ImportError as e:
        print(f"✗ QME is not importable: {e}")
        return False

    # Check if ASE is available
    try:
        import ase

        print(f"✓ ASE {ase.__version__} is available")
    except ImportError:
        print("✗ ASE is not available")
        return False

    # Check if numpy is available
    try:
        import numpy as np

        print(f"✓ NumPy {np.__version__} is available")
    except ImportError:
        print("✗ NumPy is not available")
        return False

    # Check test files exist
    test_dir = Path("tests")
    improved_test_files = [
        "test_geometry_improved.py",
        "test_integration_improved.py",
        "test_cli_improved.py",
        "test_file_io_improved.py",
        "test_error_handling_improved.py",
        "test_advanced_workflows.py",
        "conftest_improved.py",
    ]

    missing_files = []
    for test_file in improved_test_files:
        if not (test_dir / test_file).exists():
            missing_files.append(test_file)

    if missing_files:
        print(f"✗ Missing test files: {missing_files}")
        return False
    else:
        print("✓ All improved test files are present")

    print("\nTest environment is ready!")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run QME improved test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check          # Check test environment
  %(prog)s --basic          # Run basic functionality tests
  %(prog)s --fast           # Run fast tests only (no performance tests)
  %(prog)s --all            # Run all improved tests
  %(prog)s --coverage       # Run tests with coverage reporting
  %(prog)s --performance    # Run performance benchmarks
        """,
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if test environment is properly set up",
    )
    parser.add_argument(
        "--basic", action="store_true", help="Run basic functionality tests"
    )
    parser.add_argument("--cli", action="store_true", help="Run CLI tests")
    parser.add_argument("--file-io", action="store_true", help="Run file I/O tests")
    parser.add_argument(
        "--error-handling", action="store_true", help="Run error handling tests"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--backend-specific", action="store_true", help="Run backend-specific tests"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run fast tests only (exclude slow tests)"
    )
    parser.add_argument("--all", action="store_true", help="Run all improved tests")
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage reporting"
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    success = True

    if args.check:
        success = check_test_environment() and success

    if args.basic:
        success = run_basic_tests() and success

    if args.cli:
        success = run_cli_tests() and success

    if args.file_io:
        success = run_file_io_tests() and success

    if args.error_handling:
        success = run_error_handling_tests() and success

    if args.performance:
        success = run_performance_tests() and success

    if args.integration:
        success = run_integration_tests() and success

    if args.backend_specific:
        success = run_backend_specific_tests() and success

    if args.fast:
        success = run_fast_tests_only() and success

    if args.all:
        success = run_all_improved_tests() and success

    if args.coverage:
        success = run_with_coverage() and success

    if success:
        print("\n🎉 All selected tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed or encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
