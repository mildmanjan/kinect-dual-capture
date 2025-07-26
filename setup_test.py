#!/usr/bin/env python3
"""
Setup Test - Verify all dependencies are installed correctly
Fixed version for Windows compatibility (no Unicode emojis)

Location: setup_test.py (replace existing file in project root)
"""

import sys
import os
from pathlib import Path


def safe_print(text):
    """Print text safely, handling Unicode issues on Windows"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        ascii_text = text.encode("ascii", "replace").decode("ascii")
        print(ascii_text)


def test_imports():
    """Test that all required packages can be imported"""

    tests = []

    # Test basic packages
    try:
        import numpy as np

        tests.append(("NumPy", "OK", f"v{np.__version__}"))
    except ImportError as e:
        tests.append(("NumPy", "FAIL", str(e)))

    try:
        import cv2

        tests.append(("OpenCV", "OK", f"v{cv2.__version__}"))
    except ImportError as e:
        tests.append(("OpenCV", "FAIL", str(e)))

    try:
        import open3d as o3d

        tests.append(("Open3D", "OK", f"v{o3d.__version__}"))
    except ImportError as e:
        tests.append(("Open3D", "FAIL", str(e)))

    try:
        from pyk4a import PyK4A

        tests.append(("pyk4a (Kinect SDK)", "OK", "Available"))
    except ImportError as e:
        tests.append(
            (
                "pyk4a (Kinect SDK)",
                "WARN",
                "Not available - install Azure Kinect SDK first",
            )
        )

    try:
        import matplotlib.pyplot as plt

        tests.append(("Matplotlib", "OK", "Available"))
    except ImportError as e:
        tests.append(("Matplotlib", "FAIL", str(e)))

    try:
        import yaml

        tests.append(("PyYAML", "OK", "Available"))
    except ImportError as e:
        tests.append(("PyYAML", "FAIL", str(e)))

    try:
        import websockets

        tests.append(("WebSockets", "OK", "Available"))
    except ImportError as e:
        tests.append(("WebSockets", "WARN", "Install with: pip install websockets"))

    try:
        import tkinter as tk

        tests.append(("Tkinter (GUI)", "OK", "Available"))
    except ImportError as e:
        tests.append(("Tkinter (GUI)", "FAIL", "GUI not available"))

    # Print results
    safe_print("Dependency Check Results:")
    safe_print("=" * 50)
    for name, status, info in tests:
        status_symbol = {"OK": "[+]", "FAIL": "[X]", "WARN": "[!]"}.get(status, "[?]")

        safe_print(f"{status_symbol} {name:<20} {info}")

    # Check if critical dependencies are missing
    failed = [test for test in tests if test[1] == "FAIL"]
    warnings = [test for test in tests if test[1] == "WARN"]

    if failed:
        safe_print(f"\n[X] {len(failed)} critical dependencies failed!")
        safe_print("Critical failures:")
        for name, _, info in failed:
            safe_print(f"    - {name}: {info}")
        return False
    elif warnings:
        safe_print(f"\n[!] {len(warnings)} warnings (optional dependencies):")
        for name, _, info in warnings:
            safe_print(f"    - {name}: {info}")
        safe_print("\n[+] Core dependencies OK - warnings are for optional features")
        return True
    else:
        safe_print(f"\n[+] All dependencies installed successfully!")
        return True


def test_project_structure():
    """Test that project structure is correct"""

    safe_print("\nProject Structure Check:")
    safe_print("=" * 50)

    required_paths = [
        "src",
        "src/utils",
        "data",
        "config",
        "requirements.txt",
    ]

    optional_paths = [".vscode", "venv", ".git"]

    all_good = True

    # Check required paths
    safe_print("Required directories:")
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            safe_print(f"[+] {path_str}")
        else:
            safe_print(f"[X] {path_str} - MISSING")
            all_good = False

    # Check optional paths
    safe_print("\nOptional directories:")
    for path_str in optional_paths:
        path = Path(path_str)
        if path.exists():
            safe_print(f"[+] {path_str}")
        else:
            safe_print(f"[ ] {path_str} - not present (optional)")

    return all_good


def test_python_environment():
    """Test Python environment"""
    safe_print(f"\nPython Environment:")
    safe_print("=" * 50)
    safe_print(f"Python version: {sys.version.split()[0]}")
    safe_print(f"Python executable: {sys.executable}")

    # Check if in virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    venv_status = "[+] Active" if in_venv else "[!] Not active (recommended)"
    safe_print(f"Virtual environment: {venv_status}")

    # Check Python version
    if sys.version_info >= (3, 8):
        safe_print("[+] Python version compatible (3.8+)")
    else:
        safe_print(
            f"[!] Python version may be too old (current: {sys.version_info.major}.{sys.version_info.minor})"
        )

    return True


def test_key_scripts():
    """Test that key scripts exist"""
    safe_print(f"\nKey Scripts Check:")
    safe_print("=" * 50)

    key_scripts = [
        ("src/step1_single_kinect_test.py", "Single Kinect test"),
        ("src/step2_single_kinect_compression.py", "Compression test"),
        ("src/step3_dual_kinect_test.py", "Dual Kinect test"),
        ("src/step5_kinect_fusion.py", "Point cloud fusion"),
        ("src/utils/kinect_capture.py", "Kinect utilities"),
        ("src/utils/compression_utils.py", "Compression utilities"),
    ]

    all_scripts_present = True
    for script_path, description in key_scripts:
        path = Path(script_path)
        if path.exists():
            safe_print(f"[+] {script_path:<35} ({description})")
        else:
            safe_print(f"[X] {script_path:<35} MISSING")
            all_scripts_present = False

    return all_scripts_present


def check_data_directories():
    """Check and create data directories if needed"""
    safe_print(f"\nData Directories:")
    safe_print("=" * 50)

    data_dirs = [
        "data/step1_samples",
        "data/compression_test",
        "data/step3_dual_test",
        "data/fusion_results",
        "data/mesh_animation",
        "data/web_viewer",
        "config",
    ]

    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            safe_print(f"[+] {dir_path}")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                safe_print(f"[+] {dir_path} (created)")
            except Exception as e:
                safe_print(f"[X] {dir_path} (failed to create: {e})")

    return True


def main():
    """Run all tests"""
    safe_print("Kinect Dual Capture - Setup Verification")
    safe_print("=" * 60)

    # Test Python environment
    test_python_environment()

    # Test project structure
    structure_ok = test_project_structure()

    # Test imports
    imports_ok = test_imports()

    # Test key scripts
    scripts_ok = test_key_scripts()

    # Check/create data directories
    check_data_directories()

    # Final result
    safe_print("\n" + "=" * 60)

    if structure_ok and imports_ok and scripts_ok:
        safe_print("SETUP VERIFICATION COMPLETED SUCCESSFULLY!")
        safe_print("[+] Ready to start Kinect development!")

        # Provide next steps
        safe_print("\nNext Steps:")
        safe_print("1. Run kinect_diagnostic_tool.py to test hardware")
        safe_print("2. Use kinect_setup_gui.py for easy script management")
        safe_print("3. Start with Single Kinect Tests in the GUI")

        return 0
    else:
        safe_print("SETUP VERIFICATION FAILED!")
        safe_print("[X] Please fix the issues above before continuing.")

        # Provide troubleshooting tips
        safe_print("\nTroubleshooting:")
        if not imports_ok:
            safe_print("- Install missing packages: pip install -r requirements.txt")
        if not structure_ok:
            safe_print("- Check project directory structure")
        if not scripts_ok:
            safe_print("- Ensure all source files are present")

        return 1


if __name__ == "__main__":
    sys.exit(main())
