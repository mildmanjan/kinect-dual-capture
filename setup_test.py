#!/usr/bin/env python3
"""
Setup Test - Verify all dependencies are installed correctly
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported"""

    tests = []

    # Test basic packages
    try:
        import numpy as np

        tests.append(("NumPy", "✅", f"v{np.__version__}"))
    except ImportError as e:
        tests.append(("NumPy", "❌", str(e)))

    try:
        import cv2

        tests.append(("OpenCV", "✅", f"v{cv2.__version__}"))
    except ImportError as e:
        tests.append(("OpenCV", "❌", str(e)))

    try:
        import open3d as o3d

        tests.append(("Open3D", "✅", f"v{o3d.__version__}"))
    except ImportError as e:
        tests.append(("Open3D", "❌", str(e)))

    try:
        from pyk4a import PyK4A

        tests.append(("pyk4a (Kinect SDK)", "✅", "Available"))
    except ImportError as e:
        tests.append(
            (
                "pyk4a (Kinect SDK)",
                "⚠️",
                "Not available - install Azure Kinect SDK first",
            )
        )

    try:
        import matplotlib.pyplot as plt

        tests.append(("Matplotlib", "✅", "Available"))
    except ImportError as e:
        tests.append(("Matplotlib", "❌", str(e)))

    try:
        import yaml

        tests.append(("PyYAML", "✅", "Available"))
    except ImportError as e:
        tests.append(("PyYAML", "❌", str(e)))

    # Print results
    print("🔍 Dependency Check Results:")
    print("=" * 50)
    for name, status, info in tests:
        print(f"{status} {name:<20} {info}")

    # Check if critical dependencies are missing
    failed = [test for test in tests if test[1] == "❌"]
    if failed:
        print(f"\n❌ {len(failed)} critical dependencies failed!")
        return False
    else:
        print(f"\n✅ All dependencies installed successfully!")
        return True


def test_project_structure():
    """Test that project structure is correct"""

    print("\n📁 Project Structure Check:")
    print("=" * 50)

    required_paths = [
        "src",
        "src/utils",
        "data",
        "config",
        ".vscode",
        "venv",
        "requirements.txt",
    ]

    all_good = True
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ {path_str}")
        else:
            print(f"❌ {path_str} - MISSING")
            all_good = False

    return all_good


def test_python_path():
    """Test Python environment"""
    print(f"\n🐍 Python Environment:")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(
        f"Virtual environment: {'✅ Active' if 'venv' in sys.executable else '❌ Not active'}"
    )


def main():
    """Run all tests"""
    print("🚀 Kinect Dual Capture - Setup Verification")
    print("=" * 60)

    # Test Python environment
    test_python_path()

    # Test project structure
    structure_ok = test_project_structure()

    # Test imports
    imports_ok = test_imports()

    # Final result
    print("\n" + "=" * 60)
    if structure_ok and imports_ok:
        print("🎉 Setup verification completed successfully!")
        print("✅ Ready to start Kinect development!")
        return 0
    else:
        print("❌ Setup verification failed!")
        print("Please fix the issues above before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
