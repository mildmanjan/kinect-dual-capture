#!/usr/bin/env python3
"""
Ultra Simple Kinect Test
Minimal device test - just creation and start/stop

Location: ultra_simple_kinect_test.py (in project root)

This script:
- Tests the absolute minimum Kinect functionality
- Good for isolating hardware vs software issues
- Minimal dependencies and overhead
"""

import sys
from pathlib import Path

# Add utils to path if available
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode, WiredSyncMode, FPS

    KINECT_AVAILABLE = True
    print("[+] pyk4a imported successfully")
except ImportError as e:
    print(f"[X] pyk4a import failed: {e}")
    KINECT_AVAILABLE = False


def ultra_simple_test(device_id):
    """Ultra minimal device test"""
    print(f"\n[MINIMAL TEST] Device {device_id}...")

    try:
        # Absolute minimum config
        config = Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_2X2BINNED,
            camera_fps=FPS.FPS_5,
            synchronized_images_only=False,
            wired_sync_mode=WiredSyncMode.STANDALONE,
        )

        print(f"  Creating device object...", end=" ")
        device = PyK4A(config=config, device_id=device_id)
        print("OK")

        print(f"  Starting device...", end=" ")
        device.start()
        print("OK")

        print(f"  Stopping device...", end=" ")
        device.stop()
        print("OK")

        print(f"  [+] Device {device_id}: Basic operations successful")
        return True

    except Exception as e:
        print("FAIL")
        print(f"  [X] Device {device_id} failed: {e}")

        # Simple error categorization
        error_str = str(e).lower()
        if "not found" in error_str:
            print(f"      -> Device {device_id} not connected")
        elif "in use" in error_str:
            print(f"      -> Device {device_id} used by another app")
        elif "usb" in error_str:
            print(f"      -> Device {device_id} USB issue")
        else:
            print(f"      -> Device {device_id} other error")

        return False


def test_project_integration():
    """Test if project utilities work"""
    print(f"\n[PROJECT TEST] Testing project integration...")

    try:
        print(f"  Importing kinect_capture...", end=" ")
        from utils.kinect_capture import test_kinect_connection

        print("OK")

        return True
    except ImportError as e:
        print("FAIL")
        print(f"  [X] Project utilities not available: {e}")
        print(f"      -> Check that src/utils/kinect_capture.py exists")
        return False
    except Exception as e:
        print("FAIL")
        print(f"  [X] Project integration error: {e}")
        return False


def main():
    """Ultra simple main test"""
    print("Ultra Simple Kinect Test")
    print("=" * 40)
    print("Minimal functionality test for troubleshooting")
    print("")

    if not KINECT_AVAILABLE:
        print("[X] Cannot test - pyk4a not available")
        print("\nQuick fixes:")
        print("1. Install Azure Kinect SDK v1.4.1")
        print("2. Run: pip install pyk4a")
        print("3. Restart Python")
        return 1

    # Test project integration first
    project_ok = test_project_integration()

    # Test devices
    results = {}
    for device_id in [0, 1]:
        print(f"\n--- DEVICE {device_id} ---")
        results[device_id] = ultra_simple_test(device_id)

    # Summary
    print(f"\n{'='*50}")
    print("ULTRA SIMPLE TEST RESULTS")
    print("=" * 50)

    working = [dev for dev, ok in results.items() if ok]

    print(f"Project integration: {'OK' if project_ok else 'FAIL'}")
    for device_id, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"Device {device_id}: {status}")

    print(f"\nWorking devices: {len(working)} of 2")

    if len(working) == 0:
        print("\n[X] No devices working")
        print("Basic troubleshooting:")
        print("1. Check power - green LED on Kinect")
        print("2. Check USB 3.0 connection (blue port)")
        print("3. Try Microsoft Azure Kinect Viewer")
        print("4. Close other camera applications")

    elif len(working) == 1:
        print(f"\n[!] Only Device {working[0]} working")
        print("For second device:")
        print("1. Try different USB port")
        print("2. Check power cable")
        print("3. Ensure separate USB controllers")

    else:
        print("\n[+] Both devices basic functionality OK")
        print("Ready for more detailed testing")

    # Return appropriate exit code
    return 0 if len(working) == 2 else (1 if len(working) == 1 else 2)


if __name__ == "__main__":
    sys.exit(main())
