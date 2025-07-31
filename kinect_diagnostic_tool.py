#!/usr/bin/env python3
"""
Kinect Diagnostic Tool
Simple test of individual Kinect devices

Location: kinect_diagnostic_tool.py (in project root)

This script:
- Tests basic device connectivity
- Checks USB connection stability
- Identifies which devices are working
- Provides troubleshooting guidance
"""

import sys
import time
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode, WiredSyncMode, FPS
    KINECT_AVAILABLE = True
    print("[+] Azure Kinect SDK available")
except ImportError as e:
    print(f"[X] Azure Kinect SDK not available: {e}")
    print("    Install Azure Kinect SDK v1.4.1 first")
    KINECT_AVAILABLE = False


def test_device_simple(device_id):
    """Simple device test - just creation and start/stop"""
    print(f"\n[TEST] Device {device_id}...")

    try:
        # Create simple config
        config = Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_2X2BINNED,
            camera_fps=FPS.FPS_5,
            synchronized_images_only=False,
            wired_sync_mode=WiredSyncMode.STANDALONE,
        )

        print(f"  [1] Creating Device {device_id}...", end=" ")
        device = PyK4A(config=config, device_id=device_id)
        print("OK")

        print(f"  [2] Starting Device {device_id}...", end=" ")
        device.start()
        print("OK")

        print(f"  [3] Testing frame capture...", end=" ")
        capture = device.get_capture()
        if capture and (capture.color is not None or capture.depth is not None):
            print("OK")
            has_color = capture.color is not None
            has_depth = capture.depth is not None
            print(f"      Color: {'YES' if has_color else 'NO'}")
            print(f"      Depth: {'YES' if has_depth else 'NO'}")
            success = True
        else:
            print("FAIL - No data")
            success = False

        print(f"  [4] Stopping Device {device_id}...", end=" ")
        device.stop()
        print("OK")

        if success:
            print(f"  [+] Device {device_id}: ALL TESTS PASSED")
        else:
            print(f"  [X] Device {device_id}: FRAME CAPTURE FAILED")
            
        return success

    except Exception as e:
        print("FAIL")
        print(f"  [X] Device {device_id} error: {e}")

        # Categorize error
        error_str = str(e).lower()
        if "not found" in error_str or "no device" in error_str:
            print(f"      -> Device {device_id} not connected/detected")
        elif "in use" in error_str or "busy" in error_str:
            print(f"      -> Device {device_id} in use by another app")
        elif "usb" in error_str:
            print(f"      -> Device {device_id} USB connection issue")
        elif "permission" in error_str:
            print(f"      -> Device {device_id} permission issue")
        else:
            print(f"      -> Device {device_id} unknown error")

        return False


def test_with_step1_code(device_id):
    """Test with Step1 code without capture"""
    print(f"\n[INTEGRATION] Testing Device {device_id} with project code...")

    try:
        from utils.kinect_capture import BaseKinectCapture, KinectSyncMode

        print(f"  [1] Creating capture object...", end=" ")
        capture = BaseKinectCapture(
            device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
        )
        print("OK")

        print(f"  [2] Initializing device...", end=" ")
        if capture.initialize():
            print("OK")
            print(f"  [3] Getting device info...", end=" ")
            info = capture.get_device_info()
            print("OK")
            
            if info:
                print(f"      Serial: {info.get('serial_number', 'Unknown')}")
                print(f"      Color: {info.get('color_camera', {}).get('resolution', 'Unknown')}")
            
            capture.stop_capture()
            print(f"  [+] Device {device_id}: Project integration OK")
            return True
        else:
            print("FAIL")
            print(f"  [X] Device {device_id}: Project initialization failed")
            return False

    except Exception as e:
        print("FAIL")
        print(f"  [X] Device {device_id} integration error: {e}")
        return False


def main():
    """Main diagnostic function"""
    print("Kinect Hardware Diagnostic Tool")
    print("=" * 50)
    print("This tool tests basic Kinect device connectivity")
    print("")

    if not KINECT_AVAILABLE:
        print("[X] Cannot test - Azure Kinect SDK not available")
        print("\nTroubleshooting:")
        print("1. Download Azure Kinect SDK v1.4.1")
        print("2. Install to: C:\\Program Files\\Azure Kinect SDK v1.4.1")
        print("3. Restart Python/IDE")
        print("4. Run: pip install --force-reinstall pyk4a")
        return 1

    # Test devices individually
    results = {}
    
    for device_id in [0, 1]:
        print(f"\n{'='*30} DEVICE {device_id} {'='*30}")
        
        # Basic device test
        basic_ok = test_device_simple(device_id)
        results[device_id] = {"basic": basic_ok}
        
        # Integration test (only if basic works)
        if basic_ok:
            integration_ok = test_with_step1_code(device_id)
            results[device_id]["integration"] = integration_ok
        else:
            results[device_id]["integration"] = False

    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    working_devices = []
    for device_id in [0, 1]:
        basic = results[device_id]["basic"]
        integration = results[device_id]["integration"]
        
        if basic and integration:
            print(f"[+] Device {device_id}: FULLY WORKING")
            working_devices.append(device_id)
        elif basic:
            print(f"[!] Device {device_id}: Hardware OK, Integration issues")
        else:
            print(f"[X] Device {device_id}: NOT WORKING")

    print(f"\nWorking devices: {len(working_devices)} of 2")

    # Recommendations
    if len(working_devices) == 0:
        print("\n[X] NO DEVICES WORKING")
        print("Troubleshooting checklist:")
        print("1. Check USB 3.0 connections (blue ports)")
        print("2. Ensure Kinect power (green LED on device)")
        print("3. Try Microsoft's Azure Kinect Viewer")
        print("4. Close other applications using camera")
        print("5. Try different USB ports")
        print("6. Restart computer")
        
    elif len(working_devices) == 1:
        device_id = working_devices[0]
        print(f"\n[!] SINGLE DEVICE WORKING: Device {device_id}")
        print("For dual Kinect setup:")
        print("1. Check second device connections")
        print("2. Try different USB controllers")
        print("3. Ensure adequate power for both devices")
        print(f"\nCurrent recommendation: Use Device {device_id} for single Kinect tests")
        
    else:
        print(f"\n[+] DUAL KINECT READY!")
        print("Both devices working - ready for dual capture")
        print("\nNext steps:")
        print("1. Run single device tests:")
        print(f"   python src\\step1_single_kinect_test.py --device-id 0 --duration 10")
        print(f"   python src\\step1_single_kinect_test.py --device-id 1 --duration 10")
        print("2. Test dual capture:")
        print(f"   python src\\step3_dual_kinect_test.py --duration 15")

    # Exit codes
    if len(working_devices) == 2:
        return 0  # Perfect
    elif len(working_devices) == 1:
        return 1  # Partial success
    else:
        return 2  # No devices working


if __name__ == "__main__":
    sys.exit(main())