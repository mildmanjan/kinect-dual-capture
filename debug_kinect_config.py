#!/usr/bin/env python3
"""
Debug Kinect Configuration
Test different configurations to find what works with your Kinect

Location: debug_kinect_config.py (in project root)

This script:
- Tests different resolutions and frame rates
- Finds working configurations for your hardware
- Identifies USB bandwidth limitations
- Provides recommendations for optimal settings
"""

try:
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode, WiredSyncMode, FPS

    print("[+] pyk4a imported successfully")
except ImportError as e:
    print(f"[X] Failed to import pyk4a: {e}")
    print("Install Azure Kinect SDK v1.4.1 first")
    exit(1)


def show_available_options():
    """Show all available configuration options"""
    print("\n[INFO] Available Configuration Options:")
    print("=" * 50)

    print("Frame Rates:")
    fps_options = [
        (FPS.FPS_5, "5 FPS"),
        (FPS.FPS_15, "15 FPS"),
        (FPS.FPS_30, "30 FPS"),
    ]
    for fps_enum, description in fps_options:
        print(f"  {description}: {fps_enum}")

    print("\nColor Resolutions:")
    color_resolutions = [
        (ColorResolution.RES_720P, "720p (1280x720)"),
        (ColorResolution.RES_1080P, "1080p (1920x1080)"),
        (ColorResolution.RES_1440P, "1440p (2560x1440)"),
        (ColorResolution.RES_1536P, "1536p (2048x1536)"),
        (ColorResolution.RES_2160P, "2160p (3840x2160)"),
        (ColorResolution.RES_3072P, "3072p (4096x3072)"),
    ]
    for res_enum, description in color_resolutions:
        print(f"  {description}: {res_enum}")

    print("\nDepth Modes:")
    depth_modes = [
        (DepthMode.NFOV_2X2BINNED, "NFOV 2x2 Binned (320x288)"),
        (DepthMode.NFOV_UNBINNED, "NFOV Unbinned (640x576)"),
        (DepthMode.WFOV_2X2BINNED, "WFOV 2x2 Binned (512x512)"),
        (DepthMode.WFOV_UNBINNED, "WFOV Unbinned (1024x1024)"),
    ]
    for mode_enum, description in depth_modes:
        print(f"  {description}: {mode_enum}")


def test_configuration(config_info):
    """Test a specific configuration"""
    print(f"\n[TEST] {config_info['name']}")
    print(f"  Color: {config_info['color_desc']}")
    print(f"  Depth: {config_info['depth_desc']}")
    print(f"  FPS: {config_info['fps_desc']}")

    try:
        config = Config(
            color_resolution=config_info["color_resolution"],
            depth_mode=config_info["depth_mode"],
            camera_fps=config_info["camera_fps"],
            synchronized_images_only=True,
            wired_sync_mode=WiredSyncMode.STANDALONE,
        )

        # Try to create device
        print(f"  [1] Creating device...", end=" ")
        kinect = PyK4A(config=config, device_id=0)
        print("OK")

        # Try to start
        print(f"  [2] Starting cameras...", end=" ")
        kinect.start()
        print("OK")

        # Try to get a frame
        print(f"  [3] Capturing frame...", end=" ")
        capture = kinect.get_capture()
        if capture.color is not None and capture.depth is not None:
            print("OK")

            # Get frame info
            color_shape = capture.color.shape if capture.color is not None else "None"
            depth_shape = capture.depth.shape if capture.depth is not None else "None"
            print(f"      Color shape: {color_shape}")
            print(f"      Depth shape: {depth_shape}")

            success = True
        else:
            print("FAIL - No data")
            success = False

        print(f"  [4] Stopping...", end=" ")
        kinect.stop()
        print("OK")

        if success:
            print(f"  [+] {config_info['name']} - SUCCESS")
        else:
            print(f"  [X] {config_info['name']} - FRAME CAPTURE FAILED")

        return success

    except Exception as e:
        print("FAIL")
        print(f"  [X] {config_info['name']} - ERROR: {e}")

        # Categorize common errors
        error_str = str(e).lower()
        if "bandwidth" in error_str or "usb" in error_str:
            print(f"      -> USB bandwidth issue - try lower resolution/fps")
        elif "timeout" in error_str:
            print(f"      -> Device timeout - hardware issue")
        elif "in use" in error_str:
            print(f"      -> Device busy - close other applications")

        return False


def main():
    """Main configuration testing function"""
    print("Kinect Configuration Debug Tool")
    print("=" * 60)
    print("Testing different settings to find what works with your hardware")

    # Show available options
    show_available_options()

    # Test configurations from most conservative to most demanding
    test_configs = [
        {
            "name": "Ultra Conservative (720p, 5fps, Binned)",
            "color_resolution": ColorResolution.RES_720P,
            "depth_mode": DepthMode.NFOV_2X2BINNED,
            "camera_fps": FPS.FPS_5,
            "color_desc": "720p",
            "depth_desc": "NFOV 2x2 Binned",
            "fps_desc": "5 FPS",
        },
        {
            "name": "Conservative (720p, 15fps, Binned)",
            "color_resolution": ColorResolution.RES_720P,
            "depth_mode": DepthMode.NFOV_2X2BINNED,
            "camera_fps": FPS.FPS_15,
            "color_desc": "720p",
            "depth_desc": "NFOV 2x2 Binned",
            "fps_desc": "15 FPS",
        },
        {
            "name": "Balanced (1080p, 15fps, Unbinned)",
            "color_resolution": ColorResolution.RES_1080P,
            "depth_mode": DepthMode.NFOV_UNBINNED,
            "camera_fps": FPS.FPS_15,
            "color_desc": "1080p",
            "depth_desc": "NFOV Unbinned",
            "fps_desc": "15 FPS",
        },
        {
            "name": "Standard (1080p, 30fps, Unbinned)",
            "color_resolution": ColorResolution.RES_1080P,
            "depth_mode": DepthMode.NFOV_UNBINNED,
            "camera_fps": FPS.FPS_30,
            "color_desc": "1080p",
            "depth_desc": "NFOV Unbinned",
            "fps_desc": "30 FPS",
        },
        {
            "name": "High Quality (1440p, 30fps, Wide)",
            "color_resolution": ColorResolution.RES_1440P,
            "depth_mode": DepthMode.WFOV_UNBINNED,
            "camera_fps": FPS.FPS_30,
            "color_desc": "1440p",
            "depth_desc": "WFOV Unbinned",
            "fps_desc": "30 FPS",
        },
    ]

    print(f"\n[TESTING] Trying {len(test_configs)} configurations...")
    working_configs = []

    for config in test_configs:
        success = test_configuration(config)
        if success:
            working_configs.append(config)

    # Results summary
    print(f"\n{'='*60}")
    print("CONFIGURATION TEST RESULTS")
    print("=" * 60)

    if working_configs:
        print(f"[+] Found {len(working_configs)} working configuration(s):")
        for i, config in enumerate(working_configs):
            print(f"  {i+1}. {config['name']}")
            print(
                f"     {config['color_desc']} + {config['depth_desc']} @ {config['fps_desc']}"
            )

        print(f"\n[RECOMMENDATION] Best configuration:")
        best_config = working_configs[0]  # First working config (most conservative)
        print(f"  Name: {best_config['name']}")
        print(f"  Color: {best_config['color_desc']}")
        print(f"  Depth: {best_config['depth_desc']}")
        print(f"  FPS: {best_config['fps_desc']}")

        print(f"\n[USAGE] To use this in your scripts:")
        print(
            f"  color_resolution=ColorResolution.{best_config['color_resolution'].name}"
        )
        print(f"  depth_mode=DepthMode.{best_config['depth_mode'].name}")
        print(f"  camera_fps=FPS.{best_config['camera_fps'].name}")

    else:
        print("[X] No working configurations found!")
        print("\nPossible issues:")
        print("1. Kinect not connected properly")
        print("2. USB 3.0 port required")
        print("3. Kinect SDK not installed correctly")
        print("4. Another application using the Kinect")
        print("5. Insufficient USB power")

        print("\nTroubleshooting steps:")
        print("1. Try Microsoft Azure Kinect Viewer")
        print("2. Use different USB 3.0 port")
        print("3. Close other camera applications")
        print("4. Restart computer")
        print("5. Check Kinect power LED (should be green)")

    # Dual Kinect advice
    if len(working_configs) > 0:
        print(f"\n[DUAL KINECT] For dual device setup:")
        if len(working_configs) >= 3:  # Have some headroom
            recommended = working_configs[1]  # Second most conservative
            print(f"  Recommended: {recommended['name']}")
            print(f"  Should work well with 2 devices")
        else:
            recommended = working_configs[0]  # Most conservative
            print(f"  Use most conservative: {recommended['name']}")
            print(f"  May need even lower settings for 2 devices")

        print(f"  Monitor for 'queue overflow' errors with dual capture")

    return 0 if working_configs else 1


if __name__ == "__main__":
    main()
