#!/usr/bin/env python3
"""
Calibration Diagnostic
Save as: calibration_diagnostic.py
"""

import numpy as np
import json
import time
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from utils.kinect_capture import BaseKinectCapture, KinectSyncMode

    KINECT_AVAILABLE = True
except ImportError:
    print("❌ Kinect utilities not available")
    KINECT_AVAILABLE = False


def check_calibration_consistency():
    """Check if calibration is being applied consistently"""

    print("🔍 Calibration Consistency Check")
    print("=" * 50)

    calib_file = Path("config/dual_kinect_calibration.json")

    if not calib_file.exists():
        print("❌ No calibration file found")
        return

    try:
        with open(calib_file, "r") as f:
            calib_data = json.load(f)

        error = calib_data.get("calibration_error", 0)
        transform = np.array(calib_data.get("transformation_matrix", np.eye(4)))
        method = calib_data.get("calibration_method", "unknown")

        print(f"📊 Current Calibration:")
        print(f"   Error: {error:.1f}mm")
        print(f"   Method: {method}")

        # Check transformation
        translation = transform[:3, 3]
        distance = np.linalg.norm(translation)

        print(f"   Kinect separation: {distance:.3f}m")
        print(
            f"   Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
        )

        # Check rotation
        rotation = transform[:3, :3]
        det = np.linalg.det(rotation)

        print(f"   Rotation determinant: {det:.6f}")

        # Extract rotation angles (simplified)
        rx = np.arctan2(rotation[2, 1], rotation[2, 2]) * 180 / np.pi
        ry = (
            np.arctan2(
                -rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)
            )
            * 180
            / np.pi
        )
        rz = np.arctan2(rotation[1, 0], rotation[0, 0]) * 180 / np.pi

        print(f"   Rotation angles: X={rx:.1f}°, Y={ry:.1f}°, Z={rz:.1f}°")

        total_rotation = np.sqrt(rx**2 + ry**2 + rz**2)

        if total_rotation < 1.0:
            print("   ⚠️  Very small rotation - might indicate calibration issue")
        elif total_rotation > 45:
            print("   ⚠️  Large rotation - check Kinect positioning")
        else:
            print("   ✅ Reasonable rotation")

        return True

    except Exception as e:
        print(f"❌ Error reading calibration: {e}")
        return False


def test_individual_point_clouds():
    """Test individual point cloud quality before fusion"""

    print(f"\n🔍 Individual Point Cloud Quality Test")
    print("=" * 50)

    if not KINECT_AVAILABLE:
        print("❌ Cannot test - Kinect modules not available")
        return

    for device_id in [0, 1]:
        print(f"\nTesting Device {device_id}:")

        capture = BaseKinectCapture(
            device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
        )

        if not capture.start_capture():
            print(f"   ❌ Failed to start Device {device_id}")
            continue

        try:
            # Get a frame
            frame = capture.capture_frame()

            if frame is None or frame.depth_image is None:
                print(f"   ❌ No frame data")
                continue

            # Analyze depth quality
            depth = frame.depth_image
            valid_pixels = np.sum((depth > 300) & (depth < 3000))
            total_pixels = depth.shape[0] * depth.shape[1]
            coverage = (valid_pixels / total_pixels) * 100

            # Basic depth statistics
            valid_depths = depth[(depth > 300) & (depth < 3000)]
            if len(valid_depths) > 0:
                mean_depth = np.mean(valid_depths)
                std_depth = np.std(valid_depths)
                depth_range = np.max(valid_depths) - np.min(valid_depths)
            else:
                mean_depth = std_depth = depth_range = 0

            print(f"   📊 Depth Quality:")
            print(f"      Coverage: {coverage:.1f}%")
            print(f"      Mean depth: {mean_depth:.0f}mm")
            print(f"      Depth range: {depth_range:.0f}mm")
            print(f"      Noise (std): {std_depth:.1f}mm")

            if coverage < 30:
                print(f"      ⚠️  Low scene coverage")
            elif std_depth > 50:
                print(f"      ⚠️  High depth noise")
            else:
                print(f"      ✅ Good depth quality")

        except Exception as e:
            print(f"   ❌ Error testing device: {e}")

        finally:
            capture.stop_capture()


def test_sync_during_fusion():
    """Test synchronization specifically during fusion-like capture"""

    print(f"\n🔍 Fusion Synchronization Test")
    print("=" * 50)

    if not KINECT_AVAILABLE:
        print("❌ Cannot test - Kinect modules not available")
        return

    capture0 = BaseKinectCapture(device_id=0, sync_mode=KinectSyncMode.STANDALONE)
    capture1 = BaseKinectCapture(device_id=1, sync_mode=KinectSyncMode.STANDALONE)

    if not capture0.start_capture():
        print("❌ Failed to start Device 0")
        return

    if not capture1.start_capture():
        print("❌ Failed to start Device 1")
        capture0.stop_capture()
        return

    print("📊 Testing 10 synchronized captures...")

    time_diffs = []
    scene_changes = []

    try:
        for i in range(10):
            # Capture frames as close as possible (like fusion does)
            frame0 = capture0.capture_frame()
            frame1 = capture1.capture_frame()

            if frame0 is not None and frame1 is not None:
                # Time difference
                time_diff = abs(frame0.timestamp - frame1.timestamp)
                time_diffs.append(time_diff)

                # Check if scenes are similar (rough depth comparison)
                if frame0.depth_image is not None and frame1.depth_image is not None:
                    depth0_mean = np.mean(frame0.depth_image[frame0.depth_image > 0])
                    depth1_mean = np.mean(frame1.depth_image[frame1.depth_image > 0])
                    scene_diff = abs(depth0_mean - depth1_mean)
                    scene_changes.append(scene_diff)

                if i % 3 == 0:
                    print(f"   Capture {i+1}: Time diff = {time_diff*1000:.1f}ms")

            time.sleep(0.2)  # Similar to fusion capture rate

    except Exception as e:
        print(f"❌ Error during sync test: {e}")

    finally:
        capture0.stop_capture()
        capture1.stop_capture()

    if time_diffs:
        avg_time_diff = np.mean(time_diffs) * 1000
        max_time_diff = np.max(time_diffs) * 1000

        print(f"\n📊 Fusion Sync Results:")
        print(f"   Average time diff: {avg_time_diff:.1f}ms")
        print(f"   Max time diff: {max_time_diff:.1f}ms")

        if avg_time_diff > 50:
            print(f"   ❌ Poor sync during fusion - this explains 96% outliers")
            print(f"   💡 Try sync cable or different USB configuration")
        else:
            print(f"   ✅ Good sync - problem is elsewhere")

    if scene_changes:
        avg_scene_change = np.mean(scene_changes)
        print(f"   Average scene difference: {avg_scene_change:.1f}mm")

        if avg_scene_change > 100:
            print(f"   ⚠️  Large scene differences between Kinects")


def diagnose_fusion_problem():
    """Main diagnostic to figure out why fusion still fails"""

    print(f"\n💡 Fusion Problem Diagnosis")
    print("=" * 50)

    print("Based on your results:")
    print("✅ Hardware working perfectly (100% device success)")
    print("✅ USB buses separated properly")
    print("✅ Calibration computed (24.5mm, 209 correspondences)")
    print("❌ Fusion still terrible (96.0% outliers)")
    print("")

    print("🔍 Most likely causes:")
    print("1. Scene overlap insufficient for fusion (different from calibration scene)")
    print("2. Calibration scene != fusion scene (calibration doesn't apply)")
    print("3. Time sync issues during actual fusion")
    print("4. Kinect positioning changed between calibration and fusion")
    print("5. Depth quality issues in fusion scene")
    print("")

    print("🎯 Recommended next steps:")
    print("1. Run calibration and fusion in SAME scene")
    print("2. Use sync cable if available")
    print("3. Ensure Kinects haven't moved since calibration")
    print("4. Try manual calibration verification")


def main():
    """Run complete diagnostic"""

    print("🔧 Calibration Problem Diagnostic")
    print("=" * 60)
    print("Why is fusion still poor despite good calibration?")
    print("")

    # Check calibration data
    check_calibration_consistency()

    # Test individual devices
    test_individual_point_clouds()

    # Test sync during fusion
    test_sync_during_fusion()

    # Diagnose the problem
    diagnose_fusion_problem()

    print(f"\n✅ Diagnostic complete!")


if __name__ == "__main__":
    main()
