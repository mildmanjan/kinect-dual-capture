#!/usr/bin/env python3
"""
Test Kinect Synchronization Quality
Save as: test_sync_quality.py
"""

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


def test_frame_synchronization():
    """Test how synchronized the Kinect frames actually are"""

    if not KINECT_AVAILABLE:
        print("❌ Cannot test - Kinect modules not available")
        return

    print("🔍 Testing Kinect Frame Synchronization")
    print("=" * 50)

    # Initialize both captures
    capture0 = BaseKinectCapture(device_id=0, sync_mode=KinectSyncMode.STANDALONE)
    capture1 = BaseKinectCapture(device_id=1, sync_mode=KinectSyncMode.STANDALONE)

    # Start both
    if not capture0.start_capture():
        print("❌ Failed to start Device 0")
        return

    if not capture1.start_capture():
        print("❌ Failed to start Device 1")
        capture0.stop_capture()
        return

    print("✅ Both devices started")
    print("📊 Capturing 30 frame pairs to test synchronization...")

    time_differences = []
    successful_pairs = 0

    try:
        for i in range(30):
            # Capture from both devices as close as possible
            capture_start = time.time()

            frame0 = capture0.capture_frame()
            frame1 = capture1.capture_frame()

            if frame0 is not None and frame1 is not None:
                # Calculate time difference between frames
                time_diff = abs(frame0.timestamp - frame1.timestamp)
                time_differences.append(time_diff)
                successful_pairs += 1

                if i % 10 == 0:
                    print(f"   Pair {i+1}: Time diff = {time_diff*1000:.1f}ms")

            time.sleep(0.1)  # 100ms between tests

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")

    finally:
        capture0.stop_capture()
        capture1.stop_capture()

    # Analyze results
    if time_differences:
        import numpy as np

        avg_diff = np.mean(time_differences) * 1000  # Convert to ms
        max_diff = np.max(time_differences) * 1000
        std_diff = np.std(time_differences) * 1000

        print(f"\n📊 Synchronization Results:")
        print(f"   Successful pairs: {successful_pairs}/30")
        print(f"   Average time diff: {avg_diff:.1f}ms")
        print(f"   Max time diff: {max_diff:.1f}ms")
        print(f"   Std deviation: {std_diff:.1f}ms")

        # Assessment
        if avg_diff < 16:  # Half a frame at 30fps
            print("   ✅ Excellent synchronization")
        elif avg_diff < 33:  # One frame at 30fps
            print("   ✅ Good synchronization")
        elif avg_diff < 66:  # Two frames at 30fps
            print("   ⚠️  Moderate synchronization")
        else:
            print("   ❌ Poor synchronization - major timing issues")

        # Recommendations based on sync quality
        if avg_diff > 33:
            print(f"\n💡 Sync Issues Detected:")
            print("   - This explains the 96.9% outliers in fusion")
            print("   - Frames are captured at different times")
            print("   - Try sync cable if available")
            print("   - Check USB bandwidth (separate buses)")
        else:
            print(f"\n🤔 Sync seems OK - calibration algorithm issue:")
            print("   - Need better calibration scene")
            print("   - Ensure rich feature overlap")
            print("   - Try manual calibration approach")

    else:
        print("\n❌ No successful frame pairs captured")
        print("   - Check individual device functionality")
        print("   - Verify USB connections")


def check_individual_capture_quality():
    """Check if individual devices are working properly"""

    print(f"\n🔍 Testing Individual Device Quality")
    print("=" * 50)

    for device_id in [0, 1]:
        print(f"\nTesting Device {device_id}:")

        capture = BaseKinectCapture(
            device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
        )

        if not capture.start_capture():
            print(f"   ❌ Failed to start Device {device_id}")
            continue

        frame_count = 0
        successful_frames = 0

        start_time = time.time()

        try:
            while time.time() - start_time < 3:  # 3 second test
                frame = capture.capture_frame()
                frame_count += 1

                if (
                    frame is not None
                    and frame.color_image is not None
                    and frame.depth_image is not None
                ):
                    successful_frames += 1

                time.sleep(0.033)  # ~30fps

        except KeyboardInterrupt:
            pass

        finally:
            capture.stop_capture()

        success_rate = (successful_frames / frame_count * 100) if frame_count > 0 else 0

        print(f"   Frames attempted: {frame_count}")
        print(f"   Successful frames: {successful_frames}")
        print(f"   Success rate: {success_rate:.1f}%")

        if success_rate > 90:
            print(f"   ✅ Device {device_id} working excellently")
        elif success_rate > 70:
            print(f"   ✅ Device {device_id} working well")
        elif success_rate > 50:
            print(f"   ⚠️  Device {device_id} moderate issues")
        else:
            print(f"   ❌ Device {device_id} major issues")


def main():
    """Run synchronization tests"""

    print("🧪 Kinect Synchronization Quality Test")
    print("=" * 60)
    print("This will help explain the 96.9% outlier problem")
    print("")

    # Test individual devices first
    check_individual_capture_quality()

    # Test synchronization
    test_frame_synchronization()

    print(f"\n✅ Synchronization test complete!")
    print(f"\nThis data will help us understand why calibration")
    print(f"produced such poor fusion results despite a valid")
    print(f"transformation matrix.")


if __name__ == "__main__":
    main()
