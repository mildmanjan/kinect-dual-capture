#!/usr/bin/env python3
"""
Step 3: Dual Kinect Test
Test synchronized capture from two Azure Kinect devices.

This script:
- Tests individual device functionality
- Attempts synchronized dual capture
- Measures frame rates and synchronization
- Provides troubleshooting guidance
- Saves sample synchronized frames

File location: src/step3_dual_kinect_test.py
"""

import cv2
import numpy as np
import time
import threading
import queue
import sys
from pathlib import Path
import argparse
from typing import Optional, Tuple, Dict, Any

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.kinect_capture import (
    BaseKinectCapture,
    KinectSyncMode,
    KinectFrame,
    test_kinect_connection,
)


class DualKinectTester:
    """Test dual Kinect capture functionality"""

    def __init__(
        self, device1_id: int = 0, device2_id: int = 1, use_sync_cable: bool = False
    ):
        self.device1_id = device1_id
        self.device2_id = device2_id
        self.use_sync_cable = use_sync_cable

        # Determine sync modes
        if use_sync_cable:
            self.sync_mode1 = KinectSyncMode.MASTER
            self.sync_mode2 = KinectSyncMode.SUBORDINATE
        else:
            self.sync_mode1 = KinectSyncMode.STANDALONE
            self.sync_mode2 = KinectSyncMode.STANDALONE

        # Initialize captures
        self.capture1 = BaseKinectCapture(
            device_id=device1_id, sync_mode=self.sync_mode1
        )
        self.capture2 = BaseKinectCapture(
            device_id=device2_id, sync_mode=self.sync_mode2
        )

        # Threading for simultaneous capture
        self.capture_threads = []
        self.frame_queues = {
            "device1": queue.Queue(maxsize=10),
            "device2": queue.Queue(maxsize=10),
        }
        self.is_capturing = False

        # Statistics
        self.test_stats = {
            "device1_frames": 0,
            "device2_frames": 0,
            "synchronized_pairs": 0,
            "max_time_diff": 0.0,
            "avg_time_diff": 0.0,
            "frame_drops": 0,
            "start_time": None,
        }

        # Output directory
        self.output_dir = Path("data/step3_dual_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_individual_devices(self) -> Tuple[bool, bool]:
        """Test each device individually first"""
        print("🔍 Testing individual device functionality...")

        # Test Device 1
        print(f"\n📷 Testing Device {self.device1_id}...")
        device1_ok = test_kinect_connection(self.device1_id)

        if device1_ok:
            print(f"✅ Device {self.device1_id}: Working correctly")
        else:
            print(f"❌ Device {self.device1_id}: Failed basic test")

        # Test Device 2
        print(f"\n📷 Testing Device {self.device2_id}...")
        device2_ok = test_kinect_connection(self.device2_id)

        if device2_ok:
            print(f"✅ Device {self.device2_id}: Working correctly")
        else:
            print(f"❌ Device {self.device2_id}: Failed basic test")

        return device1_ok, device2_ok

    def get_device_info(self):
        """Get and display device information"""
        print("\n📋 Device Information:")
        print("=" * 40)

        try:
            # Device 1 info
            if self.capture1.initialize():
                info1 = self.capture1.get_device_info()
                print(f"Device {self.device1_id} ({self.sync_mode1.value}):")
                print(f"  Serial: {info1.get('serial_number', 'Unknown')}")
                print(
                    f"  Color: {info1.get('color_camera', {}).get('resolution', 'Unknown')}"
                )
                print(
                    f"  Depth: {info1.get('depth_camera', {}).get('resolution', 'Unknown')}"
                )
                self.capture1.stop_capture()
            else:
                print(f"Device {self.device1_id}: Failed to get info")

            time.sleep(1)  # Brief pause between devices

            # Device 2 info
            if self.capture2.initialize():
                info2 = self.capture2.get_device_info()
                print(f"Device {self.device2_id} ({self.sync_mode2.value}):")
                print(f"  Serial: {info2.get('serial_number', 'Unknown')}")
                print(
                    f"  Color: {info2.get('color_camera', {}).get('resolution', 'Unknown')}"
                )
                print(
                    f"  Depth: {info2.get('depth_camera', {}).get('resolution', 'Unknown')}"
                )
                self.capture2.stop_capture()
            else:
                print(f"Device {self.device2_id}: Failed to get info")

        except Exception as e:
            print(f"❌ Error getting device info: {e}")

    def capture_worker(self, capture: BaseKinectCapture, device_name: str):
        """Worker thread for capturing from one device"""
        frame_count = 0

        try:
            while self.is_capturing:
                frame = capture.capture_frame()

                if frame is not None:
                    frame_count += 1

                    # Add to queue
                    try:
                        self.frame_queues[device_name].put_nowait(frame)

                        # Update stats
                        if device_name == "device1":
                            self.test_stats["device1_frames"] += 1
                        else:
                            self.test_stats["device2_frames"] += 1

                    except queue.Full:
                        # Queue full, drop frame
                        self.test_stats["frame_drops"] += 1

                time.sleep(1 / 35)  # Slightly faster than 30fps to avoid bottleneck

        except Exception as e:
            print(f"❌ Capture worker error for {device_name}: {e}")

        print(f"🛑 {device_name} capture worker stopped ({frame_count} frames)")

    def get_synchronized_frames(
        self, timeout: float = 0.1
    ) -> Tuple[Optional[KinectFrame], Optional[KinectFrame]]:
        """Get the most recent frames from both devices"""
        frame1 = None
        frame2 = None

        # Get latest frame from device 1
        try:
            while not self.frame_queues["device1"].empty():
                frame1 = self.frame_queues["device1"].get_nowait()
        except queue.Empty:
            pass

        # Get latest frame from device 2
        try:
            while not self.frame_queues["device2"].empty():
                frame2 = self.frame_queues["device2"].get_nowait()
        except queue.Empty:
            pass

        # Check synchronization if we have both frames
        if frame1 is not None and frame2 is not None:
            time_diff = abs(frame1.timestamp - frame2.timestamp)
            self.test_stats["synchronized_pairs"] += 1

            # Update time difference stats
            if time_diff > self.test_stats["max_time_diff"]:
                self.test_stats["max_time_diff"] = time_diff

            # Running average of time differences
            n = self.test_stats["synchronized_pairs"]
            self.test_stats["avg_time_diff"] = (
                self.test_stats["avg_time_diff"] * (n - 1) + time_diff
            ) / n

        return frame1, frame2

    def show_dual_preview(
        self, frame1: Optional[KinectFrame], frame2: Optional[KinectFrame]
    ) -> bool:
        """Show side-by-side preview of both devices"""
        display_height = 240
        display_width = int(display_height * 16 / 9)

        # Prepare device 1 display
        if frame1 is not None and frame1.color_image is not None:
            display1 = cv2.resize(frame1.color_image, (display_width, display_height))
            cv2.putText(
                display1,
                f"Device {self.device1_id}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display1,
                f"{self.sync_mode1.value}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        else:
            display1 = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            cv2.putText(
                display1,
                f"Device {self.device1_id}: No Data",
                (20, display_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Prepare device 2 display
        if frame2 is not None and frame2.color_image is not None:
            display2 = cv2.resize(frame2.color_image, (display_width, display_height))
            cv2.putText(
                display2,
                f"Device {self.device2_id}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display2,
                f"{self.sync_mode2.value}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        else:
            display2 = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            cv2.putText(
                display2,
                f"Device {self.device2_id}: No Data",
                (20, display_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Combine displays
        combined = np.hstack([display1, display2])

        # Add overall statistics
        stats_y = display_height + 20
        stats_text = [
            f"Dual Kinect Test - {self.sync_mode1.value}/{self.sync_mode2.value}",
            f"Frames: Dev1={self.test_stats['device1_frames']}, Dev2={self.test_stats['device2_frames']}",
            f"Synchronized pairs: {self.test_stats['synchronized_pairs']}",
            f"Max time diff: {self.test_stats['max_time_diff']*1000:.1f}ms",
            f"Avg time diff: {self.test_stats['avg_time_diff']*1000:.1f}ms",
            f"Frame drops: {self.test_stats['frame_drops']}",
            "",
            "Press 'q' to quit, 's' to save sample",
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(
                combined,
                text,
                (10, stats_y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Dual Kinect Test", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        elif key == ord("s"):
            self.save_sample_frames(frame1, frame2)

        return True

    def save_sample_frames(
        self, frame1: Optional[KinectFrame], frame2: Optional[KinectFrame]
    ):
        """Save sample synchronized frames"""
        if frame1 is None and frame2 is None:
            print("⚠️  No frames to save")
            return

        timestamp = int(time.time())

        if frame1 is not None:
            self.capture1.save_frame(
                frame1, self.output_dir, f"device1_sample_{timestamp}"
            )

        if frame2 is not None:
            self.capture2.save_frame(
                frame2, self.output_dir, f"device2_sample_{timestamp}"
            )

        # Save synchronization info
        sync_info = {
            "timestamp": timestamp,
            "device1_frame_time": frame1.timestamp if frame1 else None,
            "device2_frame_time": frame2.timestamp if frame2 else None,
            "time_difference_ms": (
                abs(frame1.timestamp - frame2.timestamp) * 1000
                if (frame1 and frame2)
                else None
            ),
            "sync_mode": f"{self.sync_mode1.value}/{self.sync_mode2.value}",
            "use_sync_cable": self.use_sync_cable,
        }

        import json

        sync_file = self.output_dir / f"sync_info_{timestamp}.json"
        with open(sync_file, "w") as f:
            json.dump(sync_info, f, indent=2)

        print(f"📸 Sample frames saved with timestamp {timestamp}")

    def start_dual_capture(self) -> bool:
        """Start synchronized capture from both devices"""
        print("\n🚀 Starting dual capture...")

        # Start device 1 first (master if using sync cable)
        if not self.capture1.start_capture():
            print(f"❌ Failed to start Device {self.device1_id}")
            return False

        # Brief delay before starting second device
        time.sleep(0.5 if self.use_sync_cable else 0.1)

        # Start device 2
        if not self.capture2.start_capture():
            print(f"❌ Failed to start Device {self.device2_id}")
            self.capture1.stop_capture()
            return False

        print("✅ Both devices started successfully")

        # Start capture threads
        self.is_capturing = True

        thread1 = threading.Thread(
            target=self.capture_worker, args=(self.capture1, "device1"), daemon=True
        )
        thread2 = threading.Thread(
            target=self.capture_worker, args=(self.capture2, "device2"), daemon=True
        )

        self.capture_threads = [thread1, thread2]
        thread1.start()
        thread2.start()

        return True

    def stop_dual_capture(self):
        """Stop dual capture"""
        print("\n🛑 Stopping dual capture...")

        self.is_capturing = False
        time.sleep(0.5)  # Wait for threads to finish

        self.capture1.stop_capture()
        self.capture2.stop_capture()

        cv2.destroyAllWindows()

    def run_dual_test(self, duration: int = 30, show_preview: bool = True) -> bool:
        """Run the complete dual Kinect test"""
        print("🎯 Dual Kinect Capture Test")
        print("=" * 50)
        print(f"Device 1: {self.device1_id} ({self.sync_mode1.value})")
        print(f"Device 2: {self.device2_id} ({self.sync_mode2.value})")
        print(f"Sync cable: {'Yes' if self.use_sync_cable else 'No'}")
        print(f"Duration: {duration}s")

        # Reset statistics
        self.test_stats = {
            "device1_frames": 0,
            "device2_frames": 0,
            "synchronized_pairs": 0,
            "max_time_diff": 0.0,
            "avg_time_diff": 0.0,
            "frame_drops": 0,
            "start_time": time.time(),
        }

        # Start dual capture
        if not self.start_dual_capture():
            return False

        try:
            start_time = time.time()
            last_print_time = start_time

            while time.time() - start_time < duration:
                # Get synchronized frames
                frame1, frame2 = self.get_synchronized_frames()

                # Show preview if requested
                if show_preview:
                    if not self.show_dual_preview(frame1, frame2):
                        print("🛑 User requested quit")
                        break

                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_print_time >= 5:
                    self.print_progress()
                    last_print_time = current_time

                time.sleep(1 / 15)  # 15 FPS display rate

        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")

        finally:
            self.stop_dual_capture()

        # Print final results
        self.print_final_results()
        return True

    def print_progress(self):
        """Print current test progress"""
        elapsed = time.time() - self.test_stats["start_time"]
        fps1 = self.test_stats["device1_frames"] / elapsed if elapsed > 0 else 0
        fps2 = self.test_stats["device2_frames"] / elapsed if elapsed > 0 else 0

        print(
            f"📊 {elapsed:.0f}s: Dev1={fps1:.1f}fps ({self.test_stats['device1_frames']} frames), "
            f"Dev2={fps2:.1f}fps ({self.test_stats['device2_frames']} frames), "
            f"Pairs={self.test_stats['synchronized_pairs']}, "
            f"Drops={self.test_stats['frame_drops']}"
        )

    def print_final_results(self):
        """Print final test results and assessment"""
        elapsed = time.time() - self.test_stats["start_time"]

        print(f"\n📈 Dual Kinect Test Results")
        print("=" * 50)
        print(f"Test duration: {elapsed:.1f}s")
        print(
            f"Device {self.device1_id}: {self.test_stats['device1_frames']} frames "
            f"({self.test_stats['device1_frames']/elapsed:.1f} fps)"
        )
        print(
            f"Device {self.device2_id}: {self.test_stats['device2_frames']} frames "
            f"({self.test_stats['device2_frames']/elapsed:.1f} fps)"
        )
        print(f"Synchronized pairs: {self.test_stats['synchronized_pairs']}")
        print(f"Frame drops: {self.test_stats['frame_drops']}")

        if self.test_stats["synchronized_pairs"] > 0:
            print(f"Max time difference: {self.test_stats['max_time_diff']*1000:.1f}ms")
            print(f"Avg time difference: {self.test_stats['avg_time_diff']*1000:.1f}ms")

        # Assessment
        print(f"\n📊 Assessment:")

        min_expected_frames = elapsed * 20  # Expect at least 20 fps
        device1_ok = self.test_stats["device1_frames"] >= min_expected_frames
        device2_ok = self.test_stats["device2_frames"] >= min_expected_frames

        if device1_ok and device2_ok:
            print("✅ Both devices capturing successfully")

            # Check synchronization quality
            if self.test_stats["synchronized_pairs"] > 0:
                if (
                    self.test_stats["avg_time_diff"] < 0.033
                ):  # < 33ms (30fps frame time)
                    print("✅ Good synchronization quality")
                elif self.test_stats["avg_time_diff"] < 0.100:  # < 100ms
                    print("⚠️  Moderate synchronization quality")
                else:
                    print("❌ Poor synchronization quality")

            # Check frame drops
            if (
                self.test_stats["frame_drops"] < elapsed * 2
            ):  # Less than 2 drops per second
                print("✅ Low frame drop rate")
            else:
                print("⚠️  High frame drop rate - check USB bandwidth")

        elif device1_ok or device2_ok:
            working_device = self.device1_id if device1_ok else self.device2_id
            failing_device = self.device2_id if device1_ok else self.device1_id
            print(
                f"⚠️  Device {working_device} working, Device {failing_device} having issues"
            )
        else:
            print("❌ Both devices having capture issues")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if device1_ok and device2_ok:
            if self.use_sync_cable:
                print("✅ Sync cable setup working well")
                print("📈 Ready for Step 4: Dual Kinect compression")
                print("📈 Ready for Step 5: Point cloud fusion")
            else:
                print("✅ Standalone dual capture working")
                print("💡 Consider sync cable for better synchronization")
        else:
            print("🔧 Troubleshooting needed:")
            print("   - Check USB 3.0 connections")
            print("   - Try different USB ports/controllers")
            print("   - Close other applications")
            print("   - Check power supply adequacy")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test dual Azure Kinect capture")
    parser.add_argument("--device1", type=int, default=0, help="First Kinect device ID")
    parser.add_argument(
        "--device2", type=int, default=1, help="Second Kinect device ID"
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Test duration in seconds"
    )
    parser.add_argument(
        "--sync-cable", action="store_true", help="Use sync cable (master/subordinate)"
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable preview window"
    )
    parser.add_argument(
        "--individual-only", action="store_true", help="Test devices individually only"
    )

    args = parser.parse_args()

    print("🚀 Dual Azure Kinect Test")
    print("=" * 40)

    tester = DualKinectTester(
        device1_id=args.device1, device2_id=args.device2, use_sync_cable=args.sync_cable
    )

    # Test individual devices first
    device1_ok, device2_ok = tester.test_individual_devices()

    if not (device1_ok and device2_ok):
        print(f"\n❌ Device issues detected:")
        if not device1_ok:
            print(f"   Device {args.device1}: Not working")
        if not device2_ok:
            print(f"   Device {args.device2}: Not working")

        print("\n💡 Troubleshooting:")
        print("   - Check USB 3.0 connections")
        print("   - Ensure Kinect power (green LEDs)")
        print("   - Try: python kinect_diagnostic_tool.py")
        return 1

    # Get device information
    tester.get_device_info()

    if args.individual_only:
        print("✅ Individual device tests passed")
        return 0

    # Run dual capture test
    success = tester.run_dual_test(
        duration=args.duration, show_preview=not args.no_preview
    )

    if success:
        print(f"\n✅ Dual Kinect test completed!")
        print(f"📁 Sample frames saved to: {tester.output_dir}")
    else:
        print(f"\n❌ Dual Kinect test failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
