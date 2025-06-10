#!/usr/bin/env python3
"""
Step 1: Single Kinect Test
Test that a single Azure Kinect is working and streaming locally.

This script will:
- Initialize an Azure Kinect device
- Capture color and depth frames
- Display real-time preview
- Save sample frames for verification
- Report frame rates and data sizes
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.kinect_capture import (
    BaseKinectCapture,
    KinectSyncMode,
    test_kinect_connection,
)


class SingleKinectTester:
    """Single Azure Kinect capture and testing"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.capture = BaseKinectCapture(
            device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
        )

        # Statistics tracking
        self.data_sizes = []

        # Create data directory
        self.data_dir = Path("data/step1_samples")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def display_frame_info(self, frame, frame_number: int):
        """Display frame information"""
        sizes = {
            "color_bytes": frame.color_size,
            "depth_bytes": frame.depth_size,
            "total_bytes": frame.total_size,
        }

        self.data_sizes.append(sizes["total_bytes"])

        # Get current stats
        stats = self.capture.get_capture_stats()
        current_fps = stats.get("current_fps", 0)
        data_rate_mbps = stats.get("data_rate_mbps", 0)

        print(
            f"📊 Frame {frame_number:4d} | "
            f"FPS: {current_fps:5.1f} | "
            f"Data: {sizes['total_bytes']/1024/1024:.2f} MB | "
            f"Rate: {data_rate_mbps:.2f} MB/s"
        )

    def show_preview(self, frame, window_name: str = "Kinect Preview") -> bool:
        """Show preview window with color and depth side by side"""
        if frame.color_image is None and frame.depth_image is None:
            return True

        # Prepare color image
        if frame.color_image is not None:
            color_display = cv2.resize(frame.color_image, (640, 360))
        else:
            color_display = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(
                color_display,
                "No Color Data",
                (200, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Prepare depth image
        if frame.depth_image is not None:
            # Convert depth to colormap for visualization
            depth_normalized = cv2.normalize(
                frame.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_display = cv2.resize(depth_colormap, (640, 360))
        else:
            depth_display = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(
                depth_display,
                "No Depth Data",
                (200, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Create side-by-side display
        combined = np.hstack([color_display, depth_display])

        # Add text overlay
        stats = self.capture.get_capture_stats()
        overlay_text = [
            f"FPS: {stats.get('current_fps', 0):.1f}",
            f"Frames: {stats.get('frames_captured', 0)}",
            f"Data Rate: {stats.get('data_rate_mbps', 0):.1f} MB/s",
            f"Device: {self.device_id}",
            "Press 'q' to quit, 's' to save sample",
        ]

        y_offset = 25
        for i, text in enumerate(overlay_text):
            cv2.putText(
                combined,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        elif key == ord("s"):
            self.save_sample_frame(frame)

        return True

    def save_sample_frame(self, frame):
        """Save a sample frame"""
        self.capture.save_frame(frame, self.data_dir, "sample")
        print(f"📸 Sample frame saved to {self.data_dir}")

    def run_test(
        self, duration: int = 30, show_preview: bool = True, save_samples: bool = True
    ):
        """Run the single Kinect test"""
        print("🚀 Starting Single Kinect Test")
        print("=" * 50)

        # Start capture
        if not self.capture.start_capture():
            print("❌ Failed to start Kinect capture")
            return False

        # Print device info
        device_info = self.capture.get_device_info()
        if device_info:
            print(f"\n📷 Device Info:")
            print(f"  Serial: {device_info.get('serial_number', 'Unknown')}")
            print(
                f"  Color: {device_info.get('color_camera', {}).get('resolution', 'Unknown')}"
            )
            print(
                f"  Depth: {device_info.get('depth_camera', {}).get('resolution', 'Unknown')}"
            )
            print(f"  FPS: {device_info.get('fps', 'Unknown')}")

        print(f"\n🎥 Starting {duration}s capture test...")
        if show_preview:
            print("Press 'q' to quit early, 's' to save current frame")

        try:
            start_time = time.time()
            samples_saved = 0

            while time.time() - start_time < duration:
                # Capture frame
                frame = self.capture.capture_frame()

                if frame is None:
                    print("⚠️  No frame captured, retrying...")
                    continue

                # Display info every 30 frames (~1 second)
                if frame.frame_id % 30 == 0:
                    self.display_frame_info(frame, frame.frame_id)

                # Show preview if requested
                if show_preview:
                    if not self.show_preview(frame):
                        print("🛑 User requested quit")
                        break

                # Auto-save samples at specific intervals
                if (
                    save_samples
                    and frame.frame_id in [1, 30, 60, 150, 300]
                    and samples_saved < 5
                ):
                    self.save_sample_frame(frame)
                    samples_saved += 1

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")

        finally:
            self.capture.stop_capture()
            cv2.destroyAllWindows()

        # Print final statistics
        self.print_final_stats()
        return True

    def print_final_stats(self):
        """Print final capture statistics"""
        stats = self.capture.get_capture_stats()

        if stats["frames_captured"] > 0:
            avg_data_size = np.mean(self.data_sizes) if self.data_sizes else 0
            total_data = sum(self.data_sizes) if self.data_sizes else 0

            print(f"\n📈 Final Statistics:")
            print(f"  Total frames captured: {stats['frames_captured']}")
            print(f"  Frames dropped: {stats['frames_dropped']}")
            print(f"  Average FPS: {stats['average_fps']:.1f}")
            print(f"  Average frame size: {avg_data_size/1024/1024:.2f} MB")
            print(f"  Total data captured: {total_data/1024/1024:.1f} MB")
            print(f"  Data rate: {stats.get('data_rate_mbps', 0):.2f} MB/s")

            # Calculate bandwidth requirements
            if stats["average_fps"] > 0:
                bandwidth_mbps = (avg_data_size * stats["average_fps"] * 8) / (
                    1024 * 1024
                )
                print(f"  Estimated bandwidth needed: {bandwidth_mbps:.1f} Mbps")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test single Azure Kinect capture")
    parser.add_argument(
        "--duration", type=int, default=30, help="Test duration in seconds"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Kinect device ID")
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable preview window"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save sample frames"
    )
    parser.add_argument(
        "--test-connection", action="store_true", help="Just test connection"
    )

    args = parser.parse_args()

    # Just test connection
    if args.test_connection:
        success = test_kinect_connection(args.device_id)
        sys.exit(0 if success else 1)

    # Run full test
    print("🚀 Azure Kinect Single Device Test")
    print("=" * 50)

    tester = SingleKinectTester(device_id=args.device_id)
    success = tester.run_test(
        duration=args.duration,
        show_preview=not args.no_preview,
        save_samples=not args.no_save,
    )

    if success:
        print("✅ Test completed successfully!")
        print(f"📁 Sample files saved to: {tester.data_dir}")
        sys.exit(0)
    else:
        print("❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
