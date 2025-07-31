#!/usr/bin/env python3
"""
Step 4: Dual Kinect Compression
Test compression on dual Kinect capture to reduce total bandwidth.

This script will:
- Capture from both Kinect devices simultaneously
- Apply compression to frames from both devices
- Compare bandwidth usage: dual uncompressed vs dual compressed
- Save compression samples for quality analysis
- Generate dual compression analysis report
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
import argparse
import threading
import queue
from typing import List, Dict, Any
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.kinect_capture import BaseKinectCapture, KinectSyncMode, KinectFrame
from utils.compression_utils import DataCompressor, CompressionLevel


class DualKinectCompressionTester:
    """Test compression on dual Kinect capture"""

    def __init__(self, compression_level: CompressionLevel = CompressionLevel.MEDIUM):
        # Initialize captures
        self.capture0 = BaseKinectCapture(
            device_id=0, sync_mode=KinectSyncMode.STANDALONE
        )
        self.capture1 = BaseKinectCapture(
            device_id=1, sync_mode=KinectSyncMode.STANDALONE
        )

        # Compression
        self.compressor0 = DataCompressor(compression_level)
        self.compressor1 = DataCompressor(compression_level)
        self.compression_level = compression_level

        # Threading
        self.frame_queue0 = queue.Queue(maxsize=5)
        self.frame_queue1 = queue.Queue(maxsize=5)
        self.is_capturing = False
        self.capture_threads = []

        # Statistics
        self.stats = {
            "frames_captured": {"device0": 0, "device1": 0},
            "compression_stats": {"device0": [], "device1": []},
            "frame_pairs": 0,
            "total_original_size": 0,
            "total_compressed_size": 0,
            "start_time": None,
        }

        # Output directory
        self.output_dir = Path("data/step4_dual_compression")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔧 Dual Kinect Compression Tester")
        print(f"   Compression Level: {compression_level.value.upper()}")
        print(f"   Settings: {self.compressor0.get_compression_info()}")

    def capture_and_compress_worker(self, capture, device_id, frame_queue, compressor):
        """Worker thread for capturing and compressing from one device"""
        print(f"📷 Device {device_id} capture+compression thread started")

        frame_count = 0

        try:
            while self.is_capturing:
                # Capture frame
                frame = capture.capture_frame()

                if frame is not None:
                    frame_count += 1

                    # Compress frame immediately
                    if frame.color_image is not None and frame.depth_image is not None:
                        color_compressed, depth_compressed, metadata = (
                            compressor.compress_frame(
                                frame.color_image, frame.depth_image
                            )
                        )

                        # Store compression stats
                        compression_data = {
                            "frame_id": frame.frame_id,
                            "original_size": metadata["total_original_size"],
                            "compressed_size": metadata["total_compressed_size"],
                            "compression_ratio": metadata["total_compression_ratio"],
                            "compression_time": metadata["total_compression_time"],
                            "timestamp": time.time(),
                        }

                        # Add compressed data to frame
                        frame.compressed_color = color_compressed
                        frame.compressed_depth = depth_compressed
                        frame.compression_metadata = metadata

                        # Add to queue
                        try:
                            frame_queue.put_nowait(frame)

                            # Update stats
                            if device_id == 0:
                                self.stats["frames_captured"]["device0"] += 1
                                self.stats["compression_stats"]["device0"].append(
                                    compression_data
                                )
                            else:
                                self.stats["frames_captured"]["device1"] += 1
                                self.stats["compression_stats"]["device1"].append(
                                    compression_data
                                )

                        except queue.Full:
                            # Queue full, skip frame
                            pass

                    if frame_count % 90 == 0:  # Every 3 seconds at 30fps
                        print(
                            f"📊 Device {device_id}: {frame_count} frames captured+compressed"
                        )

                # Control capture rate
                time.sleep(1 / 35)  # Slightly faster than 30fps

        except Exception as e:
            print(f"❌ Device {device_id} capture+compression error: {e}")
        finally:
            print(f"🛑 Device {device_id} capture+compression thread stopped")

    def get_synchronized_compressed_frames(self):
        """Get compressed frames from both devices"""
        frame0 = None
        frame1 = None

        # Get latest frames
        try:
            while not self.frame_queue0.empty():
                frame0 = self.frame_queue0.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.frame_queue1.empty():
                frame1 = self.frame_queue1.get_nowait()
        except queue.Empty:
            pass

        # Update frame pair stats
        if frame0 is not None and frame1 is not None:
            self.stats["frame_pairs"] += 1

            # Update total size stats
            if hasattr(frame0, "compression_metadata") and hasattr(
                frame1, "compression_metadata"
            ):
                self.stats["total_original_size"] += (
                    frame0.compression_metadata["total_original_size"]
                    + frame1.compression_metadata["total_original_size"]
                )
                self.stats["total_compressed_size"] += (
                    frame0.compression_metadata["total_compressed_size"]
                    + frame1.compression_metadata["total_compressed_size"]
                )

        return frame0, frame1

    def show_dual_compression_preview(self, frame0, frame1):
        """Show dual compression preview with original vs compressed"""
        if frame0 is None and frame1 is None:
            return True

        # Display parameters
        display_height = 200
        display_width = int(display_height * 16 / 9)

        # Create displays for each device
        device_displays = []

        for device_id, frame in enumerate([frame0, frame1]):
            if (
                frame is not None
                and frame.color_image is not None
                and hasattr(frame, "compression_metadata")
            ):

                # Decompress for preview
                color_decompressed = self.compressor0.decompress_color_image(
                    frame.compressed_color, frame.compression_metadata["color_metadata"]
                )

                # Original and compressed side by side
                orig_resized = cv2.resize(
                    frame.color_image, (display_width, display_height)
                )
                comp_resized = cv2.resize(
                    color_decompressed, (display_width, display_height)
                )

                # Stack vertically (original on top, compressed below)
                device_display = np.vstack([orig_resized, comp_resized])

                # Add labels
                cv2.putText(
                    device_display,
                    f"Device {device_id} - ORIGINAL",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    device_display,
                    f"Device {device_id} - COMPRESSED",
                    (5, display_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Add compression info
                ratio = frame.compression_metadata["total_compression_ratio"]
                size_kb = frame.compression_metadata["total_compressed_size"] / 1024
                cv2.putText(
                    device_display,
                    f"{ratio:.1f}x, {size_kb:.0f}KB",
                    (5, display_height + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

            else:
                # No data placeholder
                device_display = np.zeros(
                    (display_height * 2, display_width, 3), dtype=np.uint8
                )
                cv2.putText(
                    device_display,
                    f"Device {device_id}: No Data",
                    (50, display_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            device_displays.append(device_display)

        # Combine devices horizontally
        combined = np.hstack(device_displays)

        # Add overall stats
        elapsed = (
            time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        )

        if (
            self.stats["total_original_size"] > 0
            and self.stats["total_compressed_size"] > 0
        ):
            overall_ratio = (
                self.stats["total_original_size"] / self.stats["total_compressed_size"]
            )
            orig_rate_mbps = (
                (self.stats["total_original_size"] * 8) / (elapsed * 1024 * 1024)
                if elapsed > 0
                else 0
            )
            comp_rate_mbps = (
                (self.stats["total_compressed_size"] * 8) / (elapsed * 1024 * 1024)
                if elapsed > 0
                else 0
            )
        else:
            overall_ratio = 0
            orig_rate_mbps = 0
            comp_rate_mbps = 0

        stats_text = [
            f"Dual Kinect Compression - {self.compression_level.value.upper()}",
            f"Time: {elapsed:.0f}s | Pairs: {self.stats['frame_pairs']}",
            f"Overall Ratio: {overall_ratio:.1f}x",
            f"Original: {orig_rate_mbps:.1f} Mbps | Compressed: {comp_rate_mbps:.1f} Mbps",
            f"Bandwidth Saved: {orig_rate_mbps - comp_rate_mbps:.1f} Mbps",
            "",
            "Press 'q' to quit, 's' to save samples",
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(
                combined,
                text,
                (10, display_height * 2 + 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Dual Kinect Compression", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        elif key == ord("s"):
            self.save_compression_samples(frame0, frame1)

        return True

    def save_compression_samples(self, frame0, frame1):
        """Save compression samples from both devices"""
        timestamp = int(time.time())

        for device_id, frame in enumerate([frame0, frame1]):
            if frame is not None and hasattr(frame, "compression_metadata"):
                sample_dir = self.output_dir / f"device{device_id}_samples"
                sample_dir.mkdir(exist_ok=True)

                # Save original frame
                self.capture0.save_frame(frame, sample_dir, f"original_{timestamp}")

                # Decompress and save compressed result
                color_decompressed = self.compressor0.decompress_color_image(
                    frame.compressed_color, frame.compression_metadata["color_metadata"]
                )
                depth_decompressed = self.compressor0.decompress_depth_image(
                    frame.compressed_depth, frame.compression_metadata["depth_metadata"]
                )

                # Create decompressed frame
                decompressed_frame = KinectFrame(
                    timestamp=frame.timestamp,
                    frame_id=frame.frame_id,
                    color_image=color_decompressed,
                    depth_image=depth_decompressed,
                )

                self.capture0.save_frame(
                    decompressed_frame, sample_dir, f"compressed_{timestamp}"
                )

                # Save metadata
                metadata_path = sample_dir / f"metadata_{timestamp}.json"
                with open(metadata_path, "w") as f:
                    json.dump(frame.compression_metadata, f, indent=2, default=str)

        print(f"📸 Dual compression samples saved at {timestamp}")

    def start_dual_capture_compression(self):
        """Start dual capture with compression"""
        print("\n🚀 Starting dual capture with compression...")

        # Start captures
        if not self.capture0.start_capture():
            print("❌ Failed to start Device 0")
            return False

        time.sleep(1)  # Stagger startup

        if not self.capture1.start_capture():
            print("❌ Failed to start Device 1")
            self.capture0.stop_capture()
            return False

        print("✅ Both devices started")

        # Start capture threads
        self.is_capturing = True
        self.stats["start_time"] = time.time()

        thread0 = threading.Thread(
            target=self.capture_and_compress_worker,
            args=(self.capture0, 0, self.frame_queue0, self.compressor0),
            daemon=True,
        )
        thread1 = threading.Thread(
            target=self.capture_and_compress_worker,
            args=(self.capture1, 1, self.frame_queue1, self.compressor1),
            daemon=True,
        )

        self.capture_threads = [thread0, thread1]
        thread0.start()
        thread1.start()

        print("✅ Dual capture+compression threads started")
        return True

    def stop_dual_capture_compression(self):
        """Stop dual capture and compression"""
        print("🛑 Stopping dual capture+compression...")

        self.is_capturing = False
        time.sleep(0.5)  # Wait for threads

        self.capture0.stop_capture()
        self.capture1.stop_capture()

        cv2.destroyAllWindows()
        print("✅ Dual capture+compression stopped")

    def run_compression_test(self, duration: int = 20):
        """Run dual compression test"""
        print(
            f"🧪 Dual Kinect Compression Test - {self.compression_level.value.upper()}"
        )
        print("=" * 70)

        if not self.start_dual_capture_compression():
            return False

        try:
            start_time = time.time()
            last_print_time = start_time

            while time.time() - start_time < duration:
                # Get compressed frames
                frame0, frame1 = self.get_synchronized_compressed_frames()

                # Show preview
                if not self.show_dual_compression_preview(frame0, frame1):
                    print("🛑 User requested quit")
                    break

                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_print_time >= 5:
                    self.print_progress()
                    last_print_time = current_time

                time.sleep(1 / 15)  # 15 FPS display

        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        finally:
            self.stop_dual_capture_compression()

        # Generate final report
        self.generate_final_report()
        return True

    def print_progress(self):
        """Print current compression progress"""
        elapsed = time.time() - self.stats["start_time"]

        dev0_frames = self.stats["frames_captured"]["device0"]
        dev1_frames = self.stats["frames_captured"]["device1"]
        pairs = self.stats["frame_pairs"]

        print(
            f"📊 {elapsed:.0f}s: Dev0={dev0_frames}, Dev1={dev1_frames}, Pairs={pairs}"
        )

    def generate_final_report(self):
        """Generate final compression report"""
        elapsed = time.time() - self.stats["start_time"]

        print(f"\n📈 Dual Kinect Compression Results")
        print("=" * 70)

        # Device stats
        for device_id in [0, 1]:
            frames = self.stats["frames_captured"][f"device{device_id}"]
            fps = frames / elapsed if elapsed > 0 else 0
            print(f"Device {device_id}: {frames} frames ({fps:.1f} fps)")

        print(f"Frame pairs: {self.stats['frame_pairs']}")

        # Compression stats
        if (
            self.stats["total_original_size"] > 0
            and self.stats["total_compressed_size"] > 0
        ):
            overall_ratio = (
                self.stats["total_original_size"] / self.stats["total_compressed_size"]
            )

            orig_bandwidth = (self.stats["total_original_size"] * 8) / (
                elapsed * 1024 * 1024
            )
            comp_bandwidth = (self.stats["total_compressed_size"] * 8) / (
                elapsed * 1024 * 1024
            )
            bandwidth_saved = orig_bandwidth - comp_bandwidth

            print(f"\nCompression Results:")
            print(f"  Overall compression ratio: {overall_ratio:.1f}x")
            print(f"  Original bandwidth: {orig_bandwidth:.1f} Mbps")
            print(f"  Compressed bandwidth: {comp_bandwidth:.1f} Mbps")
            print(
                f"  Bandwidth saved: {bandwidth_saved:.1f} Mbps ({bandwidth_saved/orig_bandwidth*100:.1f}%)"
            )

            # Assessment
            if comp_bandwidth < 50:
                print(f"  ✅ Excellent - suitable for network streaming")
            elif comp_bandwidth < 100:
                print(f"  ✅ Good - suitable for local network")
            elif comp_bandwidth < 200:
                print(f"  ⚠️  Moderate - may need higher compression")
            else:
                print(f"  ❌ High bandwidth - increase compression level")

        # Save results
        self.save_test_results()

    def save_test_results(self):
        """Save test results to JSON"""
        elapsed = time.time() - self.stats["start_time"]

        results = {
            "test_timestamp": time.time(),
            "compression_level": self.compression_level.value,
            "test_duration": elapsed,
            "device_stats": {
                "device0_frames": self.stats["frames_captured"]["device0"],
                "device1_frames": self.stats["frames_captured"]["device1"],
                "frame_pairs": self.stats["frame_pairs"],
            },
            "compression_stats": {
                "total_original_size": self.stats["total_original_size"],
                "total_compressed_size": self.stats["total_compressed_size"],
                "overall_compression_ratio": (
                    self.stats["total_original_size"]
                    / self.stats["total_compressed_size"]
                    if self.stats["total_compressed_size"] > 0
                    else 0
                ),
                "original_bandwidth_mbps": (
                    (self.stats["total_original_size"] * 8) / (elapsed * 1024 * 1024)
                    if elapsed > 0
                    else 0
                ),
                "compressed_bandwidth_mbps": (
                    (self.stats["total_compressed_size"] * 8) / (elapsed * 1024 * 1024)
                    if elapsed > 0
                    else 0
                ),
            },
        }

        results_file = (
            self.output_dir
            / f"dual_compression_results_{self.compression_level.value}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📊 Results saved to: {results_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test dual Kinect compression")
    parser.add_argument(
        "--duration", type=int, default=20, help="Test duration in seconds"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="medium",
        choices=["low", "medium", "high", "extreme"],
        help="Compression level to test",
    )

    args = parser.parse_args()

    compression_level = CompressionLevel(args.compression)

    print("🚀 Dual Azure Kinect Compression Test")
    print("=" * 50)
    print(f"📋 Configuration:")
    print(f"   Compression level: {compression_level.value.upper()}")
    print(f"   Test duration: {args.duration}s")
    print("")

    tester = DualKinectCompressionTester(compression_level)
    success = tester.run_compression_test(args.duration)

    if success:
        print("\n✅ Dual compression test completed!")
        print(f"📁 Results and samples saved to: {tester.output_dir}")
        print("\n💡 Next Steps:")
        print("   - Review bandwidth reduction results")
        print("   - Check sample quality")
        print("   - Try different compression levels")
        print("   - Ready for Step 5: Point cloud fusion")
    else:
        print("\n❌ Dual compression test failed!")


if __name__ == "__main__":
    main()
