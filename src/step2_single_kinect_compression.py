#!/usr/bin/env python3
"""
Step 2: Single Kinect Compression
Test variable compression to reduce data bandwidth from a single Kinect.

This script will:
- Capture from single Kinect with different compression levels
- Compare data rates and quality at each compression level
- Save samples for quality analysis
- Generate compression analysis report
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.kinect_capture import BaseKinectCapture, KinectSyncMode, KinectFrame
from utils.compression_utils import DataCompressor, CompressionLevel


class CompressionTester:
    """Test compression on single Kinect capture"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.capture = BaseKinectCapture(
            device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
        )

        # Results storage
        self.compression_results = []

        # Output directory
        self.output_dir = Path("data/compression_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_compression_level(
        self, compression_level: CompressionLevel, test_duration: int = 10
    ) -> Dict[str, Any]:
        """Test a specific compression level"""
        print(f"\n🧪 Testing compression level: {compression_level.value.upper()}")

        compressor = DataCompressor(compression_level)
        compression_info = compressor.get_compression_info()

        print(f"  📋 Settings: {compression_info}")

        # Statistics
        original_sizes = []
        compressed_sizes = []
        compression_times = []
        compression_ratios = []
        frame_count = 0

        start_time = time.time()

        # Create preview window
        window_name = f"Compression Test - {compression_level.value.upper()}"

        while time.time() - start_time < test_duration:
            # Capture frame
            frame = self.capture.capture_frame()
            if not frame or frame.color_image is None or frame.depth_image is None:
                continue

            frame_count += 1

            # Compress frame
            color_compressed, depth_compressed, metadata = compressor.compress_frame(
                frame.color_image, frame.depth_image
            )

            # Test decompression to ensure it works
            color_decompressed = compressor.decompress_color_image(
                color_compressed, metadata["color_metadata"]
            )
            depth_decompressed = compressor.decompress_depth_image(
                depth_compressed, metadata["depth_metadata"]
            )

            # Collect statistics
            original_sizes.append(metadata["total_original_size"])
            compressed_sizes.append(metadata["total_compressed_size"])
            compression_times.append(metadata["total_compression_time"])
            compression_ratios.append(metadata["total_compression_ratio"])

            # Show preview every 10 frames
            if frame_count % 10 == 0:
                self.show_compression_preview(
                    frame.color_image,
                    frame.depth_image,
                    color_decompressed,
                    depth_decompressed,
                    metadata,
                    frame_count,
                    window_name,
                )

            # Save sample frames at specific intervals
            if frame_count in [1, 30, 60]:
                self.save_compression_sample(
                    frame, compressor, compression_level, frame_count
                )

            # Print progress
            if frame_count % 30 == 0:
                avg_ratio = (
                    np.mean(compression_ratios[-30:]) if compression_ratios else 0
                )
                current_fps = frame_count / (time.time() - start_time)
                compressed_rate = (
                    (np.mean(compressed_sizes[-30:]) * current_fps) / (1024 * 1024)
                    if compressed_sizes
                    else 0
                )
                print(
                    f"    📊 Frame {frame_count}: {avg_ratio:.1f}x compression, {compressed_rate:.1f} MB/s"
                )

        cv2.destroyWindow(window_name)

        # Calculate results
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        results = {
            "compression_level": compression_level.value,
            "settings": compression_info,
            "frames_tested": frame_count,
            "test_duration": total_time,
            "avg_fps": avg_fps,
            "original_size_avg_mb": np.mean(original_sizes) / (1024 * 1024),
            "compressed_size_avg_mb": np.mean(compressed_sizes) / (1024 * 1024),
            "compression_ratio": np.mean(compression_ratios),
            "compression_time_avg_ms": np.mean(compression_times) * 1000,
            "original_bandwidth_mbps": (np.mean(original_sizes) * avg_fps * 8)
            / (1024 * 1024),
            "compressed_bandwidth_mbps": (np.mean(compressed_sizes) * avg_fps * 8)
            / (1024 * 1024),
            "bandwidth_reduction": np.mean(compression_ratios),
            "data_rate_mb_per_sec": (np.mean(compressed_sizes) * avg_fps)
            / (1024 * 1024),
        }

        print(f"  ✅ Results:")
        print(f"    📊 Compression ratio: {results['compression_ratio']:.1f}x")
        print(
            f"    📊 Original bandwidth: {results['original_bandwidth_mbps']:.1f} Mbps"
        )
        print(
            f"    📊 Compressed bandwidth: {results['compressed_bandwidth_mbps']:.1f} Mbps"
        )
        print(f"    📊 Data rate: {results['data_rate_mb_per_sec']:.1f} MB/s")
        print(
            f"    ⏱️  Compression time: {results['compression_time_avg_ms']:.1f}ms per frame"
        )

        return results

    def show_compression_preview(
        self,
        original_color,
        original_depth,
        compressed_color,
        compressed_depth,
        metadata,
        frame_count,
        window_name,
    ):
        """Show side-by-side comparison of original vs compressed"""

        # Resize for display
        display_height = 240
        display_width = int(display_height * 16 / 9)

        # Original images
        if original_color is not None:
            orig_color_resized = cv2.resize(
                original_color, (display_width, display_height)
            )
        else:
            orig_color_resized = np.zeros(
                (display_height, display_width, 3), dtype=np.uint8
            )

        if original_depth is not None:
            orig_depth_norm = cv2.normalize(
                original_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            orig_depth_colored = cv2.applyColorMap(orig_depth_norm, cv2.COLORMAP_JET)
            orig_depth_resized = cv2.resize(
                orig_depth_colored, (display_width, display_height)
            )
        else:
            orig_depth_resized = np.zeros(
                (display_height, display_width, 3), dtype=np.uint8
            )

        # Compressed images
        if compressed_color is not None:
            comp_color_resized = cv2.resize(
                compressed_color, (display_width, display_height)
            )
        else:
            comp_color_resized = np.zeros(
                (display_height, display_width, 3), dtype=np.uint8
            )

        if compressed_depth is not None:
            comp_depth_norm = cv2.normalize(
                compressed_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            comp_depth_colored = cv2.applyColorMap(comp_depth_norm, cv2.COLORMAP_JET)
            comp_depth_resized = cv2.resize(
                comp_depth_colored, (display_width, display_height)
            )
        else:
            comp_depth_resized = np.zeros(
                (display_height, display_width, 3), dtype=np.uint8
            )

        # Create comparison layout
        top_row = np.hstack([orig_color_resized, orig_depth_resized])
        bottom_row = np.hstack([comp_color_resized, comp_depth_resized])
        combined = np.vstack([top_row, bottom_row])

        # Add labels
        cv2.putText(
            combined,
            "ORIGINAL",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            "COMPRESSED",
            (10, display_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Add compression info
        info_text = [
            f"Frame: {frame_count}",
            f"Ratio: {metadata['total_compression_ratio']:.1f}x",
            f"Size: {metadata['total_compressed_size']/1024:.0f} KB",
            "Press 'q' to stop",
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                combined,
                text,
                (display_width + 10, 50 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow(window_name, combined)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

        return True

    def save_compression_sample(
        self,
        frame: KinectFrame,
        compressor: DataCompressor,
        level: CompressionLevel,
        frame_num: int,
    ):
        """Save sample frame with compression applied"""
        sample_dir = self.output_dir / f"samples_{level.value}"
        sample_dir.mkdir(exist_ok=True)

        # Save original
        self.capture.save_frame(frame, sample_dir, f"original_frame{frame_num}")

        # Compress and decompress
        color_compressed, depth_compressed, metadata = compressor.compress_frame(
            frame.color_image, frame.depth_image
        )

        color_decompressed = compressor.decompress_color_image(
            color_compressed, metadata["color_metadata"]
        )
        depth_decompressed = compressor.decompress_depth_image(
            depth_compressed, metadata["depth_metadata"]
        )

        # Create compressed frame object
        compressed_frame = KinectFrame(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            color_image=color_decompressed,
            depth_image=depth_decompressed,
        )

        # Save compressed result
        self.capture.save_frame(
            compressed_frame, sample_dir, f"compressed_frame{frame_num}"
        )

        # Save compression metadata
        metadata_path = sample_dir / f"metadata_frame{frame_num}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(
            f"📸 Saved compression sample: {level.value} - {metadata['total_compression_ratio']:.1f}x ratio"
        )

    def run_compression_test(
        self, levels: List[CompressionLevel], test_duration: int = 10
    ) -> List[Dict[str, Any]]:
        """Run compression test across multiple levels"""
        print("🚀 Starting Single Kinect Compression Test")
        print("=" * 60)

        # Start capture
        if not self.capture.start_capture():
            print("❌ Failed to start Kinect capture")
            return []

        # Print device info
        device_info = self.capture.get_device_info()
        print(f"\n📷 Device Info: {device_info.get('serial_number', 'Unknown')}")

        results = []

        try:
            for i, level in enumerate(levels):
                print(f"\n[{i+1}/{len(levels)}] Testing {level.value.upper()}")
                result = self.test_compression_level(level, test_duration)
                results.append(result)
                self.compression_results.append(result)

                # Brief pause between tests
                if i < len(levels) - 1:
                    print("  ⏸️  Pausing 2 seconds before next test...")
                    time.sleep(2)

        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")

        finally:
            self.capture.stop_capture()
            cv2.destroyAllWindows()

        # Save results
        self.save_test_results(results)

        return results

    def save_test_results(self, results: List[Dict[str, Any]]):
        """Save test results to JSON file"""
        results_file = self.output_dir / "compression_test_results.json"

        summary = {
            "test_timestamp": time.time(),
            "device_info": self.capture.get_device_info(),
            "results": results,
        }

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📊 Test results saved to: {results_file}")

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print test summary"""
        if not results:
            return

        print(f"\n📈 Compression Test Summary")
        print("=" * 60)
        print(
            f"{'Level':<10} {'Ratio':<8} {'Original':<12} {'Compressed':<12} {'Reduction'}"
        )
        print("-" * 60)

        for result in results:
            level = result["compression_level"]
            ratio = result["compression_ratio"]
            orig_bw = result["original_bandwidth_mbps"]
            comp_bw = result["compressed_bandwidth_mbps"]
            reduction = result["bandwidth_reduction"]

            print(
                f"{level:<10} {ratio:<8.1f} {orig_bw:<12.1f} {comp_bw:<12.1f} {reduction:<8.1f}x"
            )

        # Recommendations
        print(f"\n💡 Recommendations:")
        best_balanced = min(
            results, key=lambda x: abs(x["compressed_bandwidth_mbps"] - 50)
        )  # Target ~50 Mbps
        print(
            f"  • For balanced quality/bandwidth: {best_balanced['compression_level'].upper()}"
        )
        print(
            f"    ({best_balanced['compressed_bandwidth_mbps']:.1f} Mbps, {best_balanced['compression_ratio']:.1f}x compression)"
        )

        lowest_bandwidth = min(results, key=lambda x: x["compressed_bandwidth_mbps"])
        print(
            f"  • For lowest bandwidth: {lowest_bandwidth['compression_level'].upper()}"
        )
        print(
            f"    ({lowest_bandwidth['compressed_bandwidth_mbps']:.1f} Mbps, {lowest_bandwidth['compression_ratio']:.1f}x compression)"
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Kinect compression levels")
    parser.add_argument(
        "--duration", type=int, default=10, help="Test duration per level in seconds"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Kinect device ID")
    parser.add_argument(
        "--compression",
        type=str,
        default="all",
        choices=["all", "low", "medium", "high", "extreme"],
        help="Compression level(s) to test",
    )

    args = parser.parse_args()

    # Determine which compression levels to test
    if args.compression == "all":
        levels_to_test = [
            CompressionLevel.LOW,
            CompressionLevel.MEDIUM,
            CompressionLevel.HIGH,
            CompressionLevel.EXTREME,
        ]
    else:
        levels_to_test = [CompressionLevel(args.compression)]

    print("🚀 Azure Kinect Compression Test")
    print("=" * 50)
    print(f"📋 Testing levels: {[level.value for level in levels_to_test]}")
    print(f"⏱️  Duration per level: {args.duration}s")

    tester = CompressionTester(device_id=args.device_id)
    results = tester.run_compression_test(levels_to_test, args.duration)

    if results:
        tester.print_summary(results)
        print(f"\n✅ Compression test completed!")
        print(f"📁 Results saved to: {tester.output_dir}")
        print(f"📸 Sample images saved for quality comparison")
    else:
        print("❌ Compression test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
