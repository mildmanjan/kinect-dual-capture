#!/usr/bin/env python3
"""
Capture Scene Overlap Images
Save as: capture_overlap_test.py
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from utils.kinect_capture import BaseKinectCapture, KinectSyncMode

    KINECT_AVAILABLE = True
except ImportError:
    print("❌ Kinect utilities not available")
    KINECT_AVAILABLE = False


def capture_overlap_comparison():
    """Capture images from both Kinects to check overlap"""

    print("📷 Capturing Scene Overlap Test Images")
    print("=" * 50)

    if not KINECT_AVAILABLE:
        print("❌ Cannot capture - Kinect modules not available")
        return

    # Create output directory
    output_dir = Path("data/overlap_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize both captures
    capture0 = BaseKinectCapture(device_id=0, sync_mode=KinectSyncMode.STANDALONE)
    capture1 = BaseKinectCapture(device_id=1, sync_mode=KinectSyncMode.STANDALONE)

    print("🔌 Starting both Kinects...")

    if not capture0.start_capture():
        print("❌ Failed to start Device 0")
        return

    if not capture1.start_capture():
        print("❌ Failed to start Device 1")
        capture0.stop_capture()
        return

    print("✅ Both devices started")
    print("📸 Capturing synchronized frames...")

    try:
        # Capture several frames to get good ones
        for i in range(5):
            frame0 = capture0.capture_frame()
            frame1 = capture1.capture_frame()

            if frame0 is not None and frame1 is not None:
                if frame0.color_image is not None and frame1.color_image is not None:

                    timestamp = int(time.time())

                    # Save color images
                    color0_path = output_dir / f"device0_color_{timestamp}_{i}.jpg"
                    color1_path = output_dir / f"device1_color_{timestamp}_{i}.jpg"

                    cv2.imwrite(str(color0_path), frame0.color_image)
                    cv2.imwrite(str(color1_path), frame1.color_image)

                    print(
                        f"   ✅ Saved frame {i+1}: {color0_path.name} & {color1_path.name}"
                    )

                    # Create depth visualization
                    if frame0.depth_image is not None:
                        depth0_norm = cv2.normalize(
                            frame0.depth_image,
                            None,
                            0,
                            255,
                            cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U,
                        )
                        depth0_colored = cv2.applyColorMap(
                            depth0_norm, cv2.COLORMAP_JET
                        )
                        depth0_path = output_dir / f"device0_depth_{timestamp}_{i}.jpg"
                        cv2.imwrite(str(depth0_path), depth0_colored)

                    if frame1.depth_image is not None:
                        depth1_norm = cv2.normalize(
                            frame1.depth_image,
                            None,
                            0,
                            255,
                            cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U,
                        )
                        depth1_colored = cv2.applyColorMap(
                            depth1_norm, cv2.COLORMAP_JET
                        )
                        depth1_path = output_dir / f"device1_depth_{timestamp}_{i}.jpg"
                        cv2.imwrite(str(depth1_path), depth1_colored)

                    # Create side-by-side comparison
                    if i == 2:  # Use middle frame for comparison
                        create_comparison_image(
                            frame0.color_image,
                            frame1.color_image,
                            output_dir,
                            timestamp,
                        )

                    time.sleep(0.5)  # Brief pause between captures

    except Exception as e:
        print(f"❌ Error during capture: {e}")

    finally:
        capture0.stop_capture()
        capture1.stop_capture()
        print("🛑 Both devices stopped")

    print(f"\n📁 Images saved to: {output_dir}")
    print(f"💡 Compare device0_color_* and device1_color_* images")
    print(f"   Look for shared objects and scene overlap")


def create_comparison_image(color0, color1, output_dir, timestamp):
    """Create side-by-side comparison image"""

    try:
        # Resize images to same height for comparison
        height = 480
        width0 = int(color0.shape[1] * height / color0.shape[0])
        width1 = int(color1.shape[1] * height / color1.shape[0])

        resized0 = cv2.resize(color0, (width0, height))
        resized1 = cv2.resize(color1, (width1, height))

        # Create side-by-side image
        total_width = width0 + width1
        comparison = np.zeros((height, total_width, 3), dtype=np.uint8)

        comparison[:, :width0] = resized0
        comparison[:, width0:] = resized1

        # Add labels
        cv2.putText(
            comparison,
            "DEVICE 0",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            "DEVICE 1",
            (width0 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Add dividing line
        cv2.line(comparison, (width0, 0), (width0, height), (255, 255, 255), 2)

        # Save comparison
        comparison_path = output_dir / f"comparison_{timestamp}.jpg"
        cv2.imwrite(str(comparison_path), comparison)

        print(f"   📊 Created comparison: {comparison_path.name}")

    except Exception as e:
        print(f"   ⚠️  Could not create comparison: {e}")


def analyze_saved_images():
    """Provide analysis guidance for saved images"""

    print(f"\n🔍 Image Analysis Guide")
    print("=" * 50)
    print("Open the saved images and look for:")
    print("")
    print("✅ GOOD OVERLAP (60-80%):")
    print("   • Same room/area visible in both images")
    print("   • Same furniture/objects from different angles")
    print("   • Clear corresponding features")
    print("   • Different perspectives of same scene")
    print("")
    print("❌ POOR OVERLAP (<60%):")
    print("   • Completely different rooms/areas")
    print("   • No shared objects or features")
    print("   • One looking at wall, other at room")
    print("   • Very different scenes entirely")
    print("")
    print("⚠️  NOISE ISSUES:")
    print("   • Depth images very speckled/noisy")
    print("   • Missing depth data (black areas)")
    print("   • Strange artifacts in depth view")
    print("")
    print("🎯 NEXT STEPS:")
    print("   If overlap good → Clean lenses, improve lighting, recalibrate")
    print("   If overlap poor → Reposition Kinects for better overlap")


def main():
    """Main capture function"""

    print("🎯 Scene Overlap Analysis")
    print("=" * 60)
    print("This will capture images from both Kinects to check scene overlap")
    print("")

    # Capture comparison images
    capture_overlap_comparison()

    # Provide analysis guidance
    analyze_saved_images()

    print(f"\n✅ Overlap test complete!")
    print(f"Check the images in data/overlap_test/ to see what each Kinect sees")


if __name__ == "__main__":
    main()
