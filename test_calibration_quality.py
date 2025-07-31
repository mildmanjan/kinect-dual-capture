#!/usr/bin/env python3
"""
Test Calibration Quality
Save as: test_calibration_quality.py
"""

import json
import numpy as np
from pathlib import Path
import sys


def analyze_calibration():
    """Analyze the calibration file"""

    calib_file = Path("config/dual_kinect_calibration.json")

    if not calib_file.exists():
        print("❌ No calibration file found")
        return

    try:
        with open(calib_file, "r") as f:
            calib_data = json.load(f)

        print("📊 Calibration Analysis")
        print("=" * 40)

        # Basic info
        error = calib_data.get("calibration_error", "Unknown")
        method = calib_data.get("calibration_method", "Unknown")
        frames = calib_data.get("num_calibration_frames", "Unknown")

        print(f"Error: {error}mm")
        print(f"Method: {method}")
        print(f"Frames used: {frames}")

        # Check transformation matrix
        transform = calib_data.get("transformation_matrix", None)

        if transform is None:
            print("❌ No transformation matrix found")
            return

        transform_np = np.array(transform)
        print(f"\n🔧 Transformation Matrix:")
        print(transform_np)

        # Check if it's identity matrix (bad)
        identity = np.eye(4)
        is_identity = np.allclose(transform_np, identity, atol=1e-6)

        if is_identity:
            print("❌ PROBLEM: Identity matrix - no real calibration!")
            print("   This means the Kinects are assumed to be in identical positions")
            print("   Calibration failed or wasn't performed properly")
        else:
            print("✅ Real transformation detected")

            # Extract translation (how far apart the Kinects are)
            translation = transform_np[:3, 3]
            distance = np.linalg.norm(translation)

            print(f"\n📏 Kinect Separation:")
            print(f"   X offset: {translation[0]:.3f}m")
            print(f"   Y offset: {translation[1]:.3f}m")
            print(f"   Z offset: {translation[2]:.3f}m")
            print(f"   Total distance: {distance:.3f}m")

            # Check if distance makes sense (should be 0.1-1.0m typically)
            if distance < 0.05:
                print("⚠️  Very small separation - might be calibration error")
            elif distance > 2.0:
                print("⚠️  Very large separation - check calibration")
            else:
                print("✅ Reasonable Kinect separation")

        # Check rotation matrix
        rotation = transform_np[:3, :3]

        # A rotation matrix should have determinant = 1
        det = np.linalg.det(rotation)
        print(f"\n🔄 Rotation Analysis:")
        print(f"   Determinant: {det:.6f} (should be ~1.0)")

        if abs(det - 1.0) > 0.1:
            print("⚠️  Rotation matrix might be invalid")
        else:
            print("✅ Valid rotation matrix")

        # Calculate rotation angles
        # This is a simplified extraction - works for small rotations
        if not is_identity:
            rx = np.arctan2(rotation[2, 1], rotation[2, 2]) * 180 / np.pi
            ry = (
                np.arctan2(
                    -rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)
                )
                * 180
                / np.pi
            )
            rz = np.arctan2(rotation[1, 0], rotation[0, 0]) * 180 / np.pi

            print(f"   Rotation X: {rx:.1f}°")
            print(f"   Rotation Y: {ry:.1f}°")
            print(f"   Rotation Z: {rz:.1f}°")

            total_rotation = np.sqrt(rx**2 + ry**2 + rz**2)
            print(f"   Total rotation: {total_rotation:.1f}°")

            if total_rotation > 45:
                print("⚠️  Large rotation - check Kinect alignment")

    except Exception as e:
        print(f"❌ Error analyzing calibration: {e}")


def recommend_next_steps():
    """Recommend what to do based on analysis"""

    print(f"\n💡 Recommendations:")
    print("=" * 40)

    calib_file = Path("config/dual_kinect_calibration.json")

    if not calib_file.exists():
        print("1. Run calibration first:")
        print("   python dual_kinect_calibration.py --auto")
        return

    try:
        with open(calib_file, "r") as f:
            calib_data = json.load(f)

        transform = np.array(calib_data.get("transformation_matrix", np.eye(4)))
        is_identity = np.allclose(transform, np.eye(4), atol=1e-6)
        error = calib_data.get("calibration_error", 999)

        if is_identity:
            print("🔧 Identity matrix detected - need real calibration:")
            print("   1. python dual_kinect_calibration.py --recalibrate")
            print("   2. Ensure good scene overlap between Kinects")
            print("   3. Add textured objects in view of both devices")

        elif error == 0.0:
            print("🤔 Zero error might indicate:")
            print("   1. Perfect calibration (rare)")
            print("   2. Failed error calculation")
            print("   3. Test with fusion to see visual quality:")
            print(
                "      python src\\step5_kinect_fusion.py --mode realtime --duration 10"
            )

        else:
            print("✅ Calibration looks reasonable")
            print("   Test fusion quality and adjust if needed")

    except Exception as e:
        print(f"Error reading calibration: {e}")


def main():
    """Main analysis"""
    analyze_calibration()
    recommend_next_steps()


if __name__ == "__main__":
    main()
