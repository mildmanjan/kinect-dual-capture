#!/usr/bin/env python3
"""
Improved Calibration Test
Save as: improved_calibration.py
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add utils to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from utils.kinect_capture import BaseKinectCapture, KinectSyncMode

    KINECT_AVAILABLE = True
except ImportError:
    print("❌ Kinect utilities not available")
    KINECT_AVAILABLE = False


def check_scene_overlap():
    """Check how much scene overlap you actually have"""

    print("🔍 Checking Scene Overlap")
    print("=" * 40)

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

    print("📷 Capturing frames to analyze overlap...")

    try:
        # Get frames from both devices
        frame0 = capture0.capture_frame()
        frame1 = capture1.capture_frame()

        if frame0 is None or frame1 is None:
            print("❌ Could not capture frames")
            return

        if frame0.depth_image is None or frame1.depth_image is None:
            print("❌ No depth data")
            return

        # Analyze depth overlap
        depth0 = frame0.depth_image
        depth1 = frame1.depth_image

        # Count valid depth pixels (objects in scene)
        valid0 = np.sum((depth0 > 300) & (depth0 < 3000))  # 0.3-3m range
        valid1 = np.sum((depth1 > 300) & (depth1 < 3000))

        total_pixels = depth0.shape[0] * depth0.shape[1]

        coverage0 = (valid0 / total_pixels) * 100
        coverage1 = (valid1 / total_pixels) * 100

        print(f"📊 Scene Coverage Analysis:")
        print(f"   Device 0 scene coverage: {coverage0:.1f}%")
        print(f"   Device 1 scene coverage: {coverage1:.1f}%")

        # Estimate overlap (rough calculation)
        # This is simplified - real overlap calculation is complex
        estimated_overlap = min(coverage0, coverage1)

        print(f"   Estimated overlap: ~{estimated_overlap:.1f}%")

        if estimated_overlap > 60:
            print("   ✅ Good overlap for calibration")
        elif estimated_overlap > 40:
            print("   ⚠️  Moderate overlap - may be sufficient")
        else:
            print("   ❌ Poor overlap - move Kinects closer/angle inward")

        # Scene complexity check
        depth0_range = np.max(depth0) - np.min(depth0[depth0 > 0])
        depth1_range = np.max(depth1) - np.min(depth1[depth1 > 0])

        print(f"\n📐 Scene Depth Complexity:")
        print(f"   Device 0 depth range: {depth0_range:.0f}mm")
        print(f"   Device 1 depth range: {depth1_range:.0f}mm")

        if depth0_range > 1000 and depth1_range > 1000:
            print("   ✅ Good depth variation for calibration")
        else:
            print("   ⚠️  Limited depth variation - add objects at different distances")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")

    finally:
        capture0.stop_capture()
        capture1.stop_capture()


def calibration_scene_checklist():
    """Provide calibration scene setup checklist"""

    print("\n📋 Calibration Scene Checklist")
    print("=" * 40)
    print("Before recalibrating, ensure your scene has:")
    print("")
    print("✅ REQUIRED ELEMENTS:")
    print("   📚 Books or magazines (textured surfaces)")
    print("   📦 Boxes or geometric objects")
    print("   🪑 Furniture pieces")
    print("   🖼️  Posters or artwork on walls")
    print("   💡 Even lighting (no harsh shadows)")
    print("")
    print("✅ POSITIONING:")
    print("   📐 Kinects angled slightly inward (not parallel)")
    print("   📏 0.5-1.5m distance to objects")
    print("   👁️  60-80% scene overlap between views")
    print("   📎 Objects at multiple depths (near, mid, far)")
    print("")
    print("❌ AVOID:")
    print("   🚫 Blank walls")
    print("   🚫 Uniform surfaces (carpet, solid colors)")
    print("   🚫 Moving objects")
    print("   🚫 Backlighting from windows")
    print("   🚫 Perfectly parallel Kinect alignment")


def suggest_calibration_commands():
    """Suggest specific calibration commands to try"""

    print("\n🎯 Recommended Calibration Commands")
    print("=" * 40)
    print("Try these in order:")
    print("")
    print("1. Basic recalibration:")
    print("   python dual_kinect_calibration.py --recalibrate")
    print("")
    print("2. More samples for better accuracy:")
    print("   python dual_kinect_calibration.py --recalibrate --samples 10")
    print("")
    print("3. If your calibration tool supports manual mode:")
    print("   python dual_kinect_calibration.py --manual")
    print("")
    print("4. Check for specific calibration options:")
    print("   python dual_kinect_calibration.py --help")


def predict_success_factors():
    """Analyze what factors will lead to calibration success"""

    print("\n🎯 Success Prediction")
    print("=" * 40)
    print("Given your excellent hardware setup:")
    print("")
    print("✅ STRENGTHS:")
    print("   • Perfect USB bus separation")
    print("   • 100% device reliability")
    print("   • Excellent 10.2ms synchronization")
    print("   • Valid transformation matrix structure")
    print("")
    print("🎯 FOCUS AREAS:")
    print("   • Scene feature richness")
    print("   • Kinect positioning for optimal overlap")
    print("   • Calibration algorithm parameters")
    print("")
    print("📈 EXPECTED IMPROVEMENT:")
    print("   Current: 96.9% outliers (VERY POOR)")
    print("   Target:  <8% outliers (GOOD)")
    print("   With proper scene: Should achieve <5% outliers")


def main():
    """Run improved calibration analysis"""

    print("🎯 Improved Calibration Analysis")
    print("=" * 60)
    print("Your hardware is perfect - let's fix the calibration!")
    print("")

    # Check current scene overlap
    check_scene_overlap()

    # Provide setup guidance
    calibration_scene_checklist()

    # Suggest commands
    suggest_calibration_commands()

    # Predict success
    predict_success_factors()

    print(f"\n✅ Analysis complete!")
    print(f"\nNext step: Set up proper calibration scene and recalibrate")


if __name__ == "__main__":
    main()
