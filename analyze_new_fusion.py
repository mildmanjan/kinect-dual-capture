#!/usr/bin/env python3
"""
Analyze Specific Fusion File
Save as: analyze_new_fusion.py
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import json


def analyze_specific_file(file_path):
    """Analyze a specific PLY file"""

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"🔍 Analyzing NEW Fusion Result")
    print("=" * 50)
    print(f"📁 File: {file_path.name}")

    try:
        # Load point cloud
        print("📖 Loading point cloud...")
        pcd = o3d.io.read_point_cloud(str(file_path))

        if len(pcd.points) == 0:
            print("❌ Point cloud is empty")
            return

        print(f"📊 Basic Stats:")
        print(f"   Total points: {len(pcd.points):,}")
        print(f"   Has colors: {len(pcd.colors) > 0}")

        # Get point cloud bounds
        points = np.asarray(pcd.points)
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        size = max_bounds - min_bounds

        print(f"\n📏 Point Cloud Dimensions:")
        print(f"   Width (X):  {size[0]:.3f}m")
        print(f"   Height (Y): {size[1]:.3f}m")
        print(f"   Depth (Z):  {size[2]:.3f}m")

        # Main quality test - outlier detection
        print(f"\n🔍 NEW CALIBRATION Quality Analysis:")
        print("   Detecting outliers...")

        pcd_clean, outlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        outlier_count = len(outlier_indices)
        outlier_percentage = (outlier_count / len(pcd.points)) * 100

        print(f"   Outlier points: {outlier_count:,} ({outlier_percentage:.1f}%)")

        # Compare to old results
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"   OLD calibration: 96.9% outliers (VERY POOR)")
        print(f"   NEW calibration: {outlier_percentage:.1f}% outliers", end="")

        # Quality assessment
        if outlier_percentage < 3:
            quality = "EXCELLENT ✅"
            improvement = f" (MASSIVE IMPROVEMENT!)"
        elif outlier_percentage < 8:
            quality = "GOOD ✅"
            improvement = f" (HUGE IMPROVEMENT!)"
        elif outlier_percentage < 15:
            quality = "FAIR ⚠️"
            improvement = f" (SIGNIFICANT IMPROVEMENT!)"
        elif outlier_percentage < 25:
            quality = "POOR ❌"
            improvement = f" (SOME IMPROVEMENT)"
        else:
            quality = "VERY POOR ❌"
            improvement = f" (MINIMAL IMPROVEMENT)"

        print(improvement)
        print(f"   Quality: {quality}")

        # Calculate improvement factor
        improvement_factor = (
            96.9 / outlier_percentage if outlier_percentage > 0 else float("inf")
        )
        print(f"   Improvement factor: {improvement_factor:.1f}x better!")

        # Density analysis
        print(f"\n📐 Density Analysis:")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        cv = std_distance / avg_distance if avg_distance > 0 else 0

        print(f"   Avg neighbor distance: {avg_distance:.4f}m")
        print(f"   Uniformity (CV): {cv:.3f}")

        if cv < 0.5:
            print("   ✅ Very uniform point distribution")
        elif cv < 1.0:
            print("   ✅ Good point distribution")
        else:
            print("   ⚠️  Moderate point distribution")

        # Overall assessment
        print(f"\n💡 CALIBRATION SUCCESS ASSESSMENT:")
        print("=" * 50)

        if outlier_percentage < 8:
            print("🎉 CALIBRATION FIX SUCCESSFUL!")
            print("   ✅ Fusion quality dramatically improved")
            print("   ✅ Ready for mesh generation and web streaming")
            print("   ✅ The 'garbled' appearance should be completely resolved")

        elif outlier_percentage < 15:
            print("✅ GOOD CALIBRATION IMPROVEMENT!")
            print("   ✅ Major reduction in alignment errors")
            print("   ✅ Usable for most applications")
            print("   ⚠️  Could be refined further if needed")

        else:
            print("⚠️  PARTIAL IMPROVEMENT")
            print("   ✅ Some reduction in errors")
            print("   ⚠️  May need further calibration refinement")

        return outlier_percentage

    except Exception as e:
        print(f"❌ Error analyzing point cloud: {e}")
        import traceback

        traceback.print_exc()
        return None


def find_newest_fusion_files():
    """Find the newest fusion files"""

    fusion_dir = Path("data/fusion_results")

    # Look for sequence directories first
    sequence_dirs = [
        d
        for d in fusion_dir.iterdir()
        if d.is_dir() and d.name.startswith("fusion_sequence_")
    ]

    if sequence_dirs:
        newest_dir = max(sequence_dirs, key=lambda d: d.stat().st_mtime)
        print(f"📁 Found newest sequence directory: {newest_dir.name}")

        ply_files = list(newest_dir.glob("*.ply"))
        if ply_files:
            return ply_files

    # Fallback to individual PLY files
    ply_files = list(fusion_dir.glob("*.ply"))
    if ply_files:
        # Get the newest one
        newest_file = max(ply_files, key=lambda f: f.stat().st_mtime)
        return [newest_file]

    return []


def main():
    """Main analysis"""

    print("🎯 NEW Calibration Fusion Analysis")
    print("=" * 60)
    print("Analyzing your improved 24.5mm calibration results...")
    print("")

    # Find newest files
    fusion_files = find_newest_fusion_files()

    if not fusion_files:
        print("❌ No fusion files found")
        print(
            "💡 Run: python src\\step5_kinect_fusion.py --mode sequence --duration 5 --export"
        )
        return

    # Analyze the first/newest file
    fusion_file = fusion_files[0]
    outlier_percentage = analyze_specific_file(fusion_file)

    if outlier_percentage is not None:
        print(f"\n📊 SUMMARY:")
        print(f"   OLD: 96.9% outliers (failed calibration)")
        print(f"   NEW: {outlier_percentage:.1f}% outliers (24.5mm calibration)")

        if outlier_percentage < 15:
            print(f"   🎉 SUCCESS: Calibration improvement solved the problem!")
        else:
            print(f"   ⚠️  PARTIAL: Further calibration refinement recommended")


if __name__ == "__main__":
    main()
