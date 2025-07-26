#!/usr/bin/env python3
"""
Measure Alignment Quality
Save as: measure_alignment.py in project root
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import json


def find_latest_fusion_file():
    """Find the most recent fusion result file"""
    fusion_dir = Path("data/fusion_results")

    if not fusion_dir.exists():
        print("❌ No fusion_results directory found")
        print(
            "💡 Run: python src\\step5_kinect_fusion.py --mode sequence --duration 5 --export"
        )
        return None

    # Look for PLY files (they preserve colors)
    ply_files = list(fusion_dir.glob("*.ply"))

    if not ply_files:
        print("❌ No .ply files found in fusion_results")
        print(
            "💡 Run: python src\\step5_kinect_fusion.py --mode sequence --duration 5 --export"
        )
        return None

    # Get the most recent file
    latest_file = max(ply_files, key=lambda f: f.stat().st_mtime)

    print(f"📁 Analyzing: {latest_file.name}")
    return latest_file


def measure_calibration_quality(ply_file):
    """Measure point cloud quality metrics"""

    try:
        # Load point cloud
        print("📖 Loading point cloud...")
        pcd = o3d.io.read_point_cloud(str(ply_file))

        if len(pcd.points) == 0:
            print("❌ Point cloud is empty")
            return

        print(f"📊 Basic Stats:")
        print(f"   Total points: {len(pcd.points):,}")
        print(f"   Has colors: {len(pcd.colors) > 0}")
        print(f"   Has normals: {len(pcd.normals) > 0}")

        # Get point cloud bounds
        points = np.asarray(pcd.points)
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        size = max_bounds - min_bounds

        print(f"\n📏 Point Cloud Dimensions:")
        print(f"   Width (X):  {size[0]:.3f}m")
        print(f"   Height (Y): {size[1]:.3f}m")
        print(f"   Depth (Z):  {size[2]:.3f}m")
        print(
            f"   Center: ({(min_bounds + max_bounds)[0]/2:.3f}, {(min_bounds + max_bounds)[1]/2:.3f}, {(min_bounds + max_bounds)[2]/2:.3f})"
        )

        # Statistical outlier detection (main quality metric)
        print(f"\n🔍 Quality Analysis:")
        print("   Detecting outliers...")

        pcd_clean, outlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        outlier_count = len(outlier_indices)
        outlier_percentage = (outlier_count / len(pcd.points)) * 100

        print(f"   Outlier points: {outlier_count:,} ({outlier_percentage:.1f}%)")

        # Quality assessment based on outlier percentage
        if outlier_percentage < 3:
            quality = "EXCELLENT ✅"
            calibration_assessment = "Perfect calibration - minimal noise"
        elif outlier_percentage < 8:
            quality = "GOOD ✅"
            calibration_assessment = "Good calibration - minor alignment issues"
        elif outlier_percentage < 15:
            quality = "FAIR ⚠️"
            calibration_assessment = "Moderate calibration - noticeable misalignment"
        elif outlier_percentage < 25:
            quality = "POOR ❌"
            calibration_assessment = "Poor calibration - significant problems"
        else:
            quality = "VERY POOR ❌"
            calibration_assessment = "Failed calibration - major misalignment"

        print(f"   Quality: {quality}")
        print(f"   Assessment: {calibration_assessment}")

        # Density analysis
        print(f"\n📐 Density Analysis:")

        # Calculate average distance to nearest neighbors
        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)

        print(f"   Avg neighbor distance: {avg_distance:.4f}m")
        print(f"   Distance std dev: {std_distance:.4f}m")

        # Points per cubic meter (rough estimate)
        volume = size[0] * size[1] * size[2]
        if volume > 0:
            density = len(pcd.points) / volume
            print(f"   Point density: {density:.0f} points/m³")

        # Uniformity check (coefficient of variation)
        cv = std_distance / avg_distance if avg_distance > 0 else 0
        print(f"   Uniformity (CV): {cv:.3f} (lower is better)")

        if cv < 0.5:
            print("   ✅ Very uniform point distribution")
        elif cv < 1.0:
            print("   ✅ Good point distribution")
        elif cv < 2.0:
            print("   ⚠️  Moderate point distribution")
        else:
            print("   ❌ Poor point distribution - check calibration")

        # Planar surface analysis (if walls/floors are visible)
        print(f"\n🏠 Surface Analysis:")
        try:
            # Try to detect planar surfaces
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.01,  # 1cm threshold
                ransac_n=3,
                num_iterations=1000,
            )

            inlier_percentage = (len(inliers) / len(pcd.points)) * 100
            print(
                f"   Largest plane: {len(inliers):,} points ({inlier_percentage:.1f}%)"
            )

            if inlier_percentage > 20:
                print("   ✅ Good planar surfaces detected (walls/floor)")
            elif inlier_percentage > 10:
                print("   ⚠️  Some planar surfaces detected")
            else:
                print("   ❌ Few planar surfaces - complex scene or poor alignment")

        except Exception as e:
            print(f"   ⚠️  Could not analyze surfaces: {e}")

        # Overall recommendation
        print(f"\n💡 Recommendations:")
        print("=" * 50)

        if outlier_percentage < 8:
            print("✅ Calibration quality is good!")
            print("   - Ready for production use")
            print(
                "   - Consider mesh generation: python src\\step6_kinect_to_mesh_demo.py"
            )
            print("   - Try web streaming: python src\\step7_kinect_web_streaming.py")

        elif outlier_percentage < 15:
            print("⚠️  Calibration could be improved:")
            print("   - Try recalibration with better scene overlap")
            print("   - Ensure static environment during calibration")
            print("   - Check USB bus separation still working")

        else:
            print("❌ Calibration needs significant improvement:")
            print("   - Recalibrate: python dual_kinect_calibration.py --recalibrate")
            print("   - Check Kinect positioning (60-80% overlap)")
            print("   - Verify USB 3.0 on separate buses")
            print("   - Ensure good lighting and textured scene")

        return {
            "total_points": len(pcd.points),
            "outlier_percentage": outlier_percentage,
            "quality": quality,
            "avg_neighbor_distance": avg_distance,
            "uniformity_cv": cv,
            "bounds": size.tolist(),
        }

    except Exception as e:
        print(f"❌ Error analyzing point cloud: {e}")
        import traceback

        traceback.print_exc()
        return None


def save_analysis_report(results, ply_file):
    """Save analysis results to JSON"""
    if results is None:
        return

    report = {
        "analysis_timestamp": ply_file.stat().st_mtime,
        "analyzed_file": ply_file.name,
        "metrics": results,
    }

    report_file = Path("data/fusion_results/alignment_analysis.json")

    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Analysis saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")


def main():
    """Main analysis function"""
    print("🔍 Kinect Fusion Alignment Analysis")
    print("=" * 50)

    # Find latest fusion file
    ply_file = find_latest_fusion_file()
    if ply_file is None:
        return

    # Analyze quality
    results = measure_calibration_quality(ply_file)

    # Save report
    if results:
        save_analysis_report(results, ply_file)

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
