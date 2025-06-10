#!/usr/bin/env python3
"""
Improved Dual Kinect Calibration Tool
Better handling of depth data and 3D point generation

File location: improved_dual_kinect_calibration.py (project root)
"""

import numpy as np
import cv2
import open3d as o3d
import time
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent / "src"))
from utils.kinect_capture import BaseKinectCapture, KinectSyncMode, KinectFrame


class ImprovedDualKinectCalibrator:
    """Improved calibrator with better depth handling"""

    def __init__(self, device1_id: int = 0, device2_id: int = 1):
        self.device1_id = device1_id
        self.device2_id = device2_id

        # Kinect captures
        self.capture1 = BaseKinectCapture(
            device_id=device1_id, sync_mode=KinectSyncMode.STANDALONE
        )
        self.capture2 = BaseKinectCapture(
            device_id=device2_id, sync_mode=KinectSyncMode.STANDALONE
        )

        # Camera intrinsics (approximate for Azure Kinect depth camera)
        self.intrinsics = {
            "fx": 504.0,
            "fy": 504.0,
            "cx": 320.0,
            "cy": 288.0,
            "width": 640,
            "height": 576,
        }

        # Calibration data storage
        self.calibration_frames = []

        # Output
        self.output_dir = Path("data/calibration")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_and_preview_frames(
        self,
    ) -> Tuple[Optional[KinectFrame], Optional[KinectFrame]]:
        """Capture frames with live preview to help positioning"""
        print("📸 Starting capture preview...")
        print("Position your calibration object so both cameras can see it clearly")
        print("Press SPACE to capture when ready, 'q' to skip")

        # Start both captures
        if not self.capture1.start_capture():
            print("❌ Failed to start device 1")
            return None, None

        if not self.capture2.start_capture():
            print("❌ Failed to start device 2")
            self.capture1.stop_capture()
            return None, None

        frame1, frame2 = None, None

        try:
            while True:
                # Get current frames
                current_frame1 = self.capture1.capture_frame()
                current_frame2 = self.capture2.capture_frame()

                if current_frame1 is not None and current_frame2 is not None:
                    # Show preview
                    self.show_capture_preview(current_frame1, current_frame2)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):  # Space to capture
                        frame1, frame2 = current_frame1, current_frame2
                        print("✅ Frame pair captured!")
                        break
                    elif key == ord("q"):  # Quit
                        print("⏭️  Skipping capture")
                        break

                time.sleep(1 / 30)  # 30 FPS preview

        finally:
            self.capture1.stop_capture()
            self.capture2.stop_capture()
            cv2.destroyAllWindows()

        return frame1, frame2

    def show_capture_preview(self, frame1: KinectFrame, frame2: KinectFrame):
        """Show live preview of both cameras"""
        if frame1.color_image is None or frame2.color_image is None:
            return

        # Resize for display
        display_height = 300
        display_width = int(display_height * 16 / 9)

        color1_resized = cv2.resize(frame1.color_image, (display_width, display_height))
        color2_resized = cv2.resize(frame2.color_image, (display_width, display_height))

        # Add labels
        cv2.putText(
            color1_resized,
            f"Device {self.device1_id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            color2_resized,
            f"Device {self.device2_id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Show depth quality indicators
        depth1_valid = np.sum((frame1.depth_image > 300) & (frame1.depth_image < 3000))
        depth2_valid = np.sum((frame2.depth_image > 300) & (frame2.depth_image < 3000))

        cv2.putText(
            color1_resized,
            f"Depth pixels: {depth1_valid}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        cv2.putText(
            color2_resized,
            f"Depth pixels: {depth2_valid}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        # Combine views
        combined = np.hstack([color1_resized, color2_resized])

        # Add instructions
        cv2.putText(
            combined,
            "Position calibration object visible to both cameras",
            (10, display_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            "Press SPACE to capture, 'q' to skip",
            (10, display_height + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Calibration Preview", combined)

    def detect_features_with_depth_validation(
        self, frame1: KinectFrame, frame2: KinectFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect features and validate they have good depth data"""
        if frame1.color_image is None or frame2.color_image is None:
            return np.array([]), np.array([])

        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1.color_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2.color_image, cv2.COLOR_BGR2GRAY)

        # Create depth masks (areas with valid depth)
        depth_mask1 = (frame1.depth_image > 300) & (frame1.depth_image < 3000)
        depth_mask2 = (frame2.depth_image > 300) & (frame2.depth_image < 3000)

        # Resize depth masks to match color image if necessary
        if depth_mask1.shape != gray1.shape:
            depth_mask1 = cv2.resize(
                depth_mask1.astype(np.uint8), (gray1.shape[1], gray1.shape[0])
            ).astype(bool)
        if depth_mask2.shape != gray2.shape:
            depth_mask2 = cv2.resize(
                depth_mask2.astype(np.uint8), (gray2.shape[1], gray2.shape[0])
            ).astype(bool)

        # Detect features only in areas with valid depth
        orb = cv2.ORB_create(nfeatures=2000)

        # Apply depth masks to feature detection
        kp1, desc1 = orb.detectAndCompute(gray1, depth_mask1.astype(np.uint8))
        kp2, desc2 = orb.detectAndCompute(gray2, depth_mask2.astype(np.uint8))

        print(f"   Found {len(kp1)} features in device 1 (with valid depth)")
        print(f"   Found {len(kp2)} features in device 2 (with valid depth)")

        if desc1 is None or desc2 is None or len(kp1) < 20 or len(kp2) < 20:
            print("⚠️  Insufficient features detected in valid depth areas")
            return np.array([]), np.array([])

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            print(f"⚠️  Only {len(matches)} matches found (need at least 10)")
            return np.array([]), np.array([])

        # Filter matches by distance
        distance_threshold = np.mean([m.distance for m in matches]) + np.std(
            [m.distance for m in matches]
        )
        filtered_matches = [m for m in matches if m.distance < distance_threshold]

        # Use best matches
        good_matches = filtered_matches[: min(50, len(filtered_matches))]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        print(f"✅ Found {len(good_matches)} good feature correspondences")
        return pts1, pts2

    def pixels_to_3d_points_improved(
        self, pixel_points: np.ndarray, depth_image: np.ndarray, color_shape: tuple
    ) -> np.ndarray:
        """Improved conversion with better depth handling"""
        points_3d = []

        # Get depth image dimensions
        depth_height, depth_width = depth_image.shape
        color_height, color_width = color_shape[:2]

        # Calculate scaling factors if color and depth resolutions differ
        scale_x = depth_width / color_width
        scale_y = depth_height / color_height

        for pt in pixel_points:
            # Convert color coordinates to depth coordinates
            u_color, v_color = pt[0], pt[1]
            u_depth = int(u_color * scale_x)
            v_depth = int(v_color * scale_y)

            # Check bounds
            if 0 <= u_depth < depth_width and 0 <= v_depth < depth_height:
                # Sample depth in a small neighborhood to be more robust
                u_start = max(0, u_depth - 2)
                u_end = min(depth_width, u_depth + 3)
                v_start = max(0, v_depth - 2)
                v_end = min(depth_height, v_depth + 3)

                depth_patch = depth_image[v_start:v_end, u_start:u_end]
                valid_depths = depth_patch[(depth_patch > 300) & (depth_patch < 3000)]

                if len(valid_depths) > 0:
                    # Use median depth for robustness
                    depth_value = np.median(valid_depths)

                    # Convert to 3D using camera intrinsics (use depth coordinates)
                    z = depth_value / 1000.0  # Convert to meters
                    x = (u_depth - self.intrinsics["cx"]) * z / self.intrinsics["fx"]
                    y = (v_depth - self.intrinsics["cy"]) * z / self.intrinsics["fy"]

                    points_3d.append([x, y, z])

        return np.array(points_3d)

    def validate_3d_points(
        self, points_3d: np.ndarray, min_distance: float = 0.01
    ) -> np.ndarray:
        """Remove 3D points that are too close to each other"""
        if len(points_3d) < 2:
            return points_3d

        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(points_3d))

        # Keep points that are at least min_distance apart
        valid_indices = []
        for i in range(len(points_3d)):
            if i == 0:
                valid_indices.append(i)
            else:
                # Check if this point is far enough from all previously selected points
                min_dist_to_selected = min([distances[i][j] for j in valid_indices])
                if min_dist_to_selected >= min_distance:
                    valid_indices.append(i)

        return points_3d[valid_indices]

    def collect_improved_calibration_data(self, num_captures: int = 5) -> bool:
        """Collect calibration data with improved depth handling"""
        print(f"🎯 Collecting {num_captures} calibration frame pairs...")
        print("\nCalibration tips:")
        print("  1. Use a textured object (book, magazine, patterned board)")
        print("  2. Hold object 0.5-2 meters from cameras")
        print("  3. Ensure object is well-lit")
        print("  4. Object should be visible to BOTH cameras")
        print("  5. Hold steady during capture")

        successful_captures = 0

        for i in range(num_captures):
            print(f"\n📸 Capture {i+1}/{num_captures}")

            # Capture with preview
            frame1, frame2 = self.capture_and_preview_frames()

            if frame1 is None or frame2 is None:
                print("❌ No frames captured, skipping...")
                continue

            # Detect features with depth validation
            pts1, pts2 = self.detect_features_with_depth_validation(frame1, frame2)

            if len(pts1) < 10:
                print("❌ Insufficient features detected, skipping...")
                continue

            # Convert to 3D points with improved method
            points_3d_1 = self.pixels_to_3d_points_improved(
                pts1, frame1.depth_image, frame1.color_image.shape
            )
            points_3d_2 = self.pixels_to_3d_points_improved(
                pts2, frame2.depth_image, frame2.color_image.shape
            )

            print(
                f"   Raw 3D points: Device1={len(points_3d_1)}, Device2={len(points_3d_2)}"
            )

            if len(points_3d_1) < 8 or len(points_3d_2) < 8:
                print("❌ Insufficient 3D points, try different object position...")
                print(
                    "   Tips: Move object closer, ensure good lighting, use more textured object"
                )
                continue

            # Validate and filter 3D points
            points_3d_1 = self.validate_3d_points(points_3d_1)
            points_3d_2 = self.validate_3d_points(points_3d_2)

            # Ensure we have the same number of points
            min_points = min(len(points_3d_1), len(points_3d_2))
            if min_points < 8:
                print(f"❌ After filtering: only {min_points} valid point pairs")
                continue

            points_3d_1 = points_3d_1[:min_points]
            points_3d_2 = points_3d_2[:min_points]

            # Store calibration data
            self.calibration_frames.append(
                {
                    "frame1": frame1,
                    "frame2": frame2,
                    "points_3d_1": points_3d_1,
                    "points_3d_2": points_3d_2,
                    "timestamp": time.time(),
                }
            )

            successful_captures += 1
            print(
                f"✅ Capture {successful_captures} successful ({len(points_3d_1)} point pairs)"
            )

            # Save debug images
            self.save_debug_images(frame1, frame2, pts1, pts2, i + 1)

        print(f"\n📊 Collected {successful_captures} successful calibration captures")
        return successful_captures >= 3

    def save_debug_images(
        self,
        frame1: KinectFrame,
        frame2: KinectFrame,
        pts1: np.ndarray,
        pts2: np.ndarray,
        capture_num: int,
    ):
        """Save debug images showing detected features"""
        if len(pts1) == 0 or len(pts2) == 0:
            return

        debug_dir = self.output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

        # Draw features on color images
        img1_debug = frame1.color_image.copy()
        img2_debug = frame2.color_image.copy()

        for pt in pts1:
            cv2.circle(img1_debug, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        for pt in pts2:
            cv2.circle(img2_debug, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # Save debug images
        cv2.imwrite(
            str(debug_dir / f"capture_{capture_num}_device1_features.jpg"), img1_debug
        )
        cv2.imwrite(
            str(debug_dir / f"capture_{capture_num}_device2_features.jpg"), img2_debug
        )

        print(f"   🔍 Debug images saved to {debug_dir}")

    def compute_transformation_matrix(self) -> Tuple[np.ndarray, float]:
        """Compute transformation matrix from collected data"""
        if len(self.calibration_frames) < 3:
            print("❌ Need at least 3 calibration frames")
            return np.eye(4), float("inf")

        print("🔧 Computing transformation matrix...")

        # Collect all point correspondences
        all_points_1 = []
        all_points_2 = []

        for calib_data in self.calibration_frames:
            all_points_1.extend(calib_data["points_3d_1"])
            all_points_2.extend(calib_data["points_3d_2"])

        points_1 = np.array(all_points_1)
        points_2 = np.array(all_points_2)

        print(f"📊 Using {len(points_1)} point correspondences")

        # Use ICP to find transformation
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points_1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points_2)

        # Initial alignment using centroids
        centroid_1 = np.mean(points_1, axis=0)
        centroid_2 = np.mean(points_2, axis=0)

        initial_transform = np.eye(4)
        initial_transform[:3, 3] = centroid_1 - centroid_2

        # Refine with ICP
        threshold = 0.05  # 5cm threshold
        reg_result = o3d.pipelines.registration.registration_icp(
            pcd2,
            pcd1,
            threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )

        transformation_matrix = reg_result.transformation
        fitness = reg_result.fitness
        inlier_rmse = reg_result.inlier_rmse

        print(f"✅ Transformation computed:")
        print(f"   Fitness: {fitness:.3f}")
        print(f"   RMSE: {inlier_rmse*1000:.1f}mm")

        return transformation_matrix, inlier_rmse * 1000

    def save_calibration(self, transformation_matrix: np.ndarray, error: float):
        """Save calibration data to file"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        calibration_data = {
            "transformation_matrix": transformation_matrix.tolist(),
            "device1_intrinsics": self.intrinsics.copy(),
            "device2_intrinsics": self.intrinsics.copy(),
            "calibration_error": error,
            "calibration_method": "improved_feature_matching_icp",
            "calibration_timestamp": time.time(),
            "device1_id": self.device1_id,
            "device2_id": self.device2_id,
            "num_calibration_frames": len(self.calibration_frames),
        }

        calib_file = config_dir / "dual_kinect_calibration.json"
        with open(calib_file, "w") as f:
            json.dump(calibration_data, f, indent=2)

        print(f"💾 Calibration saved to: {calib_file}")

    def run_improved_calibration(self) -> bool:
        """Run the improved calibration process"""
        print("🎯 Improved Dual Kinect Calibration")
        print("=" * 50)

        # Collect calibration data
        if not self.collect_improved_calibration_data(num_captures=5):
            print("❌ Failed to collect sufficient calibration data")
            return False

        # Compute transformation
        transformation_matrix, rmse_error = self.compute_transformation_matrix()

        # Save calibration
        self.save_calibration(transformation_matrix, rmse_error)

        # Assessment
        print(f"\n📊 Calibration Complete!")
        print("=" * 50)
        print(f"Final error: {rmse_error:.1f}mm")

        if rmse_error < 15.0:
            print("✅ Excellent calibration quality!")
        elif rmse_error < 30.0:
            print("✅ Good calibration quality")
        elif rmse_error < 50.0:
            print("⚠️  Acceptable calibration quality")
        else:
            print("❌ Poor calibration quality - consider recalibrating")

        return rmse_error < 50.0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Improved dual Kinect calibration")
    parser.add_argument("--device1", type=int, default=0, help="First Kinect device ID")
    parser.add_argument(
        "--device2", type=int, default=1, help="Second Kinect device ID"
    )

    args = parser.parse_args()

    try:
        import scipy

        print("✅ SciPy available for advanced point filtering")
    except ImportError:
        print("⚠️  SciPy not available - install with: pip install scipy")
        print("    (Will use basic filtering instead)")

    calibrator = ImprovedDualKinectCalibrator(
        device1_id=args.device1, device2_id=args.device2
    )

    success = calibrator.run_improved_calibration()

    if success:
        print("\n✅ Improved calibration successful!")
        print("📈 Ready for fusion: python src/step5_kinect_fusion.py")
    else:
        print("\n❌ Calibration failed!")


if __name__ == "__main__":
    main()
