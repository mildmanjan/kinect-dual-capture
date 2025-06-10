#!/usr/bin/env python3
"""
Step 5: Kinect Point Cloud Fusion
Fuse point clouds from dual Azure Kinect devices in real-time.

This script:
- Synchronizes capture from two Kinect devices
- Converts RGB-D data to point clouds
- Registers and aligns point clouds using calibration
- Fuses point clouds into unified 3D representation
- Exports fused point clouds and meshes
- Provides real-time visualization

File location: src/step5_kinect_fusion.py
"""

import numpy as np
import cv2
import open3d as o3d
import time
import sys
import threading
import queue
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json
import argparse
from dataclasses import dataclass

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.kinect_capture import BaseKinectCapture, KinectSyncMode, KinectFrame
from utils.compression_utils import DataCompressor, CompressionLevel


@dataclass
class CalibrationData:
    """Calibration data between two Kinect devices"""

    # Transformation matrix from Kinect1 to Kinect2 coordinate system
    transformation_matrix: np.ndarray

    # Camera intrinsics for each device
    device1_intrinsics: Dict[str, float]
    device2_intrinsics: Dict[str, float]

    # Calibration quality metrics
    calibration_error: float = 0.0
    calibration_method: str = "manual"
    calibration_timestamp: float = 0.0


class DualKinectFusion:
    """Dual Kinect point cloud fusion system"""

    def __init__(
        self, device1_id: int = 0, device2_id: int = 1, use_sync_cable: bool = True
    ):
        self.device1_id = device1_id
        self.device2_id = device2_id
        self.use_sync_cable = use_sync_cable

        # Kinect captures - use appropriate sync mode
        if use_sync_cable:
            print("🔗 Using sync cable mode (master/subordinate)")
            sync_mode1 = KinectSyncMode.MASTER
            sync_mode2 = KinectSyncMode.SUBORDINATE
        else:
            print("📡 Using standalone mode (no sync cable required)")
            sync_mode1 = KinectSyncMode.STANDALONE
            sync_mode2 = KinectSyncMode.STANDALONE

        self.capture1 = BaseKinectCapture(device_id=device1_id, sync_mode=sync_mode1)
        self.capture2 = BaseKinectCapture(device_id=device2_id, sync_mode=sync_mode2)

        # Calibration data
        self.calibration = self._load_or_create_calibration()

        # Point cloud processing settings
        self.voxel_size = 0.005  # 5mm voxels
        self.depth_scale = 1000.0  # Convert mm to meters
        self.depth_min = 0.3  # Minimum depth in meters
        self.depth_max = 3.0  # Maximum depth in meters

        # Fusion settings
        self.enable_filtering = True
        self.enable_registration = True
        self.max_points_per_cloud = 50000  # Limit for performance

        # Threading for synchronized capture
        self.capture_thread = None
        self.is_capturing = False
        self.frame_queue1 = queue.Queue(maxsize=10)
        self.frame_queue2 = queue.Queue(maxsize=10)

        # Statistics
        self.fusion_stats = {
            "frames_processed": 0,
            "successful_fusions": 0,
            "avg_points_device1": 0,
            "avg_points_device2": 0,
            "avg_points_fused": 0,
            "avg_fusion_time": 0.0,
        }

        # Output directory
        self.output_dir = Path("data/fusion_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_calibration(self) -> CalibrationData:
        """Load existing calibration or create default"""
        calib_file = Path("config/dual_kinect_calibration.json")

        if calib_file.exists():
            try:
                with open(calib_file, "r") as f:
                    data = json.load(f)

                calibration = CalibrationData(
                    transformation_matrix=np.array(data["transformation_matrix"]),
                    device1_intrinsics=data["device1_intrinsics"],
                    device2_intrinsics=data["device2_intrinsics"],
                    calibration_error=data.get("calibration_error", 0.0),
                    calibration_method=data.get("calibration_method", "loaded"),
                    calibration_timestamp=data.get("calibration_timestamp", 0.0),
                )

                print(f"✅ Loaded calibration from {calib_file}")
                print(f"   Error: {calibration.calibration_error:.4f}mm")
                return calibration

            except Exception as e:
                print(f"⚠️  Failed to load calibration: {e}")

        # Create default calibration (assumes devices side-by-side, 20cm apart)
        print("📐 Creating default calibration (devices 20cm apart)")

        # Default transformation: 20cm translation in X direction
        default_transform = np.eye(4)
        default_transform[0, 3] = 0.20  # 20cm offset in X

        # Default camera intrinsics (Azure Kinect depth camera approximate values)
        default_intrinsics = {
            "fx": 504.0,
            "fy": 504.0,
            "cx": 320.0,
            "cy": 288.0,
            "width": 640,
            "height": 576,
        }

        return CalibrationData(
            transformation_matrix=default_transform,
            device1_intrinsics=default_intrinsics.copy(),
            device2_intrinsics=default_intrinsics.copy(),
            calibration_error=999.0,  # High error to indicate uncalibrated
            calibration_method="default",
            calibration_timestamp=time.time(),
        )

    def save_calibration(self, calibration: CalibrationData):
        """Save calibration data to file"""
        calib_dir = Path("config")
        calib_dir.mkdir(exist_ok=True)

        calib_file = calib_dir / "dual_kinect_calibration.json"

        data = {
            "transformation_matrix": calibration.transformation_matrix.tolist(),
            "device1_intrinsics": calibration.device1_intrinsics,
            "device2_intrinsics": calibration.device2_intrinsics,
            "calibration_error": calibration.calibration_error,
            "calibration_method": calibration.calibration_method,
            "calibration_timestamp": calibration.calibration_timestamp,
        }

        with open(calib_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"💾 Calibration saved to {calib_file}")

    def frame_to_pointcloud(
        self, frame: KinectFrame, intrinsics: Dict[str, float]
    ) -> Optional[o3d.geometry.PointCloud]:
        """Convert Kinect frame to Open3D point cloud"""
        if frame.color_image is None or frame.depth_image is None:
            return None

        try:
            # Get dimensions
            depth_height, depth_width = frame.depth_image.shape
            color_height, color_width = frame.color_image.shape[:2]

            # Resize color to match depth resolution
            color_resized = cv2.resize(
                frame.color_image,
                (depth_width, depth_height),
                interpolation=cv2.INTER_AREA,
            )

            # Convert BGR to RGB
            color_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

            # Create Open3D images
            color_o3d = o3d.geometry.Image(color_rgb)
            depth_o3d = o3d.geometry.Image(
                frame.depth_image.astype(np.float32) / self.depth_scale
            )

            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,  # Already scaled
                depth_trunc=self.depth_max,
                convert_rgb_to_intensity=False,
            )

            # Create camera intrinsics object
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
                fx=intrinsics["fx"],
                fy=intrinsics["fy"],
                cx=intrinsics["cx"],
                cy=intrinsics["cy"],
            )

            # Generate point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)

            return pcd

        except Exception as e:
            print(f"❌ Error creating point cloud: {e}")
            return None

    def filter_pointcloud(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        """Filter and clean point cloud"""
        if pcd is None or len(pcd.points) == 0:
            return pcd

        # Downsample for performance
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Filter by depth range
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Keep points within depth range
        depth_mask = (points[:, 2] >= self.depth_min) & (points[:, 2] <= self.depth_max)

        if np.any(depth_mask):
            pcd.points = o3d.utility.Vector3dVector(points[depth_mask])
            if len(colors) > 0:
                pcd.colors = o3d.utility.Vector3dVector(colors[depth_mask])

        # Limit number of points for performance
        if len(pcd.points) > self.max_points_per_cloud:
            indices = np.random.choice(
                len(pcd.points), self.max_points_per_cloud, replace=False
            )
            pcd = pcd.select_by_index(indices)

        return pcd

    def register_pointclouds(
        self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Register point clouds using calibration"""
        if pcd1 is None or pcd2 is None:
            return pcd1, pcd2

        try:
            # Transform pcd2 to pcd1's coordinate system using calibration
            pcd2_transformed = pcd2.transform(self.calibration.transformation_matrix)

            # If registration is enabled and calibration error is high, try ICP refinement
            if self.enable_registration and self.calibration.calibration_error > 10.0:
                # ICP refinement for better alignment
                threshold = 0.02  # 2cm threshold
                reg_result = o3d.pipelines.registration.registration_icp(
                    pcd2_transformed,
                    pcd1,
                    threshold,
                    np.eye(
                        4
                    ),  # Initial guess (identity since we already applied calibration)
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                if reg_result.fitness > 0.1:  # If registration was successful
                    pcd2_transformed = pcd2_transformed.transform(
                        reg_result.transformation
                    )
                    print(
                        f"🔧 ICP refinement applied (fitness: {reg_result.fitness:.3f})"
                    )

            return pcd1, pcd2_transformed

        except Exception as e:
            print(f"⚠️  Registration failed: {e}")
            return pcd1, pcd2

    def fuse_pointclouds(
        self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        """Fuse two registered point clouds"""
        if pcd1 is None and pcd2 is None:
            return o3d.geometry.PointCloud()
        elif pcd1 is None:
            return pcd2
        elif pcd2 is None:
            return pcd1

        try:
            # Simple fusion: combine point clouds
            fused_pcd = pcd1 + pcd2

            # Remove duplicate points (voxel downsampling)
            fused_pcd = fused_pcd.voxel_down_sample(voxel_size=self.voxel_size)

            # Final cleanup
            if self.enable_filtering:
                fused_pcd, _ = fused_pcd.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )

            return fused_pcd

        except Exception as e:
            print(f"❌ Fusion failed: {e}")
            return pcd1 if pcd1 is not None else o3d.geometry.PointCloud()

    def process_frame_pair(
        self, frame1: KinectFrame, frame2: KinectFrame
    ) -> Optional[o3d.geometry.PointCloud]:
        """Process a synchronized pair of frames"""
        start_time = time.time()

        # Convert frames to point clouds
        pcd1 = self.frame_to_pointcloud(frame1, self.calibration.device1_intrinsics)
        pcd2 = self.frame_to_pointcloud(frame2, self.calibration.device2_intrinsics)

        if pcd1 is None and pcd2 is None:
            return None

        # Filter point clouds
        if self.enable_filtering:
            if pcd1 is not None:
                pcd1 = self.filter_pointcloud(pcd1)
            if pcd2 is not None:
                pcd2 = self.filter_pointcloud(pcd2)

        # Register point clouds
        pcd1_reg, pcd2_reg = self.register_pointclouds(pcd1, pcd2)

        # Fuse point clouds
        fused_pcd = self.fuse_pointclouds(pcd1_reg, pcd2_reg)

        # Update statistics
        processing_time = time.time() - start_time
        self.update_stats(pcd1, pcd2, fused_pcd, processing_time)

        return fused_pcd

    def update_stats(self, pcd1, pcd2, fused_pcd, processing_time):
        """Update fusion statistics"""
        self.fusion_stats["frames_processed"] += 1

        if fused_pcd is not None and len(fused_pcd.points) > 0:
            self.fusion_stats["successful_fusions"] += 1

        # Running averages
        n = self.fusion_stats["frames_processed"]

        points1 = len(pcd1.points) if pcd1 is not None else 0
        points2 = len(pcd2.points) if pcd2 is not None else 0
        points_fused = len(fused_pcd.points) if fused_pcd is not None else 0

        self.fusion_stats["avg_points_device1"] = (
            self.fusion_stats["avg_points_device1"] * (n - 1) + points1
        ) / n
        self.fusion_stats["avg_points_device2"] = (
            self.fusion_stats["avg_points_device2"] * (n - 1) + points2
        ) / n
        self.fusion_stats["avg_points_fused"] = (
            self.fusion_stats["avg_points_fused"] * (n - 1) + points_fused
        ) / n
        self.fusion_stats["avg_fusion_time"] = (
            self.fusion_stats["avg_fusion_time"] * (n - 1) + processing_time
        ) / n

    def synchronized_capture_thread(self):
        """Thread for synchronized capture from both devices"""
        print("🎥 Starting synchronized capture thread...")

        # Start both captures
        if not self.capture1.start_capture():
            print("❌ Failed to start capture from device 1")
            return

        if not self.capture2.start_capture():
            print("❌ Failed to start capture from device 2")
            self.capture1.stop_capture()
            return

        print("✅ Both Kinect devices started successfully")

        try:
            while self.is_capturing:
                # Capture from both devices
                frame1 = self.capture1.capture_frame()
                frame2 = self.capture2.capture_frame()

                # Add to queues if frames are valid
                if frame1 is not None:
                    try:
                        self.frame_queue1.put_nowait(frame1)
                    except queue.Full:
                        # Remove oldest frame if queue is full
                        try:
                            self.frame_queue1.get_nowait()
                            self.frame_queue1.put_nowait(frame1)
                        except queue.Empty:
                            pass

                if frame2 is not None:
                    try:
                        self.frame_queue2.put_nowait(frame2)
                    except queue.Full:
                        # Remove oldest frame if queue is full
                        try:
                            self.frame_queue2.get_nowait()
                            self.frame_queue2.put_nowait(frame2)
                        except queue.Empty:
                            pass

                time.sleep(1 / 30)  # ~30 FPS max

        except Exception as e:
            print(f"❌ Capture thread error: {e}")

        finally:
            self.capture1.stop_capture()
            self.capture2.stop_capture()
            print("🛑 Synchronized capture stopped")

    def get_synchronized_frames(
        self, timeout: float = 0.1
    ) -> Tuple[Optional[KinectFrame], Optional[KinectFrame]]:
        """Get synchronized frames from both devices"""
        try:
            frame1 = (
                self.frame_queue1.get(timeout=timeout)
                if not self.frame_queue1.empty()
                else None
            )
            frame2 = (
                self.frame_queue2.get(timeout=timeout)
                if not self.frame_queue2.empty()
                else None
            )
            return frame1, frame2
        except queue.Empty:
            return None, None

    def visualize_realtime_fusion(self, duration: int = 30):
        """Real-time visualization of fused point clouds"""
        print(f"🎬 Starting real-time fusion visualization for {duration}s")
        print(
            "Controls: ESC to quit, SPACE to save current fusion, 'c' to run calibration"
        )

        # Initialize visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window("Dual Kinect Fusion", width=1280, height=720)

        # Start synchronized capture
        self.is_capturing = True
        self.capture_thread = threading.Thread(
            target=self.synchronized_capture_thread, daemon=True
        )
        self.capture_thread.start()

        try:
            start_time = time.time()
            frame_count = 0
            current_geometry = None
            last_save_time = 0

            while time.time() - start_time < duration:
                # Get synchronized frames
                frame1, frame2 = self.get_synchronized_frames()

                if frame1 is None and frame2 is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1

                # Process frames to get fused point cloud
                fused_pcd = self.process_frame_pair(frame1, frame2)

                if fused_pcd is not None and len(fused_pcd.points) > 0:
                    # Update visualization
                    if current_geometry is not None:
                        vis.remove_geometry(current_geometry, reset_bounding_box=False)

                    vis.add_geometry(fused_pcd, reset_bounding_box=(frame_count == 1))
                    current_geometry = fused_pcd

                # Update view
                vis.poll_events()
                vis.update_renderer()

                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    self.print_current_stats()

                # Auto-save every 10 seconds
                current_time = time.time()
                if current_time - last_save_time > 10:
                    if fused_pcd is not None:
                        self.save_pointcloud(
                            fused_pcd, f"auto_save_{int(current_time)}"
                        )
                    last_save_time = current_time

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")

        finally:
            self.is_capturing = False
            if self.capture_thread:
                self.capture_thread.join(timeout=2)
            vis.destroy_window()
            print("🧹 Fusion visualization stopped")

    def print_current_stats(self):
        """Print current fusion statistics"""
        stats = self.fusion_stats
        success_rate = (
            (stats["successful_fusions"] / stats["frames_processed"] * 100)
            if stats["frames_processed"] > 0
            else 0
        )

        print(
            f"📊 Fusion Stats - Frames: {stats['frames_processed']}, "
            f"Success: {success_rate:.1f}%, "
            f"Avg Points: {stats['avg_points_fused']:.0f}, "
            f"Time: {stats['avg_fusion_time']*1000:.1f}ms"
        )

    def save_pointcloud(
        self, pcd: o3d.geometry.PointCloud, name: str = "fused_pointcloud"
    ):
        """Save point cloud to file"""
        if pcd is None or len(pcd.points) == 0:
            return

        timestamp = int(time.time())

        # Save as PLY (with colors)
        ply_path = self.output_dir / f"{name}_{timestamp}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)

        # Save as PCD (Open3D format)
        pcd_path = self.output_dir / f"{name}_{timestamp}.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd)

        print(f"💾 Saved fused point cloud: {ply_path}")
        print(f"   Points: {len(pcd.points)}, Colors: {len(pcd.colors) > 0}")

    def capture_fusion_sequence(
        self, duration: int = 10, fps: int = 5
    ) -> List[o3d.geometry.PointCloud]:
        """Capture a sequence of fused point clouds"""
        print(f"🎥 Capturing fusion sequence: {duration}s at {fps} FPS")

        # Start synchronized capture
        self.is_capturing = True
        self.capture_thread = threading.Thread(
            target=self.synchronized_capture_thread, daemon=True
        )
        self.capture_thread.start()

        fusion_sequence = []
        frame_interval = 1.0 / fps

        try:
            start_time = time.time()
            last_capture_time = 0
            sequence_frame = 0

            while time.time() - start_time < duration:
                current_time = time.time() - start_time

                # Capture at specified FPS
                if current_time - last_capture_time >= frame_interval:
                    frame1, frame2 = self.get_synchronized_frames()

                    if frame1 is not None or frame2 is not None:
                        fused_pcd = self.process_frame_pair(frame1, frame2)

                        if fused_pcd is not None and len(fused_pcd.points) > 0:
                            fusion_sequence.append(fused_pcd)
                            sequence_frame += 1
                            print(
                                f"📦 Captured fusion {sequence_frame} at {current_time:.1f}s "
                                f"({len(fused_pcd.points)} points)"
                            )

                    last_capture_time = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n🛑 Sequence capture interrupted")

        finally:
            self.is_capturing = False
            if self.capture_thread:
                self.capture_thread.join(timeout=2)

        print(f"✅ Captured {len(fusion_sequence)} fused point clouds")
        return fusion_sequence

    def export_fusion_sequence(
        self,
        fusion_sequence: List[o3d.geometry.PointCloud],
        base_name: str = "fusion_sequence",
    ):
        """Export fusion sequence to files"""
        if not fusion_sequence:
            print("❌ No fusion sequence to export")
            return

        timestamp = int(time.time())
        sequence_dir = self.output_dir / f"{base_name}_{timestamp}"
        sequence_dir.mkdir(exist_ok=True)

        print(f"💾 Exporting {len(fusion_sequence)} fused point clouds...")

        exported_files = []
        for i, pcd in enumerate(fusion_sequence):
            # Export as PLY
            ply_file = sequence_dir / f"fusion_{i:04d}.ply"
            o3d.io.write_point_cloud(str(ply_file), pcd)
            exported_files.append(ply_file)

        # Create sequence metadata
        metadata = {
            "timestamp": timestamp,
            "frame_count": len(fusion_sequence),
            "base_name": base_name,
            "avg_points_per_frame": np.mean(
                [len(pcd.points) for pcd in fusion_sequence]
            ),
            "calibration_used": {
                "method": self.calibration.calibration_method,
                "error": self.calibration.calibration_error,
                "timestamp": self.calibration.calibration_timestamp,
            },
            "fusion_stats": self.fusion_stats.copy(),
            "processing_settings": {
                "voxel_size": self.voxel_size,
                "depth_range": [self.depth_min, self.depth_max],
                "filtering_enabled": self.enable_filtering,
                "registration_enabled": self.enable_registration,
            },
        }

        metadata_file = sequence_dir / "sequence_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Fusion sequence exported to: {sequence_dir}")
        print(f"📁 Files: {len(exported_files)} point clouds + metadata")

        return exported_files


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dual Kinect Point Cloud Fusion")
    parser.add_argument("--device1", type=int, default=0, help="First Kinect device ID")
    parser.add_argument(
        "--device2", type=int, default=1, help="Second Kinect device ID"
    )
    parser.add_argument(
        "--mode",
        choices=["realtime", "sequence"],
        default="realtime",
        help="Fusion mode",
    )
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument(
        "--fps", type=int, default=5, help="Capture FPS for sequence mode"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export sequence to files"
    )
    parser.add_argument(
        "--sync-cable",
        action="store_true",
        help="Use sync cable (master/subordinate mode)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.005, help="Voxel size for downsampling"
    )
    parser.add_argument(
        "--no-filter", action="store_true", help="Disable point cloud filtering"
    )
    parser.add_argument(
        "--no-registration", action="store_true", help="Disable ICP registration"
    )

    args = parser.parse_args()

    print("🎯 Dual Kinect Point Cloud Fusion")
    print("=" * 50)
    print(f"Device 1: {args.device1}")
    print(f"Device 2: {args.device2}")
    print(f"Sync cable: {'Yes' if args.sync_cable else 'No'}")
    print(f"Mode: {args.mode}")
    print(f"Duration: {args.duration}s")

    # Create fusion system
    fusion = DualKinectFusion(
        device1_id=args.device1,
        device2_id=args.device2,
        use_sync_cable=args.sync_cable,  # This was missing!
    )

    # Apply settings
    fusion.voxel_size = args.voxel_size
    fusion.enable_filtering = not args.no_filter
    fusion.enable_registration = not args.no_registration

    print(f"\n⚙️  Settings:")
    print(f"   Voxel size: {fusion.voxel_size}m")
    print(f"   Filtering: {'ON' if fusion.enable_filtering else 'OFF'}")
    print(f"   Registration: {'ON' if fusion.enable_registration else 'OFF'}")
    print(f"   Calibration error: {fusion.calibration.calibration_error:.1f}mm")

    if args.mode == "realtime":
        fusion.visualize_realtime_fusion(duration=args.duration)

    elif args.mode == "sequence":
        fusion_sequence = fusion.capture_fusion_sequence(
            duration=args.duration, fps=args.fps
        )

        if args.export and fusion_sequence:
            fusion.export_fusion_sequence(fusion_sequence)

    # Print final statistics
    print(f"\n📈 Final Fusion Statistics:")
    print("=" * 50)
    stats = fusion.fusion_stats
    if stats["frames_processed"] > 0:
        success_rate = stats["successful_fusions"] / stats["frames_processed"] * 100
        print(f"Frames processed: {stats['frames_processed']}")
        print(
            f"Successful fusions: {stats['successful_fusions']} ({success_rate:.1f}%)"
        )
        print(f"Average points per device:")
        print(f"  Device 1: {stats['avg_points_device1']:.0f}")
        print(f"  Device 2: {stats['avg_points_device2']:.0f}")
        print(f"  Fused: {stats['avg_points_fused']:.0f}")
        print(f"Average fusion time: {stats['avg_fusion_time']*1000:.1f}ms")

        if success_rate > 80:
            print("✅ Excellent fusion performance!")
        elif success_rate > 60:
            print("✅ Good fusion performance")
        else:
            print("⚠️  Consider improving calibration or reducing noise")

    print("✅ Kinect fusion complete!")


if __name__ == "__main__":
    main()
