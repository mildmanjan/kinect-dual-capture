#!/usr/bin/env python3
"""
Step 6: Kinect to Mesh Demo
Convert Kinect RGB-D data to animated 3D meshes in real-time.

This demonstrates:
- Point cloud generation from Kinect data
- Real-time mesh reconstruction
- Mesh animation (frame sequence)
- Export to common 3D formats
"""

import numpy as np
import cv2
import open3d as o3d
import time
import sys
from pathlib import Path
from typing import List, Optional
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.kinect_capture import BaseKinectCapture, KinectSyncMode, KinectFrame


class KinectMeshGenerator:
    """Generate 3D meshes from Kinect RGB-D data"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.capture = BaseKinectCapture(
            device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
        )

        # Kinect camera intrinsics - we'll use depth camera intrinsics since we're aligning to depth
        # These are approximate values for Azure Kinect depth camera
        self.depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=576, fx=504.0, fy=504.0, cx=320.0, cy=288.0
        )

        # Mesh generation settings
        self.voxel_size = 0.005  # 5mm voxels
        self.depth_scale = 1000.0  # Convert mm to meters
        self.depth_trunc = 1.5  # Truncate at 1.5 meters

        # Animation storage
        self.mesh_sequence = []
        self.pointcloud_sequence = []

        # Output directory
        self.output_dir = Path("data/mesh_animation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def kinect_frame_to_pointcloud(
        self, frame: KinectFrame
    ) -> Optional[o3d.geometry.PointCloud]:
        """Convert Kinect frame to Open3D point cloud"""
        if frame.color_image is None or frame.depth_image is None:
            return None

        try:
            # Get original dimensions
            color_height, color_width = frame.color_image.shape[:2]
            depth_height, depth_width = frame.depth_image.shape[:2]

            print(
                f"📏 Original sizes - Color: {color_width}x{color_height}, Depth: {depth_width}x{depth_height}"
            )

            # Resize color image to match depth image size
            color_resized = cv2.resize(  # pylint: disable=no-member
                frame.color_image,
                (depth_width, depth_height),
                interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
            )

            # Convert BGR to RGB for Open3D
            color_rgb = cv2.cvtColor(
                color_resized, cv2.COLOR_BGR2RGB
            )  # pylint: disable=no-member
            color_o3d = o3d.geometry.Image(color_rgb)

            # Convert depth to float and scale to meters
            depth_float = frame.depth_image.astype(np.float32) / self.depth_scale
            depth_o3d = o3d.geometry.Image(depth_float)

            # Verify sizes match
            print(
                f"📏 Aligned sizes - Color: {color_rgb.shape[1]}x{color_rgb.shape[0]}, Depth: {depth_float.shape[1]}x{depth_float.shape[0]}"
            )

            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,  # Already scaled
                depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False,
            )

            # Generate point cloud using depth camera intrinsics
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, self.depth_intrinsics
            )

            print(f"✅ Point cloud created with {len(pcd.points)} points")
            return pcd

        except Exception as e:
            print(f"❌ Error creating point cloud: {e}")
            import traceback

            traceback.print_exc()
            return None

    def clean_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Clean and filter point cloud"""
        if pcd is None or len(pcd.points) == 0:
            return pcd

        # Downsample
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Remove points too far from origin (background removal)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Keep points within reasonable range (0.3m to 2.0m)
        mask = (points[:, 2] > 0.3) & (points[:, 2] < 2.0)

        if np.any(mask):
            pcd.points = o3d.utility.Vector3dVector(points[mask])
            if len(colors) > 0:
                pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        return pcd

    def pointcloud_to_mesh(
        self, pcd: o3d.geometry.PointCloud, method: str = "ball_pivoting"
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """Convert point cloud to triangle mesh"""
        if pcd is None or len(pcd.points) < 100:
            print(f"⚠️  Point cloud too small: {len(pcd.points) if pcd else 0} points")
            return None

        try:
            print(f"🔧 Creating mesh from {len(pcd.points)} points using {method}")

            # Estimate normals (required for most methods)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(100)

            if method == "poisson":
                # Poisson surface reconstruction (good quality but can be slow)
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, linear_fit=False  # Reduced depth for speed
                )

            elif method == "ball_pivoting":
                # Ball pivoting algorithm (faster, good for detailed surfaces)
                radii = [0.005, 0.01, 0.02, 0.04]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

            elif method == "alpha_shape":
                # Alpha shape (good for thin structures, fastest)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.03
                )

            else:
                print(f"⚠️  Unknown mesh method: {method}, using ball_pivoting")
                radii = [0.005, 0.01, 0.02, 0.04]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

            if mesh is None or len(mesh.triangles) == 0:
                print("⚠️  Mesh generation failed - no triangles created")
                return None

            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            print(
                f"✅ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
            )
            return mesh

        except Exception as e:
            print(f"❌ Error creating mesh: {e}")
            import traceback

            traceback.print_exc()
            return None

    def visualize_realtime(self, duration: int = 30):
        """Real-time visualization of point clouds and meshes"""
        print(f"🎬 Starting real-time mesh visualization for {duration}s")
        print("Controls: ESC to quit, SPACE to save current mesh")

        # Initialize visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window("Kinect Real-time Mesh", width=1280, height=720)

        # Start capture
        if not self.capture.start_capture():
            print("❌ Failed to start Kinect capture")
            return

        try:
            start_time = time.time()
            frame_count = 0
            current_geometry = None
            show_mesh = True  # Toggle between point cloud and mesh

            while time.time() - start_time < duration:
                # Capture frame
                frame = self.capture.capture_frame()
                if frame is None:
                    continue

                frame_count += 1

                # Generate point cloud
                pcd = self.kinect_frame_to_pointcloud(frame)
                if pcd is None:
                    continue

                # Clean point cloud
                pcd = self.clean_pointcloud(pcd)

                # Generate mesh every few frames (computationally expensive)
                if show_mesh and frame_count % 5 == 0:
                    mesh = self.pointcloud_to_mesh(pcd, method="poisson")
                    display_geometry = mesh if mesh is not None else pcd
                else:
                    display_geometry = pcd

                # Update visualization
                if current_geometry is not None:
                    vis.remove_geometry(current_geometry, reset_bounding_box=False)

                if display_geometry is not None:
                    vis.add_geometry(
                        display_geometry, reset_bounding_box=(frame_count == 1)
                    )
                    current_geometry = display_geometry

                # Update view
                vis.poll_events()
                vis.update_renderer()

                # Print progress
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    points = len(pcd.points) if pcd else 0
                    print(f"📊 Frame {frame_count}, FPS: {fps:.1f}, Points: {points}")

                # Check for user input
                # Note: Open3D's key handling is limited in this mode
                # In a full application, you'd use proper event handling

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")

        finally:
            vis.destroy_window()
            self.capture.stop_capture()
            print("🧹 Visualization stopped")

    def capture_mesh_sequence(
        self, duration: int = 10, fps: int = 5
    ) -> List[o3d.geometry.TriangleMesh]:
        """Capture a sequence of meshes for animation"""
        print(f"🎥 Capturing mesh sequence: {duration}s at {fps} FPS")

        if not self.capture.start_capture():
            print("❌ Failed to start Kinect capture")
            return []

        mesh_sequence = []
        frame_interval = 1.0 / fps

        try:
            start_time = time.time()
            last_capture_time = 0
            frame_count = 0

            while time.time() - start_time < duration:
                current_time = time.time() - start_time

                # Capture at specified FPS
                if current_time - last_capture_time >= frame_interval:
                    frame = self.capture.capture_frame()
                    if frame is None:
                        continue

                    # Generate point cloud and mesh
                    pcd = self.kinect_frame_to_pointcloud(frame)
                    if pcd is not None:
                        pcd = self.clean_pointcloud(pcd)
                        mesh = self.pointcloud_to_mesh(pcd, method="poisson")

                        if mesh is not None:
                            mesh_sequence.append(mesh)
                            frame_count += 1
                            print(
                                f"📦 Captured mesh {frame_count} at {current_time:.1f}s"
                            )

                    last_capture_time = current_time

        except KeyboardInterrupt:
            print("\n🛑 Capture interrupted by user")

        finally:
            self.capture.stop_capture()

        print(f"✅ Captured {len(mesh_sequence)} meshes")
        return mesh_sequence

    def export_mesh_sequence(
        self,
        mesh_sequence: List[o3d.geometry.TriangleMesh],
        base_name: str = "kinect_mesh",
    ) -> List[Path]:
        """Export mesh sequence to files"""
        if not mesh_sequence:
            print("❌ No meshes to export")
            return []

        exported_files = []
        timestamp = int(time.time())

        print(f"💾 Exporting {len(mesh_sequence)} meshes...")

        for i, mesh in enumerate(mesh_sequence):
            # Export as OBJ (simple format)
            obj_filename = self.output_dir / f"{base_name}_{timestamp}_{i:04d}.obj"
            o3d.io.write_triangle_mesh(str(obj_filename), mesh)
            exported_files.append(obj_filename)

            # Export as PLY (preserves colors)
            ply_filename = self.output_dir / f"{base_name}_{timestamp}_{i:04d}.ply"
            o3d.io.write_triangle_mesh(str(ply_filename), mesh)
            exported_files.append(ply_filename)

        # Create animation metadata
        metadata = {
            "timestamp": timestamp,
            "frame_count": len(mesh_sequence),
            "base_name": base_name,
            "formats": ["obj", "ply"],
            "fps": len(mesh_sequence) / 10.0,  # Assuming 10 second capture
            "notes": "Generated from Azure Kinect RGB-D data",
        }

        metadata_file = self.output_dir / f"{base_name}_{timestamp}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Exported to: {self.output_dir}")
        print(f"📁 Files: {len(exported_files)} mesh files + metadata")

        return exported_files

    def main():
        """Main function"""
        import argparse

        parser = argparse.ArgumentParser(
            description="Generate 3D meshes from Kinect data"
        )
        parser.add_argument(
            "--mode",
            choices=["realtime", "sequence"],
            default="realtime",
            help="Visualization mode",
        )
        parser.add_argument(
            "--duration", type=int, default=10, help="Capture duration in seconds"
        )
        parser.add_argument(
            "--fps", type=int, default=5, help="Capture FPS for sequence mode"
        )
        parser.add_argument("--device-id", type=int, default=0, help="Kinect device ID")
        parser.add_argument(
            "--export", action="store_true", help="Export mesh sequence"
        )

        args = parser.parse_args()

        print("🎯 Kinect to 3D Mesh Generator")
        print("=" * 50)

        generator = KinectMeshGenerator(device_id=args.device_id)

        if args.mode == "realtime":
            generator.visualize_realtime(duration=args.duration)

        elif args.mode == "sequence":
            mesh_sequence = generator.capture_mesh_sequence(
                duration=args.duration, fps=args.fps
            )

            if args.export and mesh_sequence:
                exported_files = generator.export_mesh_sequence(mesh_sequence)
                generator.create_blender_import_script("kinect_mesh", int(time.time()))

                print(f"\n🎬 Animation ready!")
                print(f"📁 Files exported to: {generator.output_dir}")
                print(f"🎨 Import to Blender using the generated script")

        print("✅ Mesh generation complete!")


if __name__ == "__main__":
    main()
