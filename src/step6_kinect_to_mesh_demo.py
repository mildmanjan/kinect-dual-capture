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
import json
from pathlib import Path
from typing import List, Optional, Tuple
import argparse

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

        # Kinect camera intrinsics - Azure Kinect depth camera
        # These are approximate values that work well in practice
        self.depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=576, fx=504.0, fy=504.0, cx=320.0, cy=288.0
        )

        # Mesh generation settings
        self.voxel_size = 0.005  # 5mm voxels
        self.depth_scale = 1000.0  # Convert mm to meters
        self.depth_trunc = 1.5  # Truncate at 1.5 meters
        self.min_points = 1000  # Minimum points for mesh generation

        # Animation storage
        self.mesh_sequence = []
        self.pointcloud_sequence = []

        # Output directory
        self.output_dir = Path("data/mesh_animation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎯 Kinect Mesh Generator initialized")
        print(f"📁 Output directory: {self.output_dir}")

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
            color_resized = cv2.resize(
                frame.color_image,
                (depth_width, depth_height),
                interpolation=cv2.INTER_AREA,
            )

            # Convert BGR to RGB for Open3D
            color_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

            # Create Open3D images
            color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(frame.depth_image.astype(np.uint16))

            # Create RGB-D image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=self.depth_scale,
                depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False,
            )

            # Create point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, self.depth_intrinsics
            )

            # Transform from camera coordinates to world coordinates
            # Flip Y and Z axes to get standard orientation
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            print(f"✅ Point cloud created: {len(pcd.points)} points")
            return pcd

        except Exception as e:
            print(f"❌ Error creating point cloud: {e}")
            return None

    def clean_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Clean and filter point cloud"""
        if pcd is None or len(pcd.points) == 0:
            return pcd

        try:
            original_points = len(pcd.points)

            # Remove statistical outliers
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # Remove radius outliers
            pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

            # Downsample to reduce noise and computation
            if len(pcd.points) > 50000:  # Only downsample if too many points
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

            # Estimate normals for better mesh generation
            pcd.estimate_normals()
            pcd.normalize_normals()

            cleaned_points = len(pcd.points)
            reduction = ((original_points - cleaned_points) / original_points) * 100

            print(
                f"🧹 Cleaned point cloud: {cleaned_points} points ({reduction:.1f}% reduction)"
            )
            return pcd

        except Exception as e:
            print(f"❌ Error cleaning point cloud: {e}")
            return pcd

    def pointcloud_to_mesh(
        self, pcd: o3d.geometry.PointCloud, method: str = "poisson"
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """Convert point cloud to triangle mesh"""
        if pcd is None or len(pcd.points) < self.min_points:
            print(
                f"⚠️  Not enough points for mesh generation: {len(pcd.points) if pcd else 0}"
            )
            return None

        try:
            print(f"🔧 Generating mesh using {method} method...")

            if method == "poisson":
                # Poisson surface reconstruction
                mesh, densities = (
                    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=9, width=0, scale=1.1, linear_fit=False
                    )
                )

                # Remove low-density vertices (artifacts) - only if densities are provided
                if densities is not None and len(densities) > 0:
                    try:
                        # Convert densities to numpy array for comparison
                        densities_np = np.asarray(densities)
                        # Create mask for vertices to keep (high density)
                        vertices_to_remove = densities_np < np.quantile(
                            densities_np, 0.1
                        )
                        mesh.remove_vertices_by_mask(vertices_to_remove)
                    except Exception as e:
                        print(f"⚠️  Could not remove low-density vertices: {e}")
                        # Continue without removing vertices

            elif method == "ball_pivoting":
                # Ball pivoting algorithm
                # Estimate radius for ball pivoting
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist, avg_dist * 2, avg_dist * 4]

                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

            elif method == "alpha_shape":
                # Alpha shape (good for detailed surfaces)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.1
                )

            else:
                print(f"❌ Unknown mesh method: {method}")
                return None

            if mesh is None or len(mesh.triangles) == 0:
                print("⚠️  Mesh generation failed - no triangles created")
                return None

            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Smooth mesh
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)

            # Compute vertex normals for better rendering
            mesh.compute_vertex_normals()

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
        print("Controls: Press 'q' to quit early, 'm' to toggle mesh/pointcloud")

        # Initialize visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window("Kinect Real-time Mesh", width=1280, height=720)

        # Set up render options
        render_option = vis.get_render_option()
        render_option.show_coordinate_frame = True
        render_option.background_color = np.asarray([0.1, 0.1, 0.1])

        # Start capture
        if not self.capture.start_capture():
            print("❌ Failed to start Kinect capture")
            return

        try:
            start_time = time.time()
            frame_count = 0
            current_geometry = None
            show_mesh = True
            last_mesh_time = 0
            mesh_interval = 0.5  # Generate mesh every 0.5 seconds

            while time.time() - start_time < duration:
                # Capture frame
                frame = self.capture.capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                current_time = time.time()

                # Generate point cloud
                pcd = self.kinect_frame_to_pointcloud(frame)
                if pcd is None:
                    continue

                # Clean point cloud
                pcd = self.clean_pointcloud(pcd)

                # Generate mesh periodically (computationally expensive)
                display_geometry = pcd
                if show_mesh and (current_time - last_mesh_time) > mesh_interval:
                    mesh = self.pointcloud_to_mesh(pcd, method="poisson")
                    if mesh is not None:
                        display_geometry = mesh
                        last_mesh_time = current_time

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

                # Small sleep to prevent overwhelming the system
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n🛑 Capture interrupted by user")

        finally:
            self.capture.stop_capture()

        print(f"✅ Captured {len(mesh_sequence)} meshes")
        self.mesh_sequence = mesh_sequence
        return mesh_sequence

    def export_mesh_sequence(
        self,
        mesh_sequence: List[o3d.geometry.TriangleMesh],
        base_name: str = "kinect_mesh",
        fps: int = 5,
        duration: int = 10,
    ) -> List[Path]:
        """Export mesh sequence to files"""
        if not mesh_sequence:
            print("❌ No meshes to export")
            return []

        exported_files = []
        timestamp = int(time.time())

        print(f"💾 Exporting {len(mesh_sequence)} meshes...")

        for i, mesh in enumerate(mesh_sequence):
            try:
                # Export as OBJ (simple format, good for Blender)
                obj_filename = self.output_dir / f"{base_name}_{timestamp}_{i:04d}.obj"
                success = o3d.io.write_triangle_mesh(str(obj_filename), mesh)
                if success:
                    exported_files.append(obj_filename)

                # Export as PLY (preserves colors and properties)
                ply_filename = self.output_dir / f"{base_name}_{timestamp}_{i:04d}.ply"
                success = o3d.io.write_triangle_mesh(str(ply_filename), mesh)
                if success:
                    exported_files.append(ply_filename)

            except Exception as e:
                print(f"⚠️  Failed to export mesh {i}: {e}")

        # Create animation metadata
        metadata = {
            "timestamp": timestamp,
            "frame_count": len(mesh_sequence),
            "base_name": base_name,
            "formats": ["obj", "ply"],
            "fps": fps,
            "duration_seconds": duration,
            "notes": "Generated from Azure Kinect RGB-D data",
            "mesh_generation_method": "poisson",
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        metadata_file = self.output_dir / f"{base_name}_{timestamp}_metadata.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            exported_files.append(metadata_file)
        except Exception as e:
            print(f"⚠️  Failed to save metadata: {e}")

        print(f"✅ Exported to: {self.output_dir}")
        print(
            f"📁 Files: {len([f for f in exported_files if f.suffix in ['.obj', '.ply']])} mesh files + metadata"
        )

        return exported_files

    def create_blender_import_script(self, base_name: str, timestamp: int):
        """Generate a Blender Python script to import the mesh sequence"""
        script_content = f"""import bpy
import os
import mathutils

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import settings
base_path = r"{self.output_dir.absolute()}"
base_name = "{base_name}_{timestamp}"
frame_start = 1

print(f"Looking for files with pattern: {{base_name}}_*.obj in {{base_path}}")

# Get list of OBJ files
try:
    obj_files = sorted([f for f in os.listdir(base_path) 
                       if f.startswith(base_name) and f.endswith('.obj')])
    print(f"Found {{len(obj_files)}} mesh files")
except Exception as e:
    print(f"Error finding files: {{e}}")
    obj_files = []

if not obj_files:
    print("No mesh files found! Check the path and file pattern.")
else:
    # Check Blender version for import compatibility
    blender_version = bpy.app.version
    print(f"Blender version: {{blender_version}}")

    # Import each mesh as a keyframe
    for i, obj_file in enumerate(obj_files):
        file_path = os.path.join(base_path, obj_file)
        
        print(f"Importing: {{file_path}}")
        
        # Import mesh using version-appropriate method
        try:
            if blender_version >= (3, 0, 0):
                bpy.ops.wm.obj_import(filepath=file_path)
            else:
                bpy.ops.import_scene.obj(filepath=file_path)
        except Exception as e:
            print(f"Failed to import {{obj_file}}: {{e}}")
            continue
        
        # Get the imported object
        if bpy.context.selected_objects:
            imported_obj = bpy.context.selected_objects[-1]
            imported_obj.name = f"KinectMesh_{{i:04d}}"
            
            # Fix Kinect coordinate system (Y-up to Z-up conversion)
            imported_obj.rotation_euler = (1.5708, 0, 0)  # 90° around X-axis
            
            # Apply rotation
            bpy.context.view_layer.objects.active = imported_obj
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            
            # Apply smooth shading
            bpy.ops.object.shade_smooth()
            
            # Set keyframe visibility for animation
            frame_num = frame_start + i
            bpy.context.scene.frame_set(frame_num)
            
            # Hide all previous meshes
            for obj in bpy.data.objects:
                if obj.name.startswith("KinectMesh_") and obj != imported_obj:
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport")
                    obj.keyframe_insert(data_path="hide_render")
            
            # Show current mesh
            imported_obj.hide_viewport = False
            imported_obj.hide_render = False
            imported_obj.keyframe_insert(data_path="hide_viewport")
            imported_obj.keyframe_insert(data_path="hide_render")
            
            print(f"Imported frame {{frame_num}}: {{obj_file}} (orientation corrected)")

    # Set animation range
    if obj_files:
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_start + len(obj_files) - 1

        # Set to first frame and fit view
        bpy.context.scene.frame_set(frame_start)
        
        # Try to fit view (may not work in all contexts)
        try:
            bpy.ops.view3d.view_all()
        except:
            pass

        print("Animation setup complete!")
        print(f"Frame range: {{bpy.context.scene.frame_start}} - {{bpy.context.scene.frame_end}}")
        print("Press SPACEBAR to play animation!")
    else:
        print("No animation created - no files were imported.")
"""

        script_file = self.output_dir / f"import_to_blender_{timestamp}.py"
        try:
            with open(script_file, "w") as f:
                f.write(script_content)

            print(f"📝 Blender import script: {script_file}")
            print("   To use: Open Blender → Scripting → Run Script")
            print("   🔄 Includes automatic orientation correction and animation setup")
        except Exception as e:
            print(f"⚠️  Failed to create Blender script: {e}")

    def save_current_mesh(self, filename: str = None):
        """Save the current mesh from the last captured frame"""
        if not self.mesh_sequence:
            print("❌ No meshes captured yet")
            return

        if filename is None:
            timestamp = int(time.time())
            filename = f"kinect_mesh_current_{timestamp}.obj"

        filepath = self.output_dir / filename
        success = o3d.io.write_triangle_mesh(str(filepath), self.mesh_sequence[-1])

        if success:
            print(f"💾 Current mesh saved: {filepath}")
        else:
            print(f"❌ Failed to save mesh: {filepath}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate 3D meshes from Kinect data")
    parser.add_argument(
        "--mode",
        choices=["realtime", "sequence"],
        default="realtime",
        help="Visualization mode (default: realtime)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Capture duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Capture FPS for sequence mode (default: 5)"
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="Kinect device ID (default: 0)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export mesh sequence and create Blender import script",
    )
    parser.add_argument(
        "--method",
        choices=["poisson", "ball_pivoting", "alpha_shape"],
        default="poisson",
        help="Mesh generation method (default: poisson)",
    )

    args = parser.parse_args()

    print("🎯 Kinect to 3D Mesh Generator")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Duration: {args.duration}s")
    if args.mode == "sequence":
        print(f"FPS: {args.fps}")
    print(f"Device ID: {args.device_id}")
    print(f"Mesh method: {args.method}")
    print()

    # Create generator
    generator = KinectMeshGenerator(device_id=args.device_id)

    try:
        if args.mode == "realtime":
            generator.visualize_realtime(duration=args.duration)

        elif args.mode == "sequence":
            mesh_sequence = generator.capture_mesh_sequence(
                duration=args.duration, fps=args.fps
            )

            if args.export and mesh_sequence:
                print("\n📤 Exporting mesh sequence...")
                exported_files = generator.export_mesh_sequence(
                    mesh_sequence, fps=args.fps, duration=args.duration
                )

                if exported_files:
                    timestamp = int(time.time())
                    generator.create_blender_import_script("kinect_mesh", timestamp)

                    print(f"\n🎬 Animation ready!")
                    print(f"📁 Files exported to: {generator.output_dir}")
                    print(f"🎨 Import to Blender using the generated script")
                    print(
                        f"💡 Tip: In Blender, go to Scripting tab and run the import script"
                    )

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("✅ Mesh generation complete!")


if __name__ == "__main__":
    main()
