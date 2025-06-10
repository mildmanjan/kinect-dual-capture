"""
Kinect capture utilities and base classes
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode, WiredSyncMode, FPS

    KINECT_AVAILABLE = True
except ImportError:
    KINECT_AVAILABLE = False


class KinectSyncMode(Enum):
    """Kinect synchronization modes"""

    STANDALONE = "standalone"
    MASTER = "master"
    SUBORDINATE = "subordinate"


@dataclass
class KinectFrame:
    """Data class for Kinect frame data"""

    timestamp: float
    frame_id: int
    color_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    color_size: int = 0
    depth_size: int = 0
    total_size: int = 0

    def __post_init__(self):
        """Calculate sizes after initialization"""
        if self.color_image is not None:
            self.color_size = self.color_image.nbytes
        if self.depth_image is not None:
            self.depth_size = self.depth_image.nbytes
        self.total_size = self.color_size + self.depth_size


@dataclass
class KinectConfig:
    """Kinect device configuration"""

    color_resolution: ColorResolution = ColorResolution.RES_1080P
    depth_mode: DepthMode = DepthMode.NFOV_UNBINNED
    camera_fps: FPS = FPS.FPS_30  # Use proper FPS enum instead of int
    synchronized_images_only: bool = True
    depth_delay_off_color_usec: int = 0
    wired_sync_mode: WiredSyncMode = WiredSyncMode.STANDALONE
    subordinate_delay_off_master_usec: int = 0


class BaseKinectCapture:
    """Base class for Kinect capture functionality"""

    def __init__(
        self, device_id: int = 0, sync_mode: KinectSyncMode = KinectSyncMode.STANDALONE
    ):
        self.device_id = device_id
        self.sync_mode = sync_mode
        self.kinect = None
        self.is_running = False

        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.capture_stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "total_data_bytes": 0,
            "average_fps": 0.0,
        }

        # Configuration
        self.config = self._create_config()

    def _create_config(self) -> KinectConfig:
        """Create Kinect configuration based on sync mode"""
        config = KinectConfig()

        if self.sync_mode == KinectSyncMode.MASTER:
            config.wired_sync_mode = WiredSyncMode.MASTER
        elif self.sync_mode == KinectSyncMode.SUBORDINATE:
            config.wired_sync_mode = WiredSyncMode.SUBORDINATE
            config.subordinate_delay_off_master_usec = (
                160  # Small delay for subordinate
            )
        else:
            config.wired_sync_mode = WiredSyncMode.STANDALONE

        return config

    def initialize(self) -> bool:
        """Initialize the Kinect device"""
        if not KINECT_AVAILABLE:
            print(f"❌ pyk4a not available for device {self.device_id}")
            return False

        try:
            print(
                f"🔌 Initializing Kinect {self.device_id} ({self.sync_mode.value})..."
            )

            # Create pyk4a config
            pyk4a_config = Config(
                color_resolution=self.config.color_resolution,
                depth_mode=self.config.depth_mode,
                camera_fps=self.config.camera_fps,
                synchronized_images_only=self.config.synchronized_images_only,
                depth_delay_off_color_usec=self.config.depth_delay_off_color_usec,
                wired_sync_mode=self.config.wired_sync_mode,
                subordinate_delay_off_master_usec=self.config.subordinate_delay_off_master_usec,
            )

            self.kinect = PyK4A(config=pyk4a_config, device_id=self.device_id)
            self.kinect.start()

            print(f"✅ Kinect {self.device_id} initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize Kinect {self.device_id}: {e}")
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        if not self.kinect:
            return {}

        try:
            # pylint: disable=no-member
            calibration = self.kinect.calibration
            serial_number = getattr(self.kinect, "serial", "Unknown")

            # Basic device info
            device_info = {
                "device_id": self.device_id,
                "serial_number": serial_number,
                "sync_mode": self.sync_mode.value,
                "color_resolution": str(self.config.color_resolution),
                "depth_mode": str(self.config.depth_mode),
                "fps": self.config.camera_fps,
            }

            # Try to get camera calibration info (may not be available on all systems)
            try:
                color_cal = getattr(calibration, "color_camera_calibration", None)
                if color_cal:
                    color_intrinsics = getattr(color_cal, "intrinsics", None)
                    if color_intrinsics and hasattr(color_intrinsics, "parameters"):
                        params = color_intrinsics.parameters.param
                        device_info["color_camera"] = {
                            "resolution": f"{color_cal.resolution_width}x{color_cal.resolution_height}",
                            "fx": getattr(params, "fx", 0),
                            "fy": getattr(params, "fy", 0),
                            "cx": getattr(params, "cx", 0),
                            "cy": getattr(params, "cy", 0),
                        }
            except (AttributeError, TypeError):
                device_info["color_camera"] = {
                    "resolution": "1920x1080",
                    "info": "Calibration not available",
                }

            try:
                depth_cal = getattr(calibration, "depth_camera_calibration", None)
                if depth_cal:
                    depth_intrinsics = getattr(depth_cal, "intrinsics", None)
                    if depth_intrinsics and hasattr(depth_intrinsics, "parameters"):
                        params = depth_intrinsics.parameters.param
                        device_info["depth_camera"] = {
                            "resolution": f"{depth_cal.resolution_width}x{depth_cal.resolution_height}",
                            "fx": getattr(params, "fx", 0),
                            "fy": getattr(params, "fy", 0),
                            "cx": getattr(params, "cx", 0),
                            "cy": getattr(params, "cy", 0),
                        }
            except (AttributeError, TypeError):
                device_info["depth_camera"] = {
                    "resolution": "640x576",
                    "info": "Calibration not available",
                }

            return device_info

        except Exception as e:
            print(f"⚠️  Could not get device info for Kinect {self.device_id}: {e}")
            return {"device_id": self.device_id, "error": str(e)}

    def capture_frame(self) -> Optional[KinectFrame]:
        """Capture a single frame"""
        if not self.kinect or not self.is_running:
            return None

        try:
            capture = self.kinect.get_capture()

            color_image = None
            depth_image = None

            # Process color image
            if capture.color is not None:
                # Convert BGRA to BGR for OpenCV
                color_image = capture.color[:, :, :3]

            # Process depth image
            if capture.depth is not None:
                depth_image = capture.depth

            # Create frame object
            frame = KinectFrame(
                timestamp=time.time(),
                frame_id=self.frame_count,
                color_image=color_image,
                depth_image=depth_image,
            )

            self.frame_count += 1
            self.capture_stats["frames_captured"] += 1
            self.capture_stats["total_data_bytes"] += frame.total_size

            return frame

        except Exception as e:
            print(f"❌ Error capturing frame from Kinect {self.device_id}: {e}")
            self.capture_stats["frames_dropped"] += 1
            return None

    def start_capture(self) -> bool:
        """Start capture process"""
        if not self.initialize():
            return False

        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0

        # Reset stats
        self.capture_stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "total_data_bytes": 0,
            "average_fps": 0.0,
        }

        return True

    def stop_capture(self):
        """Stop capture process"""
        self.is_running = False
        if self.kinect:
            self.kinect.stop()

        # Calculate final stats
        if self.start_time:
            total_time = time.time() - self.start_time
            if total_time > 0:
                self.capture_stats["average_fps"] = (
                    self.capture_stats["frames_captured"] / total_time
                )

        print(f"🛑 Kinect {self.device_id} capture stopped")

    def get_capture_stats(self) -> Dict[str, Any]:
        """Get current capture statistics"""
        stats = self.capture_stats.copy()

        if self.start_time:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            if elapsed_time > 0:
                stats["current_fps"] = (
                    self.capture_stats["frames_captured"] / elapsed_time
                )
                stats["data_rate_mbps"] = (
                    self.capture_stats["total_data_bytes"] / elapsed_time
                ) / (1024 * 1024)
                stats["elapsed_time"] = elapsed_time

        return stats

    def save_frame(self, frame: KinectFrame, output_dir: Path, prefix: str = ""):
        """Save a frame to disk"""
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp_str = str(int(frame.timestamp))
        device_prefix = (
            f"device{self.device_id}_{prefix}" if prefix else f"device{self.device_id}"
        )

        # Save color image
        if frame.color_image is not None:
            color_path = output_dir / f"{device_prefix}_color_{timestamp_str}.jpg"
            cv2.imwrite(str(color_path), frame.color_image)
            print(f"💾 Saved color: {color_path}")

        # Save depth image (normalized for visualization)
        if frame.depth_image is not None:
            depth_normalized = cv2.normalize(
                frame.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_path = output_dir / f"{device_prefix}_depth_{timestamp_str}.png"
            cv2.imwrite(str(depth_path), depth_normalized)

            # Save raw depth data
            depth_raw_path = (
                output_dir / f"{device_prefix}_depth_raw_{timestamp_str}.npy"
            )
            np.save(str(depth_raw_path), frame.depth_image)
            print(f"💾 Saved depth: {depth_path} (and raw data)")


def test_kinect_connection(device_id: int = 0) -> bool:
    """Quick test to see if Kinect is connected"""
    try:
        print(f"🔍 Testing Kinect {device_id} connection...")
        capture = BaseKinectCapture(device_id=device_id)
        if capture.initialize():
            print(f"✅ Kinect {device_id} connection successful!")
            capture.stop_capture()
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Kinect {device_id} connection failed: {e}")
        return False
