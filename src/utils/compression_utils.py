"""
Compression utilities for reducing Kinect data bandwidth
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import time


class CompressionLevel(Enum):
    """Compression level settings"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class CompressionMethod(Enum):
    """Available compression methods"""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    DEPTH_CUSTOM = "depth_custom"


@dataclass
class CompressionSettings:
    """Compression configuration"""

    color_method: CompressionMethod = CompressionMethod.JPEG
    color_quality: int = 85
    depth_method: CompressionMethod = CompressionMethod.PNG
    depth_quantization: int = 16  # Reduce depth precision
    resize_factor: float = 1.0  # 1.0 = no resize, 0.5 = half size


@dataclass
class CompressionResult:
    """Results from compression operation"""

    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float = 0.0


class DataCompressor:
    """Handles compression of Kinect data"""

    # Predefined compression levels
    COMPRESSION_PRESETS = {
        CompressionLevel.NONE: CompressionSettings(
            color_method=CompressionMethod.PNG,
            color_quality=100,
            depth_quantization=1,
            resize_factor=1.0,
        ),
        CompressionLevel.LOW: CompressionSettings(
            color_method=CompressionMethod.JPEG,
            color_quality=95,
            depth_quantization=4,
            resize_factor=1.0,
        ),
        CompressionLevel.MEDIUM: CompressionSettings(
            color_method=CompressionMethod.JPEG,
            color_quality=85,
            depth_quantization=8,
            resize_factor=0.8,
        ),
        CompressionLevel.HIGH: CompressionSettings(
            color_method=CompressionMethod.JPEG,
            color_quality=70,
            depth_quantization=16,
            resize_factor=0.6,
        ),
        CompressionLevel.EXTREME: CompressionSettings(
            color_method=CompressionMethod.WEBP,
            color_quality=50,
            depth_quantization=32,
            resize_factor=0.4,
        ),
    }

    def __init__(self, compression_level: CompressionLevel = CompressionLevel.MEDIUM):
        self.settings = self.COMPRESSION_PRESETS[compression_level]
        self.compression_level = compression_level

    def compress_color_image(self, image: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Compress color image and return bytes + metadata"""
        if image is None:
            return b"", {}

        start_time = time.time()
        original_size = image.nbytes

        # Resize if needed
        processed_image = image
        if self.settings.resize_factor != 1.0:
            new_height = int(image.shape[0] * self.settings.resize_factor)
            new_width = int(image.shape[1] * self.settings.resize_factor)
            processed_image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        # Initialize variables
        success = False
        encoded = None

        # Compress based on method
        if self.settings.color_method == CompressionMethod.JPEG:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.settings.color_quality]
            success, encoded = cv2.imencode(".jpg", processed_image, encode_params)
        elif self.settings.color_method == CompressionMethod.PNG:
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            success, encoded = cv2.imencode(".png", processed_image, encode_params)
        elif self.settings.color_method == CompressionMethod.WEBP:
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, self.settings.color_quality]
            success, encoded = cv2.imencode(".webp", processed_image, encode_params)
        else:
            # Unknown compression method
            print(f"⚠️  Unknown color compression method: {self.settings.color_method}")
            success = False

        compression_time = time.time() - start_time

        if not success or encoded is None:
            return b"", {
                "error": "Compression failed",
                "compression_time": compression_time,
            }

        compressed_data = encoded.tobytes()

        metadata = {
            "original_shape": image.shape,
            "processed_shape": processed_image.shape,
            "method": self.settings.color_method.value,
            "quality": self.settings.color_quality,
            "resize_factor": self.settings.resize_factor,
            "compressed_size": len(compressed_data),
            "original_size": original_size,
            "compression_ratio": (
                original_size / len(compressed_data) if len(compressed_data) > 0 else 0
            ),
            "compression_time": compression_time,
        }

        return compressed_data, metadata

    def compress_depth_image(
        self, depth_image: np.ndarray
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress depth image with custom quantization"""
        if depth_image is None:
            return b"", {}

        start_time = time.time()
        original_shape = depth_image.shape
        original_size = depth_image.nbytes

        # Resize if needed
        processed_depth = depth_image
        if self.settings.resize_factor != 1.0:
            new_height = int(depth_image.shape[0] * self.settings.resize_factor)
            new_width = int(depth_image.shape[1] * self.settings.resize_factor)
            processed_depth = cv2.resize(
                depth_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            )

        # Quantize depth values to reduce precision
        if self.settings.depth_quantization > 1:
            # Reduce bit depth by quantization
            depth_quantized = (
                processed_depth // self.settings.depth_quantization
            ).astype(np.uint16)
            # Clamp to fit in reduced bit range
            max_val = 65535 // self.settings.depth_quantization
            depth_quantized = np.clip(depth_quantized, 0, max_val)
        else:
            depth_quantized = processed_depth

        # Compress with PNG (lossless for the quantized data)
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        success, encoded = cv2.imencode(".png", depth_quantized, encode_params)

        compression_time = time.time() - start_time

        if not success or encoded is None:
            return b"", {
                "error": "Depth compression failed",
                "compression_time": compression_time,
            }

        compressed_data = encoded.tobytes()

        metadata = {
            "original_shape": original_shape,
            "processed_shape": processed_depth.shape,
            "quantization": self.settings.depth_quantization,
            "resize_factor": self.settings.resize_factor,
            "compressed_size": len(compressed_data),
            "original_size": original_size,
            "compression_ratio": (
                original_size / len(compressed_data) if len(compressed_data) > 0 else 0
            ),
            "compression_time": compression_time,
            "dtype": str(depth_image.dtype),
        }

        return compressed_data, metadata

    def decompress_color_image(
        self, compressed_data: bytes, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Decompress color image from bytes"""
        if not compressed_data:
            return None

        start_time = time.time()

        # Decode image
        nparr = np.frombuffer(compressed_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return None

        # Resize back to original if needed
        if metadata.get("resize_factor", 1.0) != 1.0:
            original_shape = metadata["original_shape"]
            image = cv2.resize(
                image,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        return image

    def decompress_depth_image(
        self, compressed_data: bytes, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Decompress depth image from bytes"""
        if not compressed_data:
            return None

        start_time = time.time()

        # Decode image
        nparr = np.frombuffer(compressed_data, np.uint8)
        depth_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if depth_image is None:
            return None

        # Restore quantization
        quantization = metadata.get("quantization", 1)
        if quantization > 1:
            depth_image = depth_image * quantization

        # Resize back to original if needed
        if metadata.get("resize_factor", 1.0) != 1.0:
            original_shape = metadata["original_shape"]
            depth_image = cv2.resize(
                depth_image,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return depth_image

    def compress_frame(
        self, color_image: np.ndarray, depth_image: np.ndarray
    ) -> Tuple[bytes, bytes, Dict[str, Any]]:
        """Compress both color and depth images"""
        # Compress color
        color_compressed, color_meta = self.compress_color_image(color_image)

        # Compress depth
        depth_compressed, depth_meta = self.compress_depth_image(depth_image)

        # Combined metadata
        total_original = color_meta.get("original_size", 0) + depth_meta.get(
            "original_size", 0
        )
        total_compressed = len(color_compressed) + len(depth_compressed)

        combined_meta = {
            "color_metadata": color_meta,
            "depth_metadata": depth_meta,
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "total_compression_ratio": (
                total_original / total_compressed if total_compressed > 0 else 0
            ),
            "compression_level": self.compression_level.value,
            "total_compression_time": color_meta.get("compression_time", 0)
            + depth_meta.get("compression_time", 0),
        }

        return color_compressed, depth_compressed, combined_meta

    def get_compression_info(self) -> Dict[str, Any]:
        """Get current compression settings info"""
        return {
            "level": self.compression_level.value,
            "color_method": self.settings.color_method.value,
            "color_quality": self.settings.color_quality,
            "depth_quantization": self.settings.depth_quantization,
            "resize_factor": self.settings.resize_factor,
        }

    def estimate_bandwidth_reduction(
        self, original_fps: float, original_size_mb: float
    ) -> Dict[str, float]:
        """Estimate bandwidth reduction based on settings"""
        # Rough estimates based on typical compression ratios
        color_ratio_estimates = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.LOW: 8.0,
            CompressionLevel.MEDIUM: 15.0,
            CompressionLevel.HIGH: 25.0,
            CompressionLevel.EXTREME: 40.0,
        }

        depth_ratio_estimates = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.LOW: 3.0,
            CompressionLevel.MEDIUM: 5.0,
            CompressionLevel.HIGH: 8.0,
            CompressionLevel.EXTREME: 12.0,
        }

        estimated_color_ratio = color_ratio_estimates.get(self.compression_level, 1.0)
        estimated_depth_ratio = depth_ratio_estimates.get(self.compression_level, 1.0)

        # Apply resize factor
        resize_reduction = self.settings.resize_factor**2

        # Rough estimate: 70% color, 30% depth by size
        overall_ratio = (
            0.7 * estimated_color_ratio + 0.3 * estimated_depth_ratio
        ) * resize_reduction

        compressed_size_mb = original_size_mb / overall_ratio
        compressed_bandwidth_mbps = (
            compressed_size_mb * original_fps * 8
        )  # Convert to Mbps

        return {
            "original_bandwidth_mbps": original_size_mb * original_fps * 8,
            "compressed_bandwidth_mbps": compressed_bandwidth_mbps,
            "estimated_ratio": overall_ratio,
            "bandwidth_reduction": (original_size_mb * original_fps * 8)
            / compressed_bandwidth_mbps,
        }
