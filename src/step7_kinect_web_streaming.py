#!/usr/bin/env python3
"""
Step 7: Kinect to Web Browser Streaming
Stream 3D point cloud data from Kinect to web browser in real-time.

This script creates:
- WebSocket server for real-time 3D data streaming
- HTTP server for web interface
- Point cloud processing and compression
- Browser-based 3D visualization
"""

import asyncio
import websockets
import json
import numpy as np
import cv2
import time
import threading
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List
import base64
import struct

# Web server imports
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from functools import partial

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.kinect_capture import BaseKinectCapture, KinectSyncMode, KinectFrame
from utils.compression_utils import DataCompressor, CompressionLevel


class KinectWebStreamer:
    """Stream Kinect 3D data to web browsers"""

    def __init__(self, device_id: int = 0, port: int = 8765, web_port: int = 8000):
        self.device_id = device_id
        self.port = port
        self.web_port = web_port

        # Kinect capture
        self.capture = BaseKinectCapture(
            device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
        )
        self.compressor = DataCompressor(
            CompressionLevel.HIGH
        )  # High compression for web

        # Streaming state
        self.is_streaming = False
        self.connected_clients = set()
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Point cloud settings
        self.max_points = 5000  # Limit points for web performance
        self.downsample_factor = 4  # Skip every N pixels
        self.depth_min = 0.3  # Minimum depth in meters
        self.depth_max = 2.0  # Maximum depth in meters

        # Camera intrinsics (approximate for depth camera)
        self.fx = 504.0
        self.fy = 504.0
        self.cx = 320.0
        self.cy = 288.0

        # Create web files
        self.create_web_files()

    def create_web_files(self):
        """Create HTML and JavaScript files for 3D visualization"""
        web_dir = Path("data/web_viewer")
        web_dir.mkdir(parents=True, exist_ok=True)

        # Create HTML file
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Kinect 3D Web Viewer</title>
    <style>
        body { margin: 0; padding: 0; background: #000; color: #fff; font-family: Arial; }
        #container { position: relative; width: 100vw; height: 100vh; }
        #info { position: absolute; top: 10px; left: 10px; z-index: 100; }
        #controls { position: absolute; top: 10px; right: 10px; z-index: 100; }
        button { margin: 5px; padding: 10px; background: #333; color: #fff; border: none; cursor: pointer; }
        button:hover { background: #555; }
        #status { color: #0f0; }
    </style>
</head>
<body>
    <div id="container">
        <div id="info">
            <h3>Kinect 3D Web Viewer</h3>
            <div id="status">Connecting...</div>
            <div id="stats">Points: 0 | FPS: 0</div>
        </div>
        <div id="controls">
            <button onclick="togglePointSize()">Toggle Point Size</button>
            <button onclick="toggleColors()">Toggle Colors</button>
            <button onclick="resetView()">Reset View</button>
            <button onclick="toggleAutoRotate()">Auto Rotate</button>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/controls/OrbitControls.js"></script>
    <script src="kinect_viewer.js"></script>
</body>
</html>"""

        with open(web_dir / "index.html", "w") as f:
            f.write(html_content)

        # Create JavaScript file
        js_content = """
// Kinect 3D Web Viewer
let scene, camera, renderer, controls;
let pointCloud, pointsMaterial, pointsGeometry;
let socket;
let frameCount = 0;
let lastTime = Date.now();

// Settings
let pointSize = 0.01;
let showColors = true;
let autoRotate = false;

init();
animate();

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    // Camera setup
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 1);
    
    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('container').appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Initialize point cloud
    pointsGeometry = new THREE.BufferGeometry();
    pointsMaterial = new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: true,
        sizeAttenuation: true
    });
    pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointCloud);
    
    // WebSocket connection
    connectWebSocket();
    
    // Window resize handler
    window.addEventListener('resize', onWindowResize, false);
}

function connectWebSocket() {
    const wsUrl = `ws://localhost:8765`;
    socket = new WebSocket(wsUrl);
    
    socket.onopen = function(event) {
        document.getElementById('status').textContent = 'Connected';
        document.getElementById('status').style.color = '#0f0';
    };
    
    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            updatePointCloud(data);
            updateStats();
        } catch (error) {
            console.error('Error parsing data:', error);
        }
    };
    
    socket.onclose = function(event) {
        document.getElementById('status').textContent = 'Disconnected';
        document.getElementById('status').style.color = '#f00';
        
        // Try to reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
    };
    
    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
        document.getElementById('status').textContent = 'Connection Error';
        document.getElementById('status').style.color = '#f00';
    };
}

function updatePointCloud(data) {
    if (!data.points || data.points.length === 0) {
        return;
    }
    
    const numPoints = data.points.length / 3;
    
    // Convert flat array to Vector3 array
    const positions = new Float32Array(data.points);
    const colors = new Float32Array(data.colors || []);
    
    // Update geometry
    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors.length > 0 && showColors) {
        pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    
    pointsGeometry.computeBoundingSphere();
    pointsGeometry.attributes.position.needsUpdate = true;
    if (pointsGeometry.attributes.color) {
        pointsGeometry.attributes.color.needsUpdate = true;
    }
    
    frameCount++;
}

function updateStats() {
    const now = Date.now();
    const deltaTime = now - lastTime;
    
    if (deltaTime >= 1000) { // Update every second
        const fps = Math.round(frameCount * 1000 / deltaTime);
        const pointCount = pointsGeometry.attributes.position ? 
                          pointsGeometry.attributes.position.count : 0;
        
        document.getElementById('stats').textContent = 
            `Points: ${pointCount} | FPS: ${fps}`;
        
        frameCount = 0;
        lastTime = now;
    }
}

function animate() {
    requestAnimationFrame(animate);
    
    if (autoRotate) {
        pointCloud.rotation.y += 0.01;
    }
    
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Control functions
function togglePointSize() {
    pointSize = pointSize === 0.01 ? 0.02 : 0.01;
    pointsMaterial.size = pointSize;
}

function toggleColors() {
    showColors = !showColors;
    pointsMaterial.vertexColors = showColors;
    pointsMaterial.needsUpdate = true;
}

function resetView() {
    camera.position.set(0, 0, 1);
    controls.reset();
}

function toggleAutoRotate() {
    autoRotate = !autoRotate;
}
"""

        with open(web_dir / "kinect_viewer.js", "w") as f:
            f.write(js_content)

        print(f"📁 Web files created in: {web_dir}")

    def depth_to_pointcloud(self, frame: KinectFrame) -> Optional[Dict[str, Any]]:
        """Convert Kinect frame to point cloud data for web"""
        if frame.color_image is None or frame.depth_image is None:
            return None

        try:
            # Get image dimensions
            depth_height, depth_width = frame.depth_image.shape
            color_height, color_width = frame.color_image.shape[:2]

            # Resize color to match depth
            color_resized = cv2.resize(
                frame.color_image, (depth_width, depth_height)
            )  # pylint: disable=no-member

            # Downsample for performance
            depth_down = frame.depth_image[
                :: self.downsample_factor, :: self.downsample_factor
            ]
            color_down = color_resized[
                :: self.downsample_factor, :: self.downsample_factor
            ]

            # Get valid depth pixels
            valid_mask = (depth_down > self.depth_min * 1000) & (
                depth_down < self.depth_max * 1000
            )

            if not np.any(valid_mask):
                return None

            # Get pixel coordinates
            h, w = depth_down.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Scale coordinates back to original resolution for camera calculations
            u_scaled = u * self.downsample_factor
            v_scaled = v * self.downsample_factor

            # Convert to 3D points using camera intrinsics
            z = depth_down.astype(np.float32) / 1000.0  # Convert mm to meters
            x = (u_scaled - self.cx) * z / self.fx
            y = (v_scaled - self.cy) * z / self.fy

            # Apply valid mask
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]

            # Get colors
            color_valid = color_down[valid_mask]

            # Limit number of points for web performance
            if len(x_valid) > self.max_points:
                indices = np.random.choice(len(x_valid), self.max_points, replace=False)
                x_valid = x_valid[indices]
                y_valid = y_valid[indices]
                z_valid = z_valid[indices]
                color_valid = color_valid[indices]

            # Create point cloud data
            points = []
            colors = []

            for i in range(len(x_valid)):
                # Coordinate system conversion (Kinect to web)
                points.extend([x_valid[i], -y_valid[i], -z_valid[i]])  # Flip Y and Z

                # Color (BGR to RGB, normalize to 0-1)
                b, g, r = color_valid[i]
                colors.extend([r / 255.0, g / 255.0, b / 255.0])

            return {
                "points": points,
                "colors": colors,
                "timestamp": time.time(),
                "point_count": len(points) // 3,
            }

        except Exception as e:
            print(f"❌ Error creating point cloud: {e}")
            return None

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.connected_clients.add(websocket)
        print(f"🔗 Client connected. Total clients: {len(self.connected_clients)}")

        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            print(
                f"🔌 Client disconnected. Total clients: {len(self.connected_clients)}"
            )

    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """Broadcast frame data to all connected clients"""
        if not self.connected_clients:
            return

        message = json.dumps(frame_data)
        disconnected = []

        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
            except Exception as e:
                print(f"⚠️  Error sending to client: {e}")
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
            self.connected_clients.discard(client)

    def capture_thread(self):
        """Kinect capture thread"""
        if not self.capture.start_capture():
            print("❌ Failed to start Kinect capture")
            return

        print("🎥 Kinect capture thread started")

        try:
            while self.is_streaming:
                frame = self.capture.capture_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame

                time.sleep(1 / 30)  # ~30 FPS max

        except Exception as e:
            print(f"❌ Capture thread error: {e}")

        finally:
            self.capture.stop_capture()
            print("🛑 Kinect capture thread stopped")

    async def streaming_loop(self):
        """Main streaming loop"""
        print("🌐 Starting streaming loop...")

        while self.is_streaming:
            if self.current_frame is not None:
                with self.frame_lock:
                    frame = self.current_frame

                # Convert to point cloud
                point_cloud_data = self.depth_to_pointcloud(frame)

                if point_cloud_data is not None:
                    await self.broadcast_frame(point_cloud_data)

            await asyncio.sleep(1 / 15)  # ~15 FPS for web streaming

    def start_web_server(self):
        """Start HTTP server for web files"""
        web_dir = Path("data/web_viewer")

        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(web_dir), **kwargs)

        httpd = HTTPServer(("localhost", self.web_port), CustomHandler)

        def serve_forever():
            print(f"🌐 Web server running at http://localhost:{self.web_port}")
            httpd.serve_forever()

        web_thread = threading.Thread(target=serve_forever, daemon=True)
        web_thread.start()

        return httpd

    async def start_streaming(self):
        """Start the complete streaming system"""
        print("🚀 Starting Kinect Web Streaming System")
        print("=" * 50)

        # Start web server
        web_server = self.start_web_server()

        # Start Kinect capture thread
        self.is_streaming = True
        capture_thread = threading.Thread(target=self.capture_thread, daemon=True)
        capture_thread.start()

        # Start WebSocket server
        start_server = websockets.serve(self.handle_client, "localhost", self.port)

        print(f"🔌 WebSocket server starting on ws://localhost:{self.port}")
        print(f"🌐 Web interface available at http://localhost:{self.web_port}")
        print("🎯 Opening web browser...")

        # Open browser
        webbrowser.open(f"http://localhost:{self.web_port}")

        # Start servers
        await asyncio.gather(start_server, self.streaming_loop())


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Stream Kinect 3D data to web browser")
    parser.add_argument("--device-id", type=int, default=0, help="Kinect device ID")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--web-port", type=int, default=8000, help="Web server port")
    parser.add_argument(
        "--max-points", type=int, default=5000, help="Max points for web performance"
    )

    args = parser.parse_args()

    streamer = KinectWebStreamer(
        device_id=args.device_id, port=args.port, web_port=args.web_port
    )
    streamer.max_points = args.max_points

    try:
        await streamer.start_streaming()
    except KeyboardInterrupt:
        print("\n🛑 Streaming stopped by user")
        streamer.is_streaming = False


if __name__ == "__main__":
    # Install required packages
    try:
        import websockets
    except ImportError:
        print("❌ websockets package required. Install with: pip install websockets")
        sys.exit(1)

    asyncio.run(main())
