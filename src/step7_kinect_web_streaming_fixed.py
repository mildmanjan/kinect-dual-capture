#!/usr/bin/env python3
"""
Fixed Step 7: Kinect to Web Browser Streaming
Clear separation of HTTP server (port 8000) and WebSocket server (port 8765)
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
from typing import Optional, Dict, Any
import webbrowser

# Add utils to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from utils.kinect_capture import BaseKinectCapture, KinectSyncMode, KinectFrame
    from utils.compression_utils import DataCompressor, CompressionLevel

    KINECT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Kinect utilities not available: {e}")
    KINECT_AVAILABLE = False

# Web server imports
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver


class FixedKinectWebStreamer:
    """Fixed Kinect web streamer with clear port separation"""

    def __init__(
        self, device_id: int = 0, websocket_port: int = 8765, http_port: int = 8000
    ):
        self.device_id = device_id
        self.websocket_port = websocket_port
        self.http_port = http_port

        # Kinect capture
        if KINECT_AVAILABLE:
            self.capture = BaseKinectCapture(
                device_id=device_id, sync_mode=KinectSyncMode.STANDALONE
            )
        else:
            self.capture = None

        # Streaming state
        self.is_streaming = False
        self.connected_clients = set()
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Point cloud settings
        self.max_points = 3000  # Reduced for better performance
        self.downsample_factor = 6  # Increased downsampling
        self.depth_min = 0.3  # Minimum depth in meters
        self.depth_max = 2.0  # Maximum depth in meters

        # Camera intrinsics (approximate for depth camera)
        self.fx = 504.0
        self.fy = 504.0
        self.cx = 320.0
        self.cy = 288.0

        # Create web files
        self.web_dir = self.create_web_files()

    def create_web_files(self):
        """Create improved HTML and JavaScript files"""
        web_dir = Path("data/web_viewer")
        web_dir.mkdir(parents=True, exist_ok=True)

        # Create improved HTML file
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Kinect 3D Web Viewer</title>
    <style>
        body {{ margin: 0; padding: 0; background: #000; color: #fff; font-family: Arial; overflow: hidden; }}
        #container {{ position: relative; width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; z-index: 100; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; }}
        #controls {{ position: absolute; top: 10px; right: 10px; z-index: 100; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; }}
        button {{ margin: 5px; padding: 10px; background: #333; color: #fff; border: none; cursor: pointer; border-radius: 3px; }}
        button:hover {{ background: #555; }}
        #status {{ font-size: 16px; font-weight: bold; }}
        #status.connected {{ color: #0f0; }}
        #status.disconnected {{ color: #f00; }}
        #status.connecting {{ color: #ff0; }}
        .error {{ color: #f88; }}
        .success {{ color: #8f8; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="info">
            <h3>Kinect 3D Web Viewer</h3>
            <div id="status" class="connecting">Connecting...</div>
            <div id="stats">Points: 0 | FPS: 0 | Messages: 0</div>
            <div id="debug"></div>
            <div id="instructions">
                <small>
                    Mouse: Rotate view<br>
                    Wheel: Zoom<br>
                    Controls: Use buttons ->
                </small>
            </div>
        </div>
        <div id="controls">
            <button onclick="togglePointSize()">Point Size</button>
            <button onclick="toggleColors()">Colors</button>
            <button onclick="resetView()">Reset View</button>
            <button onclick="toggleAutoRotate()">Auto Rotate</button>
            <button onclick="reconnectWebSocket()">Reconnect</button>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Global variables
        let scene, camera, renderer, controls;
        let pointCloud, pointsMaterial, pointsGeometry;
        let socket;
        let frameCount = 0;
        let messageCount = 0;
        let lastTime = Date.now();

        // Settings
        let pointSize = 0.02;
        let showColors = true;
        let autoRotate = false;

        // WebSocket settings
        const WEBSOCKET_URL = 'ws://localhost:{self.websocket_port}';
        let reconnectAttempts = 0;
        const MAX_RECONNECT_ATTEMPTS = 10;

        init();
        animate();

        function init() {{
            console.log('🚀 Initializing 3D viewer...');
            
            // Check WebGL support
            if (!window.WebGLRenderingContext) {{
                document.getElementById('status').innerHTML = '❌ WebGL not supported';
                return;
            }}

            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x001122);
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 1.5);
            
            // Renderer setup
            try {{
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.getElementById('container').appendChild(renderer.domElement);
                console.log('✅ WebGL renderer created');
            }} catch (error) {{
                console.error('❌ Failed to create WebGL renderer:', error);
                document.getElementById('status').innerHTML = '❌ WebGL failed';
                return;
            }}
            
            // Controls (simplified since OrbitControls might not load from CDN)
            let isDragging = false;
            let mouseX = 0, mouseY = 0;
            
            renderer.domElement.addEventListener('mousedown', (e) => {{
                isDragging = true;
                mouseX = e.clientX;
                mouseY = e.clientY;
            }});
            
            renderer.domElement.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    const deltaX = e.clientX - mouseX;
                    const deltaY = e.clientY - mouseY;
                    
                    camera.position.x = Math.cos(deltaX * 0.01) * 1.5;
                    camera.position.z = Math.sin(deltaX * 0.01) * 1.5;
                    camera.position.y += deltaY * 0.001;
                    
                    camera.lookAt(0, 0, 0);
                    
                    mouseX = e.clientX;
                    mouseY = e.clientY;
                }}
            }});
            
            renderer.domElement.addEventListener('mouseup', () => {{
                isDragging = false;
            }});
            
            renderer.domElement.addEventListener('wheel', (e) => {{
                const scale = e.deltaY > 0 ? 1.1 : 0.9;
                camera.position.multiplyScalar(scale);
                camera.lookAt(0, 0, 0);
                e.preventDefault();
            }});
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Initialize point cloud
            pointsGeometry = new THREE.BufferGeometry();
            pointsMaterial = new THREE.PointsMaterial({{
                size: pointSize,
                vertexColors: true,
                sizeAttenuation: true
            }});
            pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
            scene.add(pointCloud);
            
            console.log('3D scene initialized');
            
            // WebSocket connection
            connectWebSocket();
            
            // Window resize handler
            window.addEventListener('resize', onWindowResize, false);
        }}

        function connectWebSocket() {{
            console.log(`🔌 Connecting to ${{WEBSOCKET_URL}}...`);
            
            try {{
                socket = new WebSocket(WEBSOCKET_URL);
                
                socket.onopen = function(event) {{
                    console.log('✅ WebSocket connected');
                    document.getElementById('status').innerHTML = '✅ Connected';
                    document.getElementById('status').className = 'connected';
                    reconnectAttempts = 0;
                }};
                
                socket.onmessage = function(event) {{
                    try {{
                        const data = JSON.parse(event.data);
                        updatePointCloud(data);
                        updateStats();
                        messageCount++;
                    }} catch (error) {{
                        console.error('❌ Error parsing data:', error);
                        document.getElementById('debug').innerHTML = `Parse error: ${{error.message}}`;
                    }}
                }};
                
                socket.onclose = function(event) {{
                    console.log('🔌 WebSocket disconnected');
                    document.getElementById('status').innerHTML = '🔌 Disconnected';
                    document.getElementById('status').className = 'disconnected';
                    
                    // Auto-reconnect
                    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {{
                        reconnectAttempts++;
                        console.log(`🔄 Reconnecting... attempt ${{reconnectAttempts}}`);
                        setTimeout(connectWebSocket, 2000);
                    }} else {{
                        document.getElementById('status').innerHTML = '❌ Connection failed';
                        document.getElementById('debug').innerHTML = 'Max reconnect attempts reached. Click Reconnect button.';
                    }}
                }};
                
                socket.onerror = function(error) {{
                    console.error('❌ WebSocket error:', error);
                    document.getElementById('status').innerHTML = '❌ Connection Error';
                    document.getElementById('status').className = 'disconnected';
                    document.getElementById('debug').innerHTML = `WebSocket error: ${{error.message || 'Unknown error'}}`;
                }};
                
            }} catch (error) {{
                console.error('❌ Failed to create WebSocket:', error);
                document.getElementById('status').innerHTML = '❌ WebSocket Failed';
                document.getElementById('debug').innerHTML = `Failed to create WebSocket: ${{error.message}}`;
            }}
        }}

        function updatePointCloud(data) {{
            if (!data.points || data.points.length === 0) {{
                console.log('⚠️ No points in data');
                return;
            }}
            
            const numPoints = data.points.length / 3;
            console.log(`📊 Received ${{numPoints}} points`);
            
            // Convert flat array to Vector3 array
            const positions = new Float32Array(data.points);
            const colors = data.colors ? new Float32Array(data.colors) : null;
            
            // Update geometry
            pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            if (colors && showColors) {{
                pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                pointsMaterial.vertexColors = true;
            }} else {{
                pointsMaterial.vertexColors = false;
                pointsMaterial.color.setHex(0xff4444);
            }}
            
            pointsGeometry.computeBoundingSphere();
            pointsGeometry.attributes.position.needsUpdate = true;
            if (pointsGeometry.attributes.color) {{
                pointsGeometry.attributes.color.needsUpdate = true;
            }}
            
            frameCount++;
        }}

        function updateStats() {{
            const now = Date.now();
            const deltaTime = now - lastTime;
            
            if (deltaTime >= 1000) {{ // Update every second
                const fps = Math.round(frameCount * 1000 / deltaTime);
                const pointCount = pointsGeometry.attributes.position ? 
                                  pointsGeometry.attributes.position.count : 0;
                
                document.getElementById('stats').innerHTML = 
                    `Points: ${{pointCount}} | FPS: ${{fps}} | Messages: ${{messageCount}}`;
                
                frameCount = 0;
                lastTime = now;
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);
            
            if (autoRotate && pointCloud) {{
                pointCloud.rotation.y += 0.01;
            }}
            
            if (renderer && scene && camera) {{
                renderer.render(scene, camera);
            }}
        }}

        function onWindowResize() {{
            if (camera && renderer) {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}
        }}

        // Control functions
        function togglePointSize() {{
            pointSize = pointSize === 0.02 ? 0.05 : 0.02;
            if (pointsMaterial) {{
                pointsMaterial.size = pointSize;
            }}
        }}

        function toggleColors() {{
            showColors = !showColors;
            if (pointsMaterial) {{
                pointsMaterial.vertexColors = showColors;
                pointsMaterial.needsUpdate = true;
            }}
        }}

        function resetView() {{
            if (camera) {{
                camera.position.set(0, 0, 1.5);
                camera.lookAt(0, 0, 0);
            }}
        }}

        function toggleAutoRotate() {{
            autoRotate = !autoRotate;
        }}

        function reconnectWebSocket() {{
            if (socket) {{
                socket.close();
            }}
            reconnectAttempts = 0;
            document.getElementById('status').innerHTML = 'Reconnecting...';
            document.getElementById('status').className = 'connecting';
            setTimeout(connectWebSocket, 500);
        }}
    </script>
</body>
</html>"""

        with open(web_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📁 Web files created in: {web_dir}")
        return web_dir

    def depth_to_pointcloud(self, frame: KinectFrame) -> Optional[Dict[str, Any]]:
        """Convert Kinect frame to point cloud data for web (improved version)"""
        if frame.color_image is None or frame.depth_image is None:
            return None

        try:
            # Get image dimensions
            depth_height, depth_width = frame.depth_image.shape

            # Resize color to match depth
            color_resized = cv2.resize(frame.color_image, (depth_width, depth_height))

            # Downsample for performance
            step = self.downsample_factor
            depth_down = frame.depth_image[::step, ::step]
            color_down = color_resized[::step, ::step]

            # Get valid depth pixels
            valid_mask = (depth_down > self.depth_min * 1000) & (
                depth_down < self.depth_max * 1000
            )

            if not np.any(valid_mask):
                print("⚠️ No valid depth pixels found")
                return None

            # Get pixel coordinates
            h, w = depth_down.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Scale coordinates back to original resolution
            u_scaled = u * step
            v_scaled = v * step

            # Convert to 3D points
            z = depth_down.astype(np.float32) / 1000.0  # Convert mm to meters
            x = (u_scaled - self.cx) * z / self.fx
            y = (v_scaled - self.cy) * z / self.fy

            # Apply valid mask
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]
            color_valid = color_down[valid_mask]

            # Limit points for web performance
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
                # Coordinate system: Kinect Y-down to Three.js Y-up
                points.extend([x_valid[i], -y_valid[i], -z_valid[i]])

                # Color (BGR to RGB, normalize)
                if len(color_valid[i]) >= 3:
                    b, g, r = color_valid[i][:3]
                    colors.extend([r / 255.0, g / 255.0, b / 255.0])
                else:
                    colors.extend([1.0, 0.0, 0.0])  # Red fallback

            point_count = len(points) // 3
            print(f"📊 Generated {point_count} points for web")

            return {
                "points": points,
                "colors": colors,
                "timestamp": time.time(),
                "point_count": point_count,
            }

        except Exception as e:
            print(f"❌ Error creating point cloud: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.connected_clients.add(websocket)
        client_addr = websocket.remote_address
        print(
            f"🔗 WebSocket client connected: {client_addr}. Total: {len(self.connected_clients)}"
        )

        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            print(
                f"🔌 WebSocket client disconnected: {client_addr}. Total: {len(self.connected_clients)}"
            )

    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """Broadcast frame data to all connected clients"""
        if not self.connected_clients:
            return

        try:
            message = json.dumps(frame_data)
            disconnected = []

            for client in self.connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(client)
                except Exception as e:
                    print(f"⚠️ Error sending to client: {e}")
                    disconnected.append(client)

            # Remove disconnected clients
            for client in disconnected:
                self.connected_clients.discard(client)

        except Exception as e:
            print(f"❌ Broadcast error: {e}")

    def capture_thread(self):
        """Kinect capture thread"""
        if not self.capture:
            print("❌ No Kinect capture available")
            return

        if not self.capture.start_capture():
            print("❌ Failed to start Kinect capture")
            return

        print("🎥 Kinect capture thread started")
        frame_count = 0

        try:
            while self.is_streaming:
                frame = self.capture.capture_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame

                    frame_count += 1
                    if frame_count % 90 == 0:  # Every 3 seconds at 30fps
                        print(f"📷 Captured {frame_count} frames")

                time.sleep(1 / 30)  # ~30 FPS max

        except Exception as e:
            print(f"❌ Capture thread error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.capture.stop_capture()
            print("🛑 Kinect capture thread stopped")

    async def streaming_loop(self):
        """Main streaming loop"""
        print("🌐 Starting streaming loop...")
        frame_count = 0

        while self.is_streaming:
            try:
                current_frame = None

                if self.current_frame is not None:
                    with self.frame_lock:
                        current_frame = self.current_frame

                if current_frame is not None:
                    # Convert to point cloud
                    point_cloud_data = self.depth_to_pointcloud(current_frame)

                    if point_cloud_data is not None:
                        await self.broadcast_frame(point_cloud_data)
                        frame_count += 1

                        if frame_count % 45 == 0:  # Every 3 seconds at 15fps
                            print(
                                f"📡 Streamed {frame_count} frames to {len(self.connected_clients)} clients"
                            )
                    else:
                        print("⚠️ No point cloud data generated")

            except Exception as e:
                print(f"❌ Streaming loop error: {e}")

            await asyncio.sleep(1 / 15)  # ~15 FPS for web streaming

    def start_web_server(self):
        """Start HTTP server for web files"""
        web_directory = str(self.web_dir)

        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=web_directory, **kwargs)

        try:
            httpd = HTTPServer(("localhost", self.http_port), CustomHandler)

            def serve_forever():
                print(f"🌐 HTTP server started: http://localhost:{self.http_port}")
                httpd.serve_forever()

            web_thread = threading.Thread(target=serve_forever, daemon=True)
            web_thread.start()

            return httpd

        except OSError as e:
            print(f"❌ Failed to start HTTP server on port {self.http_port}: {e}")
            print("💡 Try a different port or close other applications using this port")
            return None

    async def start_streaming(self):
        """Start the complete streaming system"""
        print("🚀 Starting Fixed Kinect Web Streaming System")
        print("=" * 60)

        # Check Kinect availability
        if not KINECT_AVAILABLE:
            print("⚠️ Kinect not available - running in demo mode")
            # Could add fake data generation here
            return

        # Start HTTP server
        web_server = self.start_web_server()
        if not web_server:
            return

        # Start Kinect capture thread
        self.is_streaming = True
        capture_thread = threading.Thread(target=self.capture_thread, daemon=True)
        capture_thread.start()

        # Start WebSocket server
        print(f"🔌 Starting WebSocket server on ws://localhost:{self.websocket_port}")
        print(f"🌐 Web interface: http://localhost:{self.http_port}")
        print("=" * 60)
        print("🎯 IMPORTANT:")
        print(f"   ✅ Open browser to: http://localhost:{self.http_port}")
        print(f"   ❌ DO NOT go to: http://localhost:{self.websocket_port}")
        print("=" * 60)

        # Open browser to correct URL
        webbrowser.open(f"http://localhost:{self.http_port}")

        # Start WebSocket server
        start_server = websockets.serve(
            self.handle_client, "localhost", self.websocket_port
        )

        try:
            await asyncio.gather(start_server, self.streaming_loop())
        except KeyboardInterrupt:
            print("\n🛑 Streaming stopped by user")
            self.is_streaming = False


async def main():
    """Main function with better error handling"""
    import argparse

    parser = argparse.ArgumentParser(description="Fixed Kinect web streaming")
    parser.add_argument("--device-id", type=int, default=0, help="Kinect device ID")
    parser.add_argument(
        "--websocket-port", type=int, default=8765, help="WebSocket port"
    )
    parser.add_argument("--http-port", type=int, default=8000, help="HTTP server port")
    parser.add_argument(
        "--max-points", type=int, default=3000, help="Max points for web"
    )

    args = parser.parse_args()

    # Check websockets
    try:
        import websockets
    except ImportError:
        print("❌ websockets package required: pip install websockets")
        return

    streamer = FixedKinectWebStreamer(
        device_id=args.device_id,
        websocket_port=args.websocket_port,
        http_port=args.http_port,
    )
    streamer.max_points = args.max_points

    try:
        await streamer.start_streaming()
    except KeyboardInterrupt:
        print("\n🛑 Streaming stopped by user")
        streamer.is_streaming = False
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
