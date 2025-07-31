# Kinect Dual Capture Project

A Python project for capturing, compressing, processing, and fusing data from single and dual Azure Kinect devices. This project provides step-by-step scripts to test Kinect functionality, implement variable compression, achieve dual-device fusion, and generate animated 3D meshes.

## 🎯 Project Overview

This project addresses the high bandwidth requirements of Azure Kinect data capture by implementing:
- **Single Kinect testing and capture**
- **Variable compression algorithms** to reduce data bandwidth
- **Dual Kinect synchronization** with sync cable support
- **Real-time point cloud fusion** with automatic calibration
- **3D mesh generation** and animation export
- **Web-based 3D streaming** for browser visualization

## 📋 Prerequisites

### Hardware Requirements
- 1-2 Azure Kinect DK devices
- USB 3.0 ports (separate controllers recommended for dual setup)
- Sync cable (recommended for dual Kinect setup)
- Windows 10/11 (recommended) or Linux

### Software Requirements
- **Python 3.8+** (tested with Python 3.11)
- **Azure Kinect SDK v1.4.1** 
- **Visual Studio Code** (recommended)
- **Git** (for version control)

## 🚀 Quick Start

### 1. Install Azure Kinect SDK

1. Download [Azure Kinect SDK 1.4.1](https://docs.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download)
2. Install to default location: `C:\Program Files\Azure Kinect SDK v1.4.1`
3. Connect your Azure Kinect to USB 3.0 port

### 2. Setup Project

```powershell
# Clone or create project directory
mkdir kinect-dual-capture
cd kinect-dual-capture

# Create virtual environment
py -3 -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Setup

```powershell
# Test that everything is installed correctly
python setup_test.py

# Should show all ✅ green checkmarks
```

## 📁 Project Structure

```
kinect-dual-capture/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_test.py                     # Verify installation
├── debug_kinect_config.py             # Test Kinect configurations
├── view_compression_results.py        # View saved compression results
├── dual_kinect_calibration.py         # Calibration tool for dual setup
├── .pylintrc                         # Code linting configuration
├── .vscode/                          # VS Code settings
│   ├── settings.json
│   ├── launch.json
│   └── tasks.json
├── src/                              # Source code
│   ├── step1_single_kinect_test.py    # Step 1: Basic Kinect test
│   ├── step2_single_kinect_compression.py # Step 2: Compression test
│   ├── step3_dual_kinect_test.py      # Step 3: Dual Kinect testing
│   ├── step4_dual_kinect_compression.py # Step 4: Dual compression
│   ├── step5_kinect_fusion.py         # Step 5: Point cloud fusion
│   ├── step6_kinect_to_mesh_demo.py   # Step 6: Mesh generation
│   ├── step7_kinect_web_streaming.py  # Step 7: Web streaming
│   └── utils/                        # Shared utilities
│       ├── kinect_capture.py          # Kinect capture classes
│       ├── compression_utils.py       # Compression algorithms
│       └── visualization.py          # Display utilities
├── data/                             # Output data
│   ├── step1_samples/                # Step 1 test results
│   ├── compression_test/             # Step 2 compression results
│   ├── step3_dual_test/              # Step 3 dual testing
│   ├── step4_dual_compression/       # Step 4 compression results
│   ├── fusion_results/               # Step 5 fusion point clouds
│   ├── mesh_animation/               # Step 6 mesh sequences
│   └── web_viewer/                   # Step 7 web interface
├── config/                           # Configuration files
│   └── dual_kinect_calibration.json  # Dual Kinect calibration data
└── venv/                            # Virtual environment (hidden)
```

## 🎮 Usage Guide - All Steps

### Step 1: Single Kinect Test ✅ COMPLETE

Test that your Azure Kinect is working and streaming properly.

```powershell
# Quick connection test
python src\step1_single_kinect_test.py --test-connection

# Full capture test (10 seconds)
python src\step1_single_kinect_test.py --duration 10
```

### Step 2: Compression Testing ✅ COMPLETE

Test different compression levels to reduce bandwidth requirements.

```powershell
# Test all compression levels
python src\step2_single_kinect_compression.py --compression all --duration 8

# Test specific level
python src\step2_single_kinect_compression.py --compression medium --duration 10
```

**Expected Results:**
- **MEDIUM**: ~35 Mbps bandwidth (15x reduction)
- **HIGH**: ~20 Mbps bandwidth (25x reduction)

### Step 3: Dual Kinect Testing ✅ COMPLETE

Test synchronized capture from two Kinect devices.

```powershell
# Test without sync cable
python src\step3_dual_kinect_test.py --duration 15

# Test with sync cable (recommended)
python src\step3_dual_kinect_test.py --sync-cable --duration 15
```

### Step 4: Dual Kinect Compression ✅ COMPLETE

Apply compression to dual Kinect streams.

```powershell
# Test dual compression
python src\step4_dual_kinect_compression.py --compression medium --duration 20
```

### Step 5: Point Cloud Fusion ✅ COMPLETE

**Real-time 3D fusion with automatic calibration!**

```powershell
# Calibrate dual Kinect setup (run once)
python dual_kinect_calibration.py

# Real-time fusion visualization
python src\step5_kinect_fusion.py --sync-cable --mode realtime --duration 30

# Capture animation sequence
python src\step5_kinect_fusion.py --sync-cable --mode sequence --duration 10 --fps 5 --export
```

**Features:**
- ✅ Master/subordinate synchronization with sync cable
- ✅ Automatic point cloud fusion (60,000+ points)
- ✅ ICP registration for improved alignment
- ✅ Real-time 3D visualization
- ✅ Auto-save every 10 seconds (.ply format)
- ✅ 100% fusion success rate achieved!

### Step 6: Mesh Generation ✅ COMPLETE

Convert point clouds to animated 3D meshes.

```powershell
# Real-time mesh visualization
python src\step6_kinect_to_mesh_demo.py --mode realtime --duration 20

# Capture mesh animation sequence
python src\step6_kinect_to_mesh_demo.py --mode sequence --duration 10 --fps 5 --export
```

### Step 7: Web Streaming ✅ COMPLETE

Stream 3D data to web browsers in real-time.

```powershell
# Start web streaming system
python src\step7_kinect_web_streaming.py --device-id 0

# Open browser to: http://localhost:8000
# WebSocket connection: ws://localhost:8765
```

## 🎛️ Advanced Options

### Calibration Management
```powershell
# Calibrate dual Kinect setup
python dual_kinect_calibration.py

# View current calibration status
python src\step5_kinect_fusion.py --mode realtime --duration 5
```

### Performance Tuning
```powershell
# High-speed fusion (lower quality)
python src\step5_kinect_fusion.py --sync-cable --voxel-size 0.01 --no-registration

# High-quality fusion (slower)
python src\step5_kinect_fusion.py --sync-cable --voxel-size 0.002

# Fusion without sync cable
python src\step5_kinect_fusion.py --mode realtime --duration 30
```

### Export Options
```powershell
# Export various formats
python src\step5_kinect_fusion.py --sync-cable --mode sequence --export --duration 10

# View saved point clouds
python view_compression_results.py
```

## 📊 Current Project Status

### Completed Features ✅
- ✅ **Step 1**: Single Kinect capture and testing
- ✅ **Step 2**: Variable compression (8x to 40x reduction)
- ✅ **Step 3**: Dual Kinect synchronized capture
- ✅ **Step 4**: Dual Kinect compression
- ✅ **Step 5**: Real-time point cloud fusion with calibration
- ✅ **Step 6**: 3D mesh generation and animation
- ✅ **Step 7**: Web-based 3D streaming

### Performance Achievements 🏆
- **Dual Fusion**: 100% success rate, 69,952 average points
- **Sync Quality**: <2ms with sync cable, <100ms standalone
- **Compression**: Up to 40x bandwidth reduction
- **Real-time**: 3-5 FPS fusion processing
- **Export**: PLY, OBJ, Blender-ready animations

## 🔧 Troubleshooting

### Common Issues

**❌ "No devices found"**
- Check USB 3.0 connection
- Ensure Kinect is powered (green LED)
- Try different USB port

**❌ Sync cable errors**
```powershell
# Test sync cable setup
python src\step3_dual_kinect_test.py --sync-cable --individual-only

# Use without sync cable if needed
python src\step5_kinect_fusion.py --mode realtime --duration 30
```

**❌ Low fusion quality**
```powershell
# Recalibrate system
python dual_kinect_calibration.py

# Check current calibration error (should be <30mm)
python src\step5_kinect_fusion.py --mode realtime --duration 5
```

### Debug Tools

```powershell
# Test different Kinect configurations
python debug_kinect_config.py

# Verify all dependencies
python setup_test.py

# Check individual devices
python src\step3_dual_kinect_test.py --individual-only
```

## 🎯 Viewing Your Results

### Point Cloud Files (.ply)
**Recommended viewers:**
1. **CloudCompare** (free, professional): https://cloudcompare-org.danielgm.net/release/
2. **Point.love** (web-based): https://point.love/
3. **Open3D Python viewer** (included in project)

```powershell
# View your fusion results
python view_ply_files.py --list
```

### Mesh Animations
- **Blender**: Use generated import scripts
- **Maya/3ds Max**: Import OBJ sequences
- **Unity/Unreal**: Import as mesh sequences

## 🌐 Integration Options

### 3D Software Integration
- **Blender**: Automatic import scripts generated
- **CloudCompare**: Direct .ply import
- **Mesh processing**: Open3D, MeshLab compatibility

### Web Applications
- **Three.js**: Real-time WebSocket streaming
- **WebXR**: VR/AR compatible output
- **REST API**: Export integration possible

## 📈 Performance Expectations

### Typical Results (Dual Kinect, 1080p, 30fps)

| Component | Performance | Quality | Notes |
|-----------|-------------|---------|-------|
| Single Capture | 30 FPS | Perfect | Step 1 baseline |
| Compression | 15x reduction | Good | Step 2 optimization |
| Dual Sync | <2ms offset | Excellent | Step 3 with sync cable |
| Point Fusion | 3-5 FPS | Excellent | Step 5 real-time |
| Mesh Generation | 5 FPS | High | Step 6 animation |
| Web Streaming | 15 FPS | Good | Step 7 browser |

### Hardware Performance
- **USB Bandwidth**: Requires USB 3.0 (separate controllers for dual)
- **CPU Usage**: Moderate to high during fusion
- **Memory**: ~2-4GB for fusion processing
- **Storage**: ~100MB per minute of fusion data

## 🚀 Advanced Development

### Custom Applications
```python
# Use the fusion system in your own code
from src.step5_kinect_fusion import DualKinectFusion

fusion = DualKinectFusion(use_sync_cable=True)
# Your custom processing here
```

### Network Streaming
```python
# Stream fusion data over network
from src.step7_kinect_web_streaming import KinectWebStreamer

streamer = KinectWebStreamer(device_id=0)
# Custom network protocols
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create feature branch**: `git checkout -b feature/advanced-fusion`
3. **Test all steps**: Ensure Steps 1-7 work properly
4. **Submit pull request** with description of changes

## 📄 License

This project is for educational and research purposes. Azure Kinect SDK has its own licensing terms.

## 🆘 Support

If you encounter issues:

1. **Check troubleshooting section** above
2. **Run diagnostic tools**: `python setup_test.py` and `python debug_kinect_config.py`
3. **Test individual steps**: Start with Step 1, progress through Step 5
4. **Verify hardware**: Ensure both Kinects work with Microsoft's Kinect Viewer

---

## Quick Reference Card

```powershell
# Essential Commands - All Steps Working!
python setup_test.py                          # Verify setup
python src\step1_single_kinect_test.py --duration 10        # Test single Kinect
python src\step2_single_kinect_compression.py --compression medium --duration 10  # Compression
python src\step3_dual_kinect_test.py --sync-cable --duration 15   # Dual testing  
python dual_kinect_calibration.py                           # Calibrate dual setup
python src\step5_kinect_fusion.py --sync-cable --mode realtime --duration 30  # 🎯 FUSION!
python src\step6_kinect_to_mesh_demo.py --mode sequence --export   # Mesh animation
python src\step7_kinect_web_streaming.py                    # Web streaming

# View Results
python view_ply_files.py --list               # View point clouds
```

**Status**: All 7 steps completed and tested! 