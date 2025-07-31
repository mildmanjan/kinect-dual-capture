#!/usr/bin/env python3
"""
Clear Calibration Data
Save as: clear_calibration.py
"""

import shutil
import json
from pathlib import Path
import time


def backup_current_calibration():
    """Backup current calibration before clearing"""

    calib_file = Path("config/dual_kinect_calibration.json")

    if calib_file.exists():
        timestamp = int(time.time())
        backup_file = Path(f"config/calibration_backup_{timestamp}.json")

        try:
            shutil.copy2(calib_file, backup_file)
            print(f"📄 Backed up current calibration to: {backup_file.name}")
            return True
        except Exception as e:
            print(f"⚠️  Could not backup calibration: {e}")
            return False
    else:
        print("📄 No existing calibration to backup")
        return True


def clear_calibration_config():
    """Clear calibration configuration files"""

    print("🗑️  Clearing calibration configuration...")

    config_dir = Path("config")
    files_to_clear = [
        "dual_kinect_calibration.json",
        # Add other calibration config files if they exist
    ]

    cleared_count = 0

    for filename in files_to_clear:
        file_path = config_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ Removed: {filename}")
                cleared_count += 1
            except Exception as e:
                print(f"   ❌ Could not remove {filename}: {e}")
        else:
            print(f"   ⚪ Not found: {filename}")

    return cleared_count


def clear_calibration_data():
    """Clear calibration data files"""

    print("🗑️  Clearing calibration data...")

    data_dir = Path("data")
    dirs_to_clear = ["calibration", "step3_dual_test", "fusion_results", "overlap_test"]

    cleared_count = 0

    for dirname in dirs_to_clear:
        dir_path = data_dir / dirname
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Removed directory: {dirname}")
                cleared_count += 1
            except Exception as e:
                print(f"   ❌ Could not remove {dirname}: {e}")
        else:
            print(f"   ⚪ Not found: {dirname}")

    return cleared_count


def clear_old_fusion_results():
    """Clear old fusion results but keep directory structure"""

    print("🗑️  Clearing old fusion results...")

    fusion_dir = Path("data/fusion_results")

    if not fusion_dir.exists():
        print("   ⚪ No fusion_results directory")
        return 0

    cleared_count = 0

    # Clear individual PLY files
    for ply_file in fusion_dir.glob("*.ply"):
        try:
            ply_file.unlink()
            print(f"   ✅ Removed: {ply_file.name}")
            cleared_count += 1
        except Exception as e:
            print(f"   ❌ Could not remove {ply_file.name}: {e}")

    # Clear individual PCD files
    for pcd_file in fusion_dir.glob("*.pcd"):
        try:
            pcd_file.unlink()
            print(f"   ✅ Removed: {pcd_file.name}")
            cleared_count += 1
        except Exception as e:
            print(f"   ❌ Could not remove {pcd_file.name}: {e}")

    # Clear fusion sequence directories
    for seq_dir in fusion_dir.glob("fusion_sequence_*"):
        if seq_dir.is_dir():
            try:
                shutil.rmtree(seq_dir)
                print(f"   ✅ Removed directory: {seq_dir.name}")
                cleared_count += 1
            except Exception as e:
                print(f"   ❌ Could not remove {seq_dir.name}: {e}")

    # Clear analysis files
    analysis_files = ["alignment_analysis.json"]

    for filename in analysis_files:
        file_path = fusion_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ Removed: {filename}")
                cleared_count += 1
            except Exception as e:
                print(f"   ❌ Could not remove {filename}: {e}")

    return cleared_count


def clear_test_images():
    """Clear test and sample images"""

    print("🗑️  Clearing test images...")

    cleared_count = 0

    # Clear step1 samples
    step1_dir = Path("data/step1_samples")
    if step1_dir.exists():
        for img_file in step1_dir.glob("*"):
            try:
                img_file.unlink()
                cleared_count += 1
            except Exception as e:
                print(f"   ❌ Could not remove {img_file.name}: {e}")

        if cleared_count > 0:
            print(f"   ✅ Removed {cleared_count} step1 sample files")

    # Clear compression test results (optional - comment out to keep)
    # compression_dir = Path("data/compression_test")
    # if compression_dir.exists():
    #     try:
    #         shutil.rmtree(compression_dir)
    #         print(f"   ✅ Removed compression test data")
    #         cleared_count += 1
    #     except Exception as e:
    #         print(f"   ❌ Could not remove compression test data: {e}")

    return cleared_count


def recreate_directory_structure():
    """Recreate clean directory structure"""

    print("📁 Recreating clean directory structure...")

    directories_to_create = [
        "config",
        "data",
        "data/step1_samples",
        "data/step3_dual_test",
        "data/fusion_results",
        "data/calibration",
        "data/overlap_test",
    ]

    created_count = 0

    for dir_path_str in directories_to_create:
        dir_path = Path(dir_path_str)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {dir_path_str}")
                created_count += 1
            except Exception as e:
                print(f"   ❌ Could not create {dir_path_str}: {e}")
        else:
            print(f"   ⚪ Already exists: {dir_path_str}")

    return created_count


def create_fresh_calibration_config():
    """Create a fresh, empty calibration configuration"""

    print("📄 Creating fresh calibration configuration...")

    # Create default/empty calibration
    default_calibration = {
        "calibration_error": 999.0,
        "calibration_method": "none",
        "calibration_timestamp": time.time(),
        "device1_id": 0,
        "device2_id": 1,
        "num_calibration_frames": 0,
        "transformation_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "device1_intrinsics": {
            "fx": 504.0,
            "fy": 504.0,
            "cx": 320.0,
            "cy": 288.0,
            "width": 640,
            "height": 576,
        },
        "device2_intrinsics": {
            "fx": 504.0,
            "fy": 504.0,
            "cx": 320.0,
            "cy": 288.0,
            "width": 640,
            "height": 576,
        },
    }

    config_file = Path("config/dual_kinect_calibration.json")

    try:
        with open(config_file, "w") as f:
            json.dump(default_calibration, f, indent=2)

        print(f"   ✅ Created fresh calibration config: {config_file}")
        print(f"   📊 Status: Uncalibrated (999.0mm error)")
        return True

    except Exception as e:
        print(f"   ❌ Could not create calibration config: {e}")
        return False


def main():
    """Main calibration clearing function"""

    print("🧹 Calibration Data Cleaner")
    print("=" * 60)
    print("This will clear all calibration data for a fresh start")
    print("")

    # Confirm with user
    try:
        confirm = (
            input("⚠️  Are you sure you want to clear all calibration data? (y/N): ")
            .strip()
            .lower()
        )
        if confirm not in ["y", "yes"]:
            print("❌ Operation cancelled")
            return
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        return

    print("\n🚀 Starting calibration data cleanup...")

    # Step 1: Backup current calibration
    backup_success = backup_current_calibration()

    # Step 2: Clear calibration config
    config_cleared = clear_calibration_config()

    # Step 3: Clear calibration data
    data_cleared = clear_calibration_data()

    # Step 4: Clear old fusion results
    fusion_cleared = clear_old_fusion_results()

    # Step 5: Clear test images (optional)
    images_cleared = clear_test_images()

    # Step 6: Recreate directory structure
    dirs_created = recreate_directory_structure()

    # Step 7: Create fresh calibration config
    fresh_config = create_fresh_calibration_config()

    # Summary
    print(f"\n📊 Cleanup Summary:")
    print("=" * 60)
    print(f"✅ Calibration backup: {'Success' if backup_success else 'N/A'}")
    print(f"✅ Config files cleared: {config_cleared}")
    print(f"✅ Data directories cleared: {data_cleared}")
    print(f"✅ Fusion results cleared: {fusion_cleared}")
    print(f"✅ Test images cleared: {images_cleared}")
    print(f"✅ Directories recreated: {dirs_created}")
    print(f"✅ Fresh config created: {'Success' if fresh_config else 'Failed'}")

    print(f"\n🎯 Ready for Fresh Calibration!")
    print("=" * 60)
    print("Next steps:")
    print("1. Set up optimal scene (textured objects, good lighting)")
    print("2. Position Kinects for 60-80% overlap")
    print("3. Run: python dual_kinect_calibration.py")
    print(
        "4. Test: python src\\step5_kinect_fusion.py --mode sequence --duration 5 --export"
    )
    print("5. Check: python analyze_new_fusion.py")

    print(f"\n✅ Calibration cleanup complete!")


if __name__ == "__main__":
    main()
