#!/usr/bin/env python3
"""
Mesh Recordings Cleanup Utility
Clean up old mesh animation files and recordings to free up disk space.

This script helps manage the data/mesh_animation directory by:
- Listing all mesh recordings with file counts and sizes
- Selectively deleting old recordings
- Bulk cleanup options
- Safe deletion with confirmation prompts
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import argparse


class MeshCleanupManager:
    """Manager for cleaning up mesh animation files"""

    def __init__(self):
        self.mesh_dir = Path("data/mesh_animation")
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

    def get_recording_groups(self) -> Dict[str, Dict]:
        """Group files by recording timestamp"""
        recordings = {}

        if not self.mesh_dir.exists():
            return recordings

        for file_path in self.mesh_dir.iterdir():
            if file_path.is_file():
                name = file_path.name

                # Extract timestamp from different file types
                timestamp = None
                base_name = None

                if name.startswith("kinect_mesh_") and ("_" in name):
                    # Pattern: kinect_mesh_TIMESTAMP_0000.obj
                    parts = name.split("_")
                    if len(parts) >= 3 and parts[2].isdigit():
                        timestamp = parts[2]
                        base_name = f"kinect_mesh_{timestamp}"

                elif name.startswith("import_to_blender_") and name.endswith(".py"):
                    # Pattern: import_to_blender_TIMESTAMP.py
                    timestamp = name.replace("import_to_blender_", "").replace(
                        ".py", ""
                    )
                    if timestamp.isdigit():
                        base_name = f"import_script_{timestamp}"

                if timestamp and base_name:
                    if timestamp not in recordings:
                        recordings[timestamp] = {
                            "timestamp": timestamp,
                            "datetime": self._timestamp_to_datetime(timestamp),
                            "files": [],
                            "total_size": 0,
                            "mesh_count": 0,
                            "has_metadata": False,
                            "has_blender_script": False,
                        }

                    file_size = file_path.stat().st_size
                    recordings[timestamp]["files"].append(file_path)
                    recordings[timestamp]["total_size"] += file_size

                    # Count specific file types
                    if name.endswith((".obj", ".ply")):
                        recordings[timestamp]["mesh_count"] += 1
                    elif name.endswith(".json"):
                        recordings[timestamp]["has_metadata"] = True
                    elif name.endswith(".py"):
                        recordings[timestamp]["has_blender_script"] = True

        return recordings

    def _timestamp_to_datetime(self, timestamp: str) -> str:
        """Convert timestamp to readable datetime string"""
        try:
            dt = datetime.fromtimestamp(int(timestamp))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return "Unknown date"

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def list_recordings(self) -> None:
        """List all mesh recordings with details"""
        recordings = self.get_recording_groups()

        if not recordings:
            print("📂 No mesh recordings found in data/mesh_animation/")
            return

        print("🎬 Mesh Animation Recordings")
        print("=" * 80)
        print(
            f"{'#':<3} {'Date/Time':<20} {'Meshes':<8} {'Size':<10} {'Files':<7} {'Extras':<15}"
        )
        print("-" * 80)

        sorted_recordings = sorted(recordings.items(), key=lambda x: x[0], reverse=True)

        for i, (timestamp, info) in enumerate(sorted_recordings, 1):
            extras = []
            if info["has_metadata"]:
                extras.append("JSON")
            if info["has_blender_script"]:
                extras.append("Blender")

            extras_str = ", ".join(extras) if extras else "None"

            print(
                f"{i:<3} {info['datetime']:<20} {info['mesh_count']:<8} "
                f"{self._format_size(info['total_size']):<10} {len(info['files']):<7} {extras_str:<15}"
            )

        total_size = sum(info["total_size"] for info in recordings.values())
        total_files = sum(len(info["files"]) for info in recordings.values())

        print("-" * 80)
        print(
            f"Total: {len(recordings)} recordings, {total_files} files, {self._format_size(total_size)}"
        )

    def delete_recording(self, timestamp: str) -> bool:
        """Delete a specific recording by timestamp"""
        recordings = self.get_recording_groups()

        if timestamp not in recordings:
            print(f"❌ Recording with timestamp {timestamp} not found")
            return False

        recording = recordings[timestamp]

        print(f"\n🗑️  Deleting recording from {recording['datetime']}")
        print(f"   Files: {len(recording['files'])}")
        print(f"   Size: {self._format_size(recording['total_size'])}")
        print(f"   Meshes: {recording['mesh_count']}")

        # Confirm deletion
        confirm = input("   Are you sure? (y/N): ").lower().strip()
        if confirm != "y":
            print("   ❌ Deletion cancelled")
            return False

        # Delete files
        deleted_count = 0
        for file_path in recording["files"]:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to delete {file_path.name}: {e}")

        print(f"   ✅ Deleted {deleted_count}/{len(recording['files'])} files")
        return deleted_count == len(recording["files"])

    def delete_oldest(self, keep_count: int = 3) -> None:
        """Delete oldest recordings, keeping the most recent N recordings"""
        recordings = self.get_recording_groups()

        if len(recordings) <= keep_count:
            print(
                f"📂 Only {len(recordings)} recordings found, keeping all (requested to keep {keep_count})"
            )
            return

        # Sort by timestamp (oldest first)
        sorted_recordings = sorted(recordings.items(), key=lambda x: x[0])
        to_delete = sorted_recordings[:-keep_count]  # All except the last N

        if not to_delete:
            print(f"📂 No recordings to delete (keeping newest {keep_count})")
            return

        print(
            f"🗑️  Planning to delete {len(to_delete)} oldest recordings (keeping newest {keep_count}):"
        )

        total_size = 0
        for timestamp, info in to_delete:
            print(
                f"   - {info['datetime']} ({info['mesh_count']} meshes, {self._format_size(info['total_size'])})"
            )
            total_size += info["total_size"]

        print(f"\nTotal space to free: {self._format_size(total_size)}")

        confirm = input("Proceed with deletion? (y/N): ").lower().strip()
        if confirm != "y":
            print("❌ Deletion cancelled")
            return

        # Delete recordings
        deleted_recordings = 0
        for timestamp, info in to_delete:
            if self.delete_recording_silent(timestamp):
                deleted_recordings += 1

        print(f"✅ Deleted {deleted_recordings}/{len(to_delete)} recordings")

    def delete_recording_silent(self, timestamp: str) -> bool:
        """Delete recording without confirmation prompts"""
        recordings = self.get_recording_groups()

        if timestamp not in recordings:
            return False

        recording = recordings[timestamp]
        deleted_count = 0

        for file_path in recording["files"]:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception:
                pass

        return deleted_count == len(recording["files"])

    def clean_empty_directories(self) -> None:
        """Remove empty directories in the mesh animation folder"""
        if not self.mesh_dir.exists():
            return

        removed_count = 0
        for item in self.mesh_dir.iterdir():
            if item.is_dir() and not any(item.iterdir()):
                try:
                    item.rmdir()
                    removed_count += 1
                    print(f"🗂️  Removed empty directory: {item.name}")
                except Exception as e:
                    print(f"⚠️  Could not remove directory {item.name}: {e}")

        if removed_count == 0:
            print("📁 No empty directories found")

    def get_total_size(self) -> int:
        """Get total size of all mesh recordings"""
        total_size = 0
        if self.mesh_dir.exists():
            for file_path in self.mesh_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description="Clean up mesh animation recordings")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all recordings")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete specific recording")
    delete_parser.add_argument("timestamp", help="Timestamp of recording to delete")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Delete oldest recordings")
    clean_parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of newest recordings to keep (default: 3)",
    )

    # Clear all command
    clear_parser = subparsers.add_parser(
        "clear", help="Delete ALL recordings (dangerous!)"
    )

    args = parser.parse_args()

    cleanup_manager = MeshCleanupManager()

    if args.command == "list" or args.command is None:
        cleanup_manager.list_recordings()

    elif args.command == "delete":
        cleanup_manager.delete_recording(args.timestamp)

    elif args.command == "clean":
        cleanup_manager.delete_oldest(keep_count=args.keep)
        cleanup_manager.clean_empty_directories()

    elif args.command == "clear":
        print("⚠️  WARNING: This will delete ALL mesh recordings!")
        cleanup_manager.list_recordings()
        print("\n" + "=" * 50)
        confirm = input("Type 'DELETE ALL' to confirm: ").strip()
        if confirm == "DELETE ALL":
            recordings = cleanup_manager.get_recording_groups()
            deleted = 0
            for timestamp in recordings:
                if cleanup_manager.delete_recording_silent(timestamp):
                    deleted += 1
            print(f"🗑️  Deleted {deleted} recordings")
            cleanup_manager.clean_empty_directories()
        else:
            print("❌ Deletion cancelled")

    # Show final status
    total_size = cleanup_manager.get_total_size()
    if total_size > 0:
        print(f"\n📊 Current total size: {cleanup_manager._format_size(total_size)}")
    else:
        print(f"\n📊 Mesh animation directory is now empty")


if __name__ == "__main__":
    main()
