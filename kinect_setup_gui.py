#!/usr/bin/env python3
"""
Kinect Setup GUI Manager
A comprehensive GUI for running all Kinect calibration and setup scripts

Location: kinect_setup_gui.py (in project root, same directory as setup_test.py)

This GUI provides:
- Visual script runner for all setup/calibration tools
- Real-time output monitoring
- Quick access to all diagnostic scripts
- Progress tracking and status indicators
- Results viewing and analysis
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import queue
import time
import json
from pathlib import Path
import sys
import os


class KinectSetupGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kinect Dual Capture - Setup & Calibration Manager")
        self.root.geometry("1200x800")

        # Process management
        self.current_process = None
        self.output_queue = queue.Queue()
        self.is_running = False

        # Script definitions with their locations and purposes
        self.scripts = {
            # Setup & Diagnostic (Project Root)
            "Setup & Diagnostic": {
                "setup_test.py": {
                    "name": "Verify Installation",
                    "description": "Check dependencies and project structure",
                    "location": "root",
                    "args": [],
                },
                "kinect_diagnostic_tool.py": {
                    "name": "Test Kinect Devices",
                    "description": "Basic hardware connectivity test",
                    "location": "root",
                    "args": [],
                },
                "debug_kinect_config.py": {
                    "name": "Test Configurations",
                    "description": "Find working Kinect configurations",
                    "location": "root",
                    "args": [],
                },
                "ultra_simple_kinect_test.py": {
                    "name": "Minimal Device Test",
                    "description": "Basic device creation test",
                    "location": "root",
                    "args": [],
                },
            },
            # Single Kinect Tests (src/)
            "Single Kinect Tests": {
                "src/step1_single_kinect_test.py": {
                    "name": "Device 0 Test",
                    "description": "Test single Kinect device 0",
                    "location": "src",
                    "args": ["--device-id", "0", "--duration", "10"],
                },
                "src/step1_single_kinect_test.py": {
                    "name": "Device 1 Test",
                    "description": "Test single Kinect device 1",
                    "location": "src",
                    "args": ["--device-id", "1", "--duration", "10"],
                },
                "src/step2_single_kinect_compression.py": {
                    "name": "Compression Test",
                    "description": "Test compression levels",
                    "location": "src",
                    "args": ["--compression", "medium", "--duration", "8"],
                },
            },
            # Calibration Scripts (Project Root)
            "Calibration": {
                "dual_kinect_calibration.py": {
                    "name": "Run Calibration",
                    "description": "Calibrate dual Kinect setup",
                    "location": "root",
                    "args": [],
                },
                "test_calibration_quality.py": {
                    "name": "Check Calibration Quality",
                    "description": "Analyze calibration accuracy",
                    "location": "root",
                    "args": [],
                },
                "capture_overlap_test.py": {
                    "name": "Test Scene Overlap",
                    "description": "Check scene overlap between devices",
                    "location": "root",
                    "args": [],
                },
                "clear_calibration.py": {
                    "name": "Clear Calibration",
                    "description": "Reset all calibration data",
                    "location": "root",
                    "args": [],
                },
            },
            # Dual Kinect Tests (src/)
            "Dual Kinect Tests": {
                "src/step3_dual_kinect_test.py": {
                    "name": "Dual Capture Test",
                    "description": "Test synchronized dual capture",
                    "location": "src",
                    "args": ["--duration", "15"],
                },
                "test_sync_quality.py": {
                    "name": "Synchronization Test",
                    "description": "Measure sync quality",
                    "location": "root",
                    "args": [],
                },
                "src/step4_dual_kinect_compression.py": {
                    "name": "Dual Compression",
                    "description": "Test dual device compression",
                    "location": "src",
                    "args": ["--compression", "medium", "--duration", "15"],
                },
            },
            # Fusion & Analysis (src/ and root)
            "Fusion & Analysis": {
                "src/step5_kinect_fusion.py": {
                    "name": "Real-time Fusion",
                    "description": "Live point cloud fusion",
                    "location": "src",
                    "args": ["--mode", "realtime", "--duration", "20"],
                },
                "src/step5_kinect_fusion.py": {
                    "name": "Fusion Sequence",
                    "description": "Capture fusion sequence",
                    "location": "src",
                    "args": ["--mode", "sequence", "--duration", "10", "--export"],
                },
                "analyze_new_fusion.py": {
                    "name": "Analyze Fusion Quality",
                    "description": "Check latest fusion results",
                    "location": "root",
                    "args": [],
                },
                "measure_alignment.py": {
                    "name": "Measure Alignment",
                    "description": "Detailed fusion analysis",
                    "location": "root",
                    "args": [],
                },
            },
            # Advanced Features (src/)
            "Advanced Features": {
                "src/step6_kinect_to_mesh_demo.py": {
                    "name": "Mesh Generation",
                    "description": "Generate 3D meshes",
                    "location": "src",
                    "args": ["--mode", "sequence", "--duration", "10", "--export"],
                },
                "src/step7_kinect_web_streaming.py": {
                    "name": "Web Streaming",
                    "description": "Stream to web browser",
                    "location": "src",
                    "args": [],
                },
                "view_compression_results.py": {
                    "name": "View Results",
                    "description": "View saved compression results",
                    "location": "root",
                    "args": [],
                },
            },
        }

        self.setup_gui()
        self.check_project_status()

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frame with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Script categories and buttons
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Right panel - Output and status
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)

    def setup_left_panel(self, parent):
        """Setup left panel with script categories"""
        # Title
        title_label = ttk.Label(
            parent, text="Kinect Setup Manager", font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Status frame
        status_frame = ttk.LabelFrame(parent, text="System Status", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_labels = {}
        status_items = [
            ("Dependencies", "UNKNOWN"),
            ("Device 0", "UNKNOWN"),
            ("Device 1", "UNKNOWN"),
            ("Calibration", "UNKNOWN"),
            ("Last Fusion", "UNKNOWN"),
        ]

        for item, status in status_items:
            frame = ttk.Frame(status_frame)
            frame.pack(fill=tk.X, pady=1)
            ttk.Label(frame, text=f"{item}:", width=12).pack(side=tk.LEFT)
            self.status_labels[item] = ttk.Label(
                frame, text=status, font=("Arial", 10, "bold")
            )
            self.status_labels[item].pack(side=tk.LEFT)

        # Create notebook for script categories
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Create tabs for each category
        for category, scripts in self.scripts.items():
            self.create_category_tab(category, scripts)

    def create_category_tab(self, category, scripts):
        """Create a tab for a script category"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=category)

        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add scripts to the tab
        for script_path, script_info in scripts.items():
            self.create_script_button(scrollable_frame, script_path, script_info)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_script_button(self, parent, script_path, script_info):
        """Create a button for running a script"""
        frame = ttk.LabelFrame(parent, text=script_info["name"], padding=5)
        frame.pack(fill=tk.X, pady=2)

        # Description
        desc_label = ttk.Label(
            frame, text=script_info["description"], font=("Arial", 9), foreground="gray"
        )
        desc_label.pack(anchor="w")

        # Button frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        # Run button
        run_btn = ttk.Button(
            btn_frame,
            text="Run",
            command=lambda: self.run_script(script_path, script_info),
        )
        run_btn.pack(side=tk.LEFT)

        # Args entry (if script accepts arguments)
        if script_info.get("args"):
            args_var = tk.StringVar(value=" ".join(script_info["args"]))
            args_entry = ttk.Entry(btn_frame, textvariable=args_var, width=30)
            args_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
            script_info["args_var"] = args_var

    def setup_right_panel(self, parent):
        """Setup right panel with output and controls"""
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        # Stop button
        self.stop_btn = ttk.Button(
            control_frame,
            text="Stop Current",
            command=self.stop_script,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT)

        # Clear output button
        clear_btn = ttk.Button(
            control_frame, text="Clear Output", command=self.clear_output
        )
        clear_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))

        # Add a real-time status line
        status_info_frame = ttk.Frame(control_frame)
        status_info_frame.pack(side=tk.LEFT, padx=(20, 0))

        ttk.Label(status_info_frame, text="Status:").pack(side=tk.LEFT)
        self.status_info_label = ttk.Label(
            status_info_frame, text="Ready", foreground="green"
        )
        self.status_info_label.pack(side=tk.LEFT, padx=(5, 0))

        # Output text area
        output_frame = ttk.LabelFrame(parent, text="Script Output", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(
            output_frame, font=("Consolas", 9), bg="#1e1e1e", fg="#ffffff"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Quick Results", padding=5)
        results_frame.pack(fill=tk.X, pady=(5, 0))

        # Results buttons
        results_btn_frame = ttk.Frame(results_frame)
        results_btn_frame.pack(fill=tk.X)

        ttk.Button(
            results_btn_frame,
            text="View Compression Results",
            command=self.view_compression_results,
        ).pack(side=tk.LEFT)
        ttk.Button(
            results_btn_frame, text="Open Data Folder", command=self.open_data_folder
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(
            results_btn_frame,
            text="Check Calibration",
            command=self.quick_calibration_check,
        ).pack(side=tk.LEFT, padx=(10, 0))

    def run_script(self, script_path, script_info):
        """Run a script in a separate thread"""
        if self.is_running:
            messagebox.showwarning(
                "Already Running",
                "Another script is currently running. Please wait or stop it first.",
            )
            return

        # Get arguments if they exist
        args = []
        if script_info.get("args_var"):
            args_text = script_info["args_var"].get().strip()
            if args_text:
                args = args_text.split()

        # Start the script
        self.start_script(script_path, args)

    def start_script(self, script_path, args=[]):
        """Start a script subprocess"""
        self.is_running = True
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()
        self.status_info_label.config(
            text=f"Running: {Path(script_path).name}", foreground="orange"
        )

        # Clear output
        self.output_text.delete(1.0, tk.END)

        # Log start (Windows-safe)
        self.append_output(f"Starting: {script_path}\n")
        if args:
            self.append_output(f"Arguments: {' '.join(args)}\n")
        self.append_output("=" * 60 + "\n")

        # Start script in thread
        script_thread = threading.Thread(
            target=self.run_script_thread, args=(script_path, args)
        )
        script_thread.daemon = True
        script_thread.start()

        # Start output monitoring
        self.monitor_output()

    def run_script_thread(self, script_path, args):
        """Run script in separate thread"""
        try:
            # Build command with proper encoding for Windows
            cmd = [sys.executable, "-u", script_path] + args  # -u for unbuffered output

            # Set environment to handle Unicode properly and force unbuffered output
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output

            # Run process with real-time output
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=0,  # Unbuffered
                env=env,
                encoding="utf-8",
                errors="replace",  # Replace problematic characters
            )

            # Read output line by line in real-time
            while True:
                line = self.current_process.stdout.readline()
                if line:
                    self.output_queue.put(line)
                elif self.current_process.poll() is not None:
                    # Process has finished
                    break

            # Get any remaining output
            remaining_output = self.current_process.stdout.read()
            if remaining_output:
                self.output_queue.put(remaining_output)

            # Wait for completion
            self.current_process.wait()
            return_code = self.current_process.returncode

            # Signal completion (Windows-safe)
            self.output_queue.put(f"\n{'='*60}\n")
            if return_code == 0:
                self.output_queue.put("SUCCESS: Script completed successfully!\n")
            else:
                self.output_queue.put(
                    f"ERROR: Script failed with exit code: {return_code}\n"
                )

        except Exception as e:
            self.output_queue.put(f"ERROR: Error running script: {e}\n")
        finally:
            self.output_queue.put("SCRIPT_FINISHED")

    def monitor_output(self):
        """Monitor output queue and update GUI - faster polling for real-time display"""
        lines_processed = 0
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "SCRIPT_FINISHED":
                    self.script_finished()
                    break
                else:
                    self.append_output(line)
                    lines_processed += 1

                    # Update GUI more frequently when processing multiple lines
                    if lines_processed >= 5:
                        self.root.update_idletasks()
                        lines_processed = 0

        except queue.Empty:
            pass

        if self.is_running:
            # Poll more frequently for real-time updates (50ms instead of 100ms)
            self.root.after(50, self.monitor_output)

    def append_output(self, text):
        """Append text to output area with Unicode safety and auto-scroll"""
        try:
            # Ensure text is properly encoded for display
            safe_text = text.encode("utf-8", "replace").decode("utf-8")

            # Insert text
            self.output_text.insert(tk.END, safe_text)

            # Auto-scroll to bottom
            self.output_text.see(tk.END)

            # Force immediate GUI update for real-time display
            self.root.update_idletasks()

        except Exception as e:
            # Fallback for any encoding issues
            fallback_text = str(text).encode("ascii", "replace").decode("ascii")
            self.output_text.insert(tk.END, fallback_text)
            self.output_text.see(tk.END)

    def script_finished(self):
        """Handle script completion"""
        self.is_running = False
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        self.current_process = None
        self.status_info_label.config(text="Ready", foreground="green")

        # Update status
        self.check_project_status()

    def stop_script(self):
        """Stop currently running script"""
        if self.current_process:
            self.current_process.terminate()
            self.append_output("\nSTOPPED: Script stopped by user\n")
            self.script_finished()
            self.status_info_label.config(text="Stopped", foreground="red")

    def clear_output(self):
        """Clear output text"""
        self.output_text.delete(1.0, tk.END)

    def check_project_status(self):
        """Check and update project status indicators"""
        # Check dependencies (setup_test.py exists)
        if Path("setup_test.py").exists():
            self.status_labels["Dependencies"]["text"] = "OK"
            self.status_labels["Dependencies"]["foreground"] = "green"
        else:
            self.status_labels["Dependencies"]["text"] = "MISSING"
            self.status_labels["Dependencies"]["foreground"] = "red"

        # Check for calibration file
        calib_file = Path("config/dual_kinect_calibration.json")
        if calib_file.exists():
            try:
                with open(calib_file) as f:
                    data = json.load(f)
                error = data.get("calibration_error", 999)
                if error < 50:
                    self.status_labels["Calibration"]["text"] = f"OK {error:.1f}mm"
                    self.status_labels["Calibration"]["foreground"] = "green"
                else:
                    self.status_labels["Calibration"]["text"] = f"WARN {error:.1f}mm"
                    self.status_labels["Calibration"]["foreground"] = "orange"
            except:
                self.status_labels["Calibration"]["text"] = "ERROR"
                self.status_labels["Calibration"]["foreground"] = "red"
        else:
            self.status_labels["Calibration"]["text"] = "NONE"
            self.status_labels["Calibration"]["foreground"] = "gray"

        # Check for recent fusion results
        fusion_dir = Path("data/fusion_results")
        if fusion_dir.exists():
            ply_files = list(fusion_dir.glob("*.ply"))
            if ply_files:
                latest = max(ply_files, key=lambda x: x.stat().st_mtime)
                age_hours = (time.time() - latest.stat().st_mtime) / 3600
                if age_hours < 1:
                    self.status_labels["Last Fusion"][
                        "text"
                    ] = f"OK {age_hours:.0f}m ago"
                    self.status_labels["Last Fusion"]["foreground"] = "green"
                else:
                    self.status_labels["Last Fusion"][
                        "text"
                    ] = f"OLD {age_hours:.0f}h ago"
                    self.status_labels["Last Fusion"]["foreground"] = "orange"
            else:
                self.status_labels["Last Fusion"]["text"] = "NONE"
                self.status_labels["Last Fusion"]["foreground"] = "gray"

        # Device status would require actual device check - keeping as unknown for now
        for device in ["Device 0", "Device 1"]:
            self.status_labels[device]["text"] = "UNKNOWN"
            self.status_labels[device]["foreground"] = "gray"

    def view_compression_results(self):
        """View compression test results"""
        if Path("view_compression_results.py").exists():
            self.start_script("view_compression_results.py")
        else:
            messagebox.showinfo(
                "Not Found", "view_compression_results.py not found in project root"
            )

    def open_data_folder(self):
        """Open data folder in file explorer"""
        data_path = Path("data")
        if data_path.exists():
            if sys.platform == "win32":
                os.startfile(data_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", data_path])
            else:
                subprocess.run(["xdg-open", data_path])
        else:
            messagebox.showinfo("Not Found", "Data folder not found")

    def quick_calibration_check(self):
        """Quick calibration status check"""
        if Path("test_calibration_quality.py").exists():
            self.start_script("test_calibration_quality.py")
        else:
            messagebox.showinfo(
                "Not Found", "test_calibration_quality.py not found in project root"
            )


def main():
    """Main function"""
    root = tk.Tk()
    app = KinectSetupGUI(root)

    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1200 // 2)
    y = (root.winfo_screenheight() // 2) - (800 // 2)
    root.geometry(f"1200x800+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
