"""
Standalone UI to run the playback error viewer (ground truth vs estimate)
without running the full pose pipeline. Use existing TUM trajectory files
and optional image folder.

Run with: python run_playback_ui.py
"""

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox


def make_section(parent, title):
    """Create a labeled frame section."""
    return ttk.LabelFrame(parent, text=title, padding=5)


def main():
    root = tk.Tk()
    root.title("Playback error viewer (GT vs estimate)")
    root.minsize(400, 280)

    base = Path(__file__).resolve().parent
    script_path = base / "scripts" / "playback_error_viewer.py"

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    sec = make_section(main_frame, "Files")
    sec.pack(fill=tk.X, pady=2)
    row = 0

    ttk.Label(sec, text="Estimate (TUM):").grid(row=row, column=0, sticky=tk.W, pady=2)
    estimate_var = tk.StringVar(value="e.txt")
    ttk.Entry(sec, textvariable=estimate_var, width=40).grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
    row += 1

    ttk.Label(sec, text="Ground truth (TUM):").grid(row=row, column=0, sticky=tk.W, pady=2)
    ground_truth_var = tk.StringVar(value="groundtruth.txt")
    ttk.Entry(sec, textvariable=ground_truth_var, width=40).grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
    row += 1

    ttk.Label(sec, text="Image folder (optional):").grid(row=row, column=0, sticky=tk.W, pady=2)
    images_var = tk.StringVar(value="rgb")
    ttk.Entry(sec, textvariable=images_var, width=40).grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
    ttk.Label(sec, text="e.g. rgb or path to image sequence; leave empty for no images.").grid(
        row=row + 1, column=1, sticky=tk.W, padx=5
    )
    row += 2

    sec2 = make_section(main_frame, "Options")
    sec2.pack(fill=tk.X, pady=2)
    r = 0
    ttk.Label(sec2, text="Image extension:").grid(row=r, column=0, sticky=tk.W, pady=2)
    ext_var = tk.StringVar(value=".png")
    ttk.Entry(sec2, textvariable=ext_var, width=10).grid(row=r, column=1, sticky=tk.W, padx=5, pady=2)
    ttk.Label(sec2, text="(default .png)").grid(row=r, column=2, sticky=tk.W)
    r += 1

    main_frame.columnconfigure(1, weight=1)

    def run_playback():
        est = estimate_var.get().strip()
        gt = ground_truth_var.get().strip()
        if not est or not gt:
            messagebox.showerror("Missing files", "Please set both Estimate and Ground truth file paths.")
            return
        if not script_path.is_file():
            messagebox.showerror("Script not found", f"Playback script not found: {script_path}")
            return
        cmd = [
            sys.executable,
            str(script_path),
            "-e", est,
            "-g", gt,
        ]
        images = images_var.get().strip()
        if images:
            cmd.extend(["-i", images])
        ext = ext_var.get().strip()
        if ext:
            cmd.extend(["--ext", ext])
        try:
            subprocess.Popen(cmd, cwd=str(base))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start playback viewer: {e}")

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(pady=10)
    ttk.Button(btn_frame, text="Run playback error viewer", command=run_playback).pack(side=tk.LEFT, padx=5)

    root.mainloop()


if __name__ == "__main__":
    main()
