#!/usr/bin/env python3
"""
Post-hoc playback viewer: frame images + error-vs-time graph (evo-style).

Loads estimate and ground-truth TUM trajectories, associates by timestamp,
aligns with Umeyama (scale + R + t), computes per-pose error, and shows
two panels: current frame image (left) and error vs relative time (right)
with RMSE/median horizontal lines. Slider and keyboard scrub; hover on
graph shows timestamp and frame number; click on graph jumps to that frame.

Usage:
  python scripts/playback_error_viewer.py --estimate estimate.txt --ground_truth groundtruth.txt [--images rgb] [--ext .png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.spatial.transform import Rotation
import cv2

# Optional evo for Umeyama; fallback to local implementation
try:
    from evo.core import geometry as evo_geometry
    _HAS_EVO = True
except ImportError:
    _HAS_EVO = False


def load_tum_with_timestamps(path):
    """Load TUM file: lines timestamp tx ty tz qx qy qz qw. Returns list of (timestamp, 4x4 world_T_cam)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            except (ValueError, IndexError):
                continue
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            out.append((ts, T))
    return out


def umeyama_alignment(src_3xn, tgt_3xn, with_scale=True):
    """
    Umeyama alignment: find s, R, t such that tgt ~ s*R@src + t.
    src_3xn, tgt_3xn: 3 x N arrays of positions.
    Returns R (3x3), t (3,), s (float). Aligned positions: s*R@src + t.
    """
    if _HAS_EVO:
        R, t, s = evo_geometry.umeyama_alignment(src_3xn, tgt_3xn, with_scale=with_scale)
        return R, t, s
    # Minimal implementation (Umeyama 1991)
    x = np.asarray(src_3xn, dtype=np.float64)
    y = np.asarray(tgt_3xn, dtype=np.float64)
    if x.shape[0] != 3 or y.shape[0] != 3 or x.shape[1] != y.shape[1]:
        raise ValueError("umeyama: need 3xN and 3xN same N")
    n = x.shape[1]
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    sigma_x = (1.0 / n) * np.sum((x - mean_x[:, np.newaxis]) ** 2)
    cov_xy = (1.0 / n) * (y - mean_y[:, np.newaxis]) @ (x - mean_x[:, np.newaxis]).T
    u, d, vh = np.linalg.svd(cov_xy)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh.T) < 0:
        s_mat[2, 2] = -1
    R = u @ s_mat @ vh
    if with_scale:
        scale = (1.0 / sigma_x) * np.trace(np.diag(d) @ s_mat)
    else:
        scale = 1.0
    t = mean_y - scale * (R @ mean_x)
    return R, t, scale


def associate_and_align(gt_list, est_list):
    """
    gt_list, est_list: list of (timestamp, 4x4 pose).
    For each GT timestamp, find nearest estimate by timestamp.
    Run Umeyama on positions (3xN), apply transform to estimate positions.
    Returns:
      relative_time: (N,) relative time in sec
      error_per_pose: (N,) translation error in meters
      frame_index: (N,) estimate index for each GT point
      t_gt_min: float, for relative time
      rmse, median_error: scalars
    """
    gt_ts = np.array([p[0] for p in gt_list])
    gt_poses = np.array([p[1] for p in gt_list])
    est_ts = np.array([p[0] for p in est_list])
    est_poses = np.array([p[1] for p in est_list])

    t_gt_min = float(np.min(gt_ts))
    n_gt = len(gt_list)
    frame_indices = []
    gt_positions = []
    est_positions = []
    for i in range(n_gt):
        t_gt = gt_ts[i]
        j = int(np.argmin(np.abs(est_ts - t_gt)))
        j = min(j, len(est_list) - 1)
        frame_indices.append(j)
        gt_positions.append(gt_poses[i][:3, 3])
        est_positions.append(est_poses[j][:3, 3])

    gt_pos = np.array(gt_positions).T   # 3 x N
    est_pos = np.array(est_positions).T  # 3 x N
    R, t, s = umeyama_alignment(est_pos, gt_pos, with_scale=True)
    est_aligned = (s * (R @ est_pos) + t.reshape(3, 1))
    errors = np.linalg.norm(est_aligned - gt_pos, axis=0)
    relative_time = gt_ts - t_gt_min
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_error = float(np.median(errors))
    return relative_time, errors, np.array(frame_indices), t_gt_min, rmse, median_error


def load_images(image_folder, ext=".png", max_count=None):
    """Load image paths in sorted order (same as dataset). Returns list of Path."""
    folder = Path(image_folder)
    paths = sorted(folder.glob(f"*{ext}"))
    if max_count is not None:
        paths = paths[:max_count]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Post-hoc: frame + error-vs-time playback viewer")
    parser.add_argument("--estimate", "-e", required=True, help="Estimated trajectory TUM file")
    parser.add_argument("--ground_truth", "-g", required=True, help="Ground truth trajectory TUM file")
    parser.add_argument("--images", "-i", default=None, help="Image folder for frame panel (optional)")
    parser.add_argument("--ext", default=".png", help="Image extension (default .png)")
    args = parser.parse_args()

    est_list = load_tum_with_timestamps(args.estimate)
    gt_list = load_tum_with_timestamps(args.ground_truth)
    if not est_list or not gt_list:
        print("Need non-empty estimate and ground truth TUM files.", file=sys.stderr)
        sys.exit(1)

    rel_time, errors, frame_index, t_gt_min, rmse, median_err = associate_and_align(gt_list, est_list)
    n_pts = len(rel_time)
    n_est = len(est_list)  # number of image frames (estimate poses)
    # For each image frame index, get a representative relative time for the vline (first GT pair that used this frame)
    rel_time_for_frame = {}
    for i in range(n_pts):
        fi = int(frame_index[i])
        if fi not in rel_time_for_frame:
            rel_time_for_frame[fi] = rel_time[i]
    for f in range(n_est):
        if f not in rel_time_for_frame:
            rel_time_for_frame[f] = rel_time_for_frame.get(f - 1, rel_time[0])
    print(f"Associated {n_pts} pairs, {n_est} image frames. RMSE={rmse:.4f} m, median={median_err:.4f} m")

    image_paths = None
    if args.images:
        image_paths = load_images(args.images, args.ext, max_count=len(est_list))
        if len(image_paths) < n_pts:
            print(f"Warning: only {len(image_paths)} images; frame index will be clamped.", file=sys.stderr)

    # Build figure: left = image, right = error plot
    fig, (ax_img, ax_err) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.18)

    # Error plot: curve + RMSE/median lines
    ax_err.plot(rel_time, errors, "b-", linewidth=0.8, label="error")
    ax_err.axhline(rmse, color="red", linestyle="--", linewidth=1, label=f"RMSE {rmse:.3f} m")
    ax_err.axhline(median_err, color="orange", linestyle="--", linewidth=1, label=f"median {median_err:.3f} m")
    ax_err.set_xlabel("Relative time (s)")
    ax_err.set_ylabel("Translation error (m)")
    ax_err.set_title("Error vs time (click to jump, hover for timestamp/frame)")
    ax_err.legend(loc="upper right", fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(rel_time[0], rel_time[-1] if n_pts > 1 else rel_time[0] + 0.1)

    # Vertical line for current time (position from first frame's representative time)
    vline = ax_err.axvline(rel_time_for_frame[0], color="green", linewidth=2, alpha=0.8)

    # Image panel: show frame at current index
    if image_paths:
        img0 = cv2.imread(str(image_paths[0]))
        if img0 is None:
            img0 = np.zeros((240, 320, 3), dtype=np.uint8)
            img0[:] = (40, 40, 40)
        if len(img0.shape) == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        im_show = ax_img.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        ax_img.set_title("Frame (click graph or use slider)")
        ax_img.axis("off")
    else:
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        placeholder[:] = (60, 60, 60)
        im_show = ax_img.imshow(placeholder)
        ax_img.set_title("No image folder (use --images)")
        ax_img.axis("off")

    # Prev / Play-Pause / Next buttons and rate (FPS) slider
    ax_prev = fig.add_axes([0.05, 0.04, 0.06, 0.04])
    ax_play = fig.add_axes([0.12, 0.04, 0.06, 0.04])
    ax_next = fig.add_axes([0.19, 0.04, 0.06, 0.04])
    btn_prev = Button(ax_prev, "Prev", color="0.85", hovercolor="0.95")
    btn_play = Button(ax_play, "Play", color="0.85", hovercolor="0.95")
    btn_next = Button(ax_next, "Next", color="0.85", hovercolor="0.95")
    ax_fps = fig.add_axes([0.26, 0.045, 0.10, 0.025])
    fps_slider = Slider(ax_fps, "FPS", 0.5, 15, valinit=5, valstep=0.5)
    ax_slider = fig.add_axes([0.38, 0.06, 0.52, 0.03])
    slider = Slider(ax_slider, "Frame", 0, max(n_est - 1, 0), valinit=0, valstep=1)

    # Hover annotation on error plot (only redraw when hover index changes)
    annot = ax_err.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    annot.set_visible(False)
    _last_hover_i = [None]

    def get_frame_index(slider_val):
        idx = int(round(slider_val))
        idx = max(0, min(idx, n_est - 1))
        return idx

    # Cache for loaded images to avoid repeated disk reads (max 50 frames)
    _image_cache = {}
    _CACHE_MAX = 50
    _last_display_idx = [None]

    def update_display(frame_idx):
        frame_idx = max(0, min(frame_idx, n_est - 1))
        if _last_display_idx[0] == frame_idx:
            return
        _last_display_idx[0] = frame_idx
        vline.set_xdata([rel_time_for_frame[frame_idx]])
        if image_paths:
            fi = min(frame_idx, len(image_paths) - 1)
            if fi in _image_cache:
                im_show.set_data(_image_cache[fi])
            else:
                img = cv2.imread(str(image_paths[fi]))
                if img is not None:
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if len(_image_cache) >= _CACHE_MAX:
                        _image_cache.pop(next(iter(_image_cache)))
                    _image_cache[fi] = rgb
                    im_show.set_data(rgb)
        fig.canvas.draw_idle()

    def on_slider(val):
        idx = get_frame_index(val)
        update_display(idx)

    playing = [False]  # use list so closure can mutate
    timer = [None]

    def get_play_interval_ms():
        fps = max(0.5, float(fps_slider.val))
        return int(1000 / fps)

    def start_timer():
        if timer[0] is not None:
            timer[0].stop()
        interval_ms = get_play_interval_ms()
        timer[0] = fig.canvas.new_timer(interval=interval_ms)
        timer[0].add_callback(play_step)
        timer[0].start()

    def stop_timer():
        if timer[0] is not None:
            timer[0].stop()

    def play_step():
        if not playing[0] or n_est == 0:
            return
        idx = get_frame_index(slider.val)
        idx = min(idx + 1, n_est - 1)
        slider.set_val(idx)
        update_display(idx)
        if idx >= n_est - 1:
            playing[0] = False
            btn_play.label.set_text("Play")
            stop_timer()
        elif timer[0] is not None:
            timer[0].start()

    def on_play_click(_):
        playing[0] = not playing[0]
        if playing[0]:
            btn_play.label.set_text("Pause")
            start_timer()
        else:
            btn_play.label.set_text("Play")
            stop_timer()

    def on_fps_change(_):
        if playing[0]:
            start_timer()

    def on_key(event):
        if event.key in ("left", "right"):
            playing[0] = False
            btn_play.label.set_text("Play")
            stop_timer()
            idx = get_frame_index(slider.val)
            idx += -1 if event.key == "left" else 1
            idx = max(0, min(idx, n_est - 1))
            slider.set_val(idx)
            update_display(idx)
        elif event.key == " ":
            playing[0] = not playing[0]
            if playing[0]:
                btn_play.label.set_text("Pause")
                start_timer()
            else:
                btn_play.label.set_text("Play")
                stop_timer()

    def on_motion(event):
        if event.inaxes != ax_err or n_pts == 0:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        x = event.xdata
        if x is None:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        i = int(np.argmin(np.abs(rel_time - x)))
        i = max(0, min(i, n_pts - 1))
        if _last_hover_i[0] == i and annot.get_visible():
            return
        _last_hover_i[0] = i
        t = rel_time[i]
        fi = int(frame_index[i])
        err = errors[i]
        annot.xy = (t, err)
        annot.set_text(f"t = {t:.2f} s\nframe = {fi}\nerror = {err:.4f} m")
        annot.set_visible(True)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_err or event.button != 1 or n_pts == 0:
            return
        x = event.xdata
        if x is None:
            return
        i = int(np.argmin(np.abs(rel_time - x)))
        i = max(0, min(i, n_pts - 1))
        fi = int(frame_index[i])
        fi = max(0, min(fi, n_est - 1))
        slider.set_val(fi)
        update_display(fi)

    def on_prev(_):
        idx = get_frame_index(slider.val)
        idx = max(0, idx - 1)
        slider.set_val(idx)
        update_display(idx)

    def on_next(_):
        idx = get_frame_index(slider.val)
        idx = min(n_est - 1, idx + 1)
        slider.set_val(idx)
        update_display(idx)

    btn_prev.on_clicked(on_prev)
    btn_play.on_clicked(on_play_click)
    btn_next.on_clicked(on_next)
    fps_slider.on_changed(on_fps_change)
    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.suptitle(f"Post-hoc error viewer  |  RMSE = {rmse:.4f} m  |  median = {median_err:.4f} m", fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
