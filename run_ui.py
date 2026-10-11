"""
VSLAM configuration UI. Run with: python run_ui.py
All fields start with defaults; user may change any value and click Run.
"""

import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

# Import from main so we can get defaults and run the pipeline
from main import get_parser, run_pipeline
import pipeline as pipeline_module


def get_defaults():
    """Return namespace of default argument values."""
    parser = get_parser()
    return parser.parse_args([])


def make_readonly_combobox(parent, values, default_index=0, width=18):
    """Combobox that only allows selection from values (no free text)."""
    var = tk.StringVar(value=values[default_index] if values else "")
    cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
    cb.current(default_index)
    return cb, var


def make_section(parent, title):
    """Create a labeled frame section."""
    frame = ttk.LabelFrame(parent, text=title, padding=5)
    return frame


def _balanced_multiscale_defaults():
    """Balanced preset defaults (matching OpenCV documented defaults)."""
    return {
        "orb_nfeatures": 500,
        "orb_scale_factor": 1.2,
        "orb_nlevels": 8,
        "sift_nfeatures": 0,
        "sift_n_octave_layers": 3,
        "sift_contrast_threshold": 0.04,
        "sift_edge_threshold": 10.0,
        "sift_sigma": 1.6,
    }


def on_method_change(app):
    """When method changes, show only that method's manual params (if manual mode is enabled)."""
    if not getattr(app, "multiscale_manual_var", None):
        return
    if not app.multiscale_manual_var.get():
        return
    method = app.method_var.get()
    if method == "ORB":
        app.multiscale_orb_frame.grid()
        app.multiscale_sift_frame.grid_remove()
    else:
        app.multiscale_sift_frame.grid()
        app.multiscale_orb_frame.grid_remove()


def on_multiscale_manual_change(app):
    """
    UI behavior:
    - Manual unchecked (default): preset dropdown enabled; manual params hidden
    - Manual checked: preset dropdown disabled; manual params shown for current method only
      and populated with Balanced defaults.
    """
    manual = app.multiscale_manual_var.get()
    if manual:
        # Disable preset selection
        app.multiscale_preset_cb.config(state="disabled")
        # Populate manual fields with Balanced defaults
        d = _balanced_multiscale_defaults()
        app.orb_nfeatures_var.set(str(d["orb_nfeatures"]))
        app.orb_scale_factor_var.set(str(d["orb_scale_factor"]))
        app.orb_nlevels_var.set(str(d["orb_nlevels"]))
        app.sift_nfeatures_var.set(str(d["sift_nfeatures"]))
        app.sift_n_octave_layers_var.set(str(d["sift_n_octave_layers"]))
        app.sift_contrast_threshold_var.set(str(d["sift_contrast_threshold"]))
        app.sift_edge_threshold_var.set(str(d["sift_edge_threshold"]))
        app.sift_sigma_var.set(str(d["sift_sigma"]))
        app.multiscale_manual_frame.grid()
        on_method_change(app)
    else:
        app.multiscale_preset_cb.config(state="readonly")
        app.multiscale_manual_frame.grid_remove()

def on_loop_detection_strategy_change(app):
    """
    When loop detection strategy is 'bow', auto-fill a sensible default vocabulary path
    (only if the field is currently empty).
    """
    try:
        strategy = app.loop_detection_strategy_var.get()
    except Exception:
        return
    if strategy != "bow":
        return
    default_vocab = str((Path(__file__).parent / ".deps/loop-closure-toolbox/config/sthereo_01_rgb_4_3.yaml").resolve())
    try:
        cur = (app.vocabulary_path_var.get() or "").strip()
    except Exception:
        cur = ""
    if not cur:
        app.vocabulary_path_var.set(default_vocab)


def _make_scrollable(parent):
    """Create a scrollable frame: returns (canvas, content_frame). Pack content into content_frame."""
    container = ttk.Frame(parent)
    container.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    content_frame = ttk.Frame(canvas)
    canvas_window = canvas.create_window(0, 0, window=content_frame, anchor=tk.NW)

    def _on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Keep content width in sync so horizontal scrollbar doesn't appear
        canvas.itemconfig(canvas_window, width=canvas.winfo_width())

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)

    content_frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_linux_mousewheel(event):
        if event.num == 4:
            canvas.yview_scroll(-4, "units")
        elif event.num == 5:
            canvas.yview_scroll(4, "units")

    def _bind_mousewheel(widget):
        widget.bind("<MouseWheel>", _on_mousewheel)
        widget.bind("<Button-4>", _on_linux_mousewheel)
        widget.bind("<Button-5>", _on_linux_mousewheel)

    _bind_mousewheel(canvas)
    _bind_mousewheel(content_frame)
    # Bind to top-level so scrolling works when pointer is over any child widget
    toplevel = canvas.winfo_toplevel()
    toplevel.bind("<MouseWheel>", _on_mousewheel)
    toplevel.bind("<Button-4>", _on_linux_mousewheel)
    toplevel.bind("<Button-5>", _on_linux_mousewheel)

    return canvas, content_frame


def build_invocation_string(args):
    """Build a CLI-style invocation string from args (all parameters, including defaults) for the log file."""
    tokens = ["main.py"]
    tokens.append("--dataset")
    tokens.append(str(args.dataset))
    tokens.append("--image_ext")
    tokens.append(str(getattr(args, "image_ext", ".png")))
    tokens.append("--method")
    tokens.append(str(args.method))
    tokens.append("--multiscale_preset")
    tokens.append(str(getattr(args, "multiscale_preset", "balanced")))
    if getattr(args, "multiscale_manual", False):
        tokens.append("--multiscale_manual")
        # Only include detailed params for the chosen method (keeps log readable)
        if getattr(args, "method", "ORB") == "ORB":
            tokens.append("--orb_nfeatures")
            tokens.append(str(getattr(args, "orb_nfeatures", 500)))
            tokens.append("--orb_scale_factor")
            tokens.append(str(getattr(args, "orb_scale_factor", 1.2)))
            tokens.append("--orb_nlevels")
            tokens.append(str(getattr(args, "orb_nlevels", 8)))
        else:
            tokens.append("--sift_nfeatures")
            tokens.append(str(getattr(args, "sift_nfeatures", 0)))
            tokens.append("--sift_n_octave_layers")
            tokens.append(str(getattr(args, "sift_n_octave_layers", 3)))
            tokens.append("--sift_contrast_threshold")
            tokens.append(str(getattr(args, "sift_contrast_threshold", 0.04)))
            tokens.append("--sift_edge_threshold")
            tokens.append(str(getattr(args, "sift_edge_threshold", 10.0)))
            tokens.append("--sift_sigma")
            tokens.append(str(getattr(args, "sift_sigma", 1.6)))
    tokens.append("--pnp_interval")
    tokens.append(str(args.pnp_interval))
    tokens.append("--pnp_solver")
    tokens.append(str(args.pnp_solver))
    tokens.append("--no_pnp_refinement" if not args.pnp_refinement else "--pnp_refinement")
    tokens.append("--no_pnp_initial_guess" if not args.pnp_use_initial_guess else "--pnp_use_initial_guess")
    tokens.append("--pnp_min_inlier_ratio")
    tokens.append(str(args.pnp_min_inlier_ratio))
    tokens.append("--pnp_multi_inlier_gate")
    tokens.append(str(args.pnp_multi_inlier_gate))
    tokens.append("--pnp_reprojection_error")
    tokens.append(str(getattr(args, "pnp_reprojection_error", 3.0)))
    if getattr(args, "pnp_adaptive_gate", False):
        tokens.append("--pnp_adaptive_gate")
        tokens.append("--pnp_adaptive_gate_factor")
        tokens.append(str(getattr(args, "pnp_adaptive_gate_factor", 0.2)))
    tokens.append("--ba_interval")
    tokens.append(str(args.ba_interval))
    tokens.append("--ba_mode")
    tokens.append(str(getattr(args, "ba_mode", "sliding_window")))
    tokens.append("--ba_window_size")
    tokens.append(str(args.ba_window_size))
    tokens.append("--ba_max_points")
    tokens.append(str(args.ba_max_points))
    tokens.append("--ba_max_allowed_step_factor")
    tokens.append(str(getattr(args, "ba_max_allowed_step_factor", 3.0)))
    tokens.append("--ba_bad_reproj_ratio")
    tokens.append(str(getattr(args, "ba_bad_reproj_ratio", 1.2)))
    tokens.append("--no_ba" if not args.use_ba else "--use_ba")
    tokens.append("--ba_use_numerical_jacobian" if not getattr(args, "ba_use_analytical_jacobian", True) else "--ba_use_analytical_jacobian")
    tokens.append("--no_homography" if not args.use_homography else "--use_homography")
    tokens.append("--ba_method")
    tokens.append(str(args.ba_method))
    # Keyframe-local BA parameters
    tokens.append("--keyframe_local_topk")
    tokens.append(str(getattr(args, "keyframe_local_topk", 5)))
    tokens.append("--keyframe_local_min_shared_points")
    tokens.append(str(getattr(args, "keyframe_local_min_shared_points", 20)))
    # --- Map points (BA + pruning) ---
    tokens.append("--ba_min_point_observations")
    tokens.append(str(getattr(args, "ba_min_point_observations", 3)))
    tokens.append("--no_ba_point_sanity_check" if not getattr(args, "ba_point_sanity_check", True) else "--ba_point_sanity_check")
    # Stage 04-1a: the low-observation age prune and its flags were retired; landmark culling
    # is the local mapper's (--lm_cull_after_keyframes / --lm_min_keyframe_obs, defaults).
    tokens.append("--prune_interval")
    tokens.append(str(getattr(args, "prune_interval", 30)))
    tokens.append("--no_prune_high_reproj" if not getattr(args, "prune_high_reproj", False) else "--prune_high_reproj")
    tokens.append("--no_prune_numeric_corruption" if not getattr(args, "prune_numeric_corruption", False) else "--prune_numeric_corruption")
    tokens.append("--no_prune_outlier_hits" if not getattr(args, "prune_outlier_hits", False) else "--prune_outlier_hits")
    if getattr(args, "prune_outlier_hits", False):
        tokens.append("--prune_outlier_hit_count")
        tokens.append(str(getattr(args, "prune_outlier_hit_count", 3)))
        tokens.append("--prune_outlier_last_seen_age")
        tokens.append(str(getattr(args, "prune_outlier_last_seen_age", 20)))
    tokens.append("--local_map_window")
    tokens.append(str(args.local_map_window))
    # Stage 04-1a: per-frame triangulation flags (--min_parallax_deg, multi-frame triangulation)
    # were retired; the keyframe-pair source runs with its --lm_* defaults.
    if not getattr(args, "use_relative_scale", False):
        tokens.append("--no_relative_scale")
    if args.log_file:
        tokens.append("--log_file")
        tokens.append(str(args.log_file))
    if args.no_display:
        tokens.append("--no_display")
    if args.save_estimated:
        tokens.append("--save_estimated")
        tokens.append(str(args.save_estimated))
    if args.timestamp_file:
        tokens.append("--timestamp_file")
        tokens.append(str(args.timestamp_file))
    if args.default_k:
        tokens.append("--default_k")
    if args.camera_spacing is not None:
        tokens.append("--camera_spacing")
        tokens.append(str(args.camera_spacing))
    if args.ground_truth:
        tokens.append("--ground_truth")
        tokens.append(str(args.ground_truth))
    tokens.append("--viewer_fps")
    tokens.append(str(args.viewer_fps))
    if getattr(args, 'tracker_delay_ms', 0):
        tokens.append("--tracker_delay_ms")
        tokens.append(str(args.tracker_delay_ms))
    tokens.append("--no_show_grid" if not args.show_grid else "--show_grid")
    if not args.use_loop_closure:
        tokens.append("--no_loop_closure")
    else:
        tokens.append("--use_loop_closure")
        tokens.append("--loop_closure_strategy")
        tokens.append(str(args.loop_closure_strategy))
    tokens.append("--loop_detection_strategy")
    tokens.append(str(getattr(args, "loop_detection_strategy", "spaced")))
    if getattr(args, "vocabulary_path", None):
        tokens.append("--vocabulary_path")
        tokens.append(str(args.vocabulary_path))
    tokens.append("--num_spaced_candidates")
    tokens.append(str(getattr(args, "num_spaced_candidates", 5)))
    tokens.append("--min_gap")
    tokens.append(str(getattr(args, "min_gap", 100)))
    tokens.append("--keyframe_strategy")
    tokens.append(str(args.keyframe_strategy))
    tokens.append("--keyframe_interval")
    tokens.append(str(args.keyframe_interval))
    tokens.append("--keyframe_min_spacing")
    tokens.append(str(args.keyframe_min_spacing))
    tokens.append("--keyframe_max_spacing")
    tokens.append(str(args.keyframe_max_spacing))
    tokens.append("--keyframe_ratio_threshold")
    tokens.append(str(args.keyframe_ratio_threshold))
    if not getattr(args, "use_relocalization", True):
        tokens.append("--no_relocalization")
    else:
        tokens.append("--use_relocalization")
        tokens.append("--relocalization_match_threshold")
        tokens.append(str(getattr(args, "relocalization_match_threshold", 25)))
        tokens.append("--relocalization_min_2d3d_pairs")
        tokens.append(str(getattr(args, "relocalization_min_2d3d_pairs", 10)))
        tokens.append("--relocalization_min_pnp_inliers")
        tokens.append(str(getattr(args, "relocalization_min_pnp_inliers", 10)))
        tokens.append("--relocalization_ratio_threshold")
        tokens.append(str(getattr(args, "relocalization_ratio_threshold", 0.85)))
        if getattr(args, "relocalization_sanity_check_enabled", False):
            tokens.append("--relocalization_sanity_check")
            tokens.append("--relocalization_max_distance_from_keyframe")
            tokens.append(str(getattr(args, "relocalization_max_distance_from_keyframe", 2.0)))
    return " ".join(tokens)


def run_from_ui(app):
    """Collect form values into args namespace and call run_pipeline."""
    def _get_int(var, default=None):
        s = var.get().strip()
        if not s and default is not None:
            return default
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return default

    def _get_float(var, default=0.0):
        s = var.get().strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            return default

    def _get_str(var, empty_as_none=True):
        s = var.get().strip()
        if empty_as_none and not s:
            return None
        return s or None

    ba_val = app.ba_var.get()
    use_ba = ba_val != "no_ba"
    ba_method = ba_val if use_ba else "lm"

    pnp_val = app.pnp_solver_var.get()
    if pnp_val == "no_pnp":
        pnp_interval = 0
        pnp_solver = "iterative"  # unused when interval 0
    else:
        pnp_interval = _get_int(app.pnp_interval_var, 10)
        pnp_solver = pnp_val

    display_val = app.display_var.get()
    no_display = display_val == "No display"

    try:
        args = SimpleNamespace(
            dataset=_get_str(app.dataset_var) or "rgb",
            image_ext=_get_str(app.image_ext_var) or ".png",
            method=app.method_var.get(),
            multiscale_preset=getattr(app, "multiscale_preset_var", tk.StringVar(value="balanced")).get(),
            multiscale_manual=getattr(app, "multiscale_manual_var", tk.BooleanVar(value=False)).get(),
            orb_nfeatures=_get_int(getattr(app, "orb_nfeatures_var", tk.StringVar(value="500")), 500),
            orb_scale_factor=_get_float(getattr(app, "orb_scale_factor_var", tk.StringVar(value="1.2")), 1.2),
            orb_nlevels=_get_int(getattr(app, "orb_nlevels_var", tk.StringVar(value="8")), 8),
            sift_nfeatures=_get_int(getattr(app, "sift_nfeatures_var", tk.StringVar(value="0")), 0),
            sift_n_octave_layers=_get_int(getattr(app, "sift_n_octave_layers_var", tk.StringVar(value="3")), 3),
            sift_contrast_threshold=_get_float(getattr(app, "sift_contrast_threshold_var", tk.StringVar(value="0.04")), 0.04),
            sift_edge_threshold=_get_float(getattr(app, "sift_edge_threshold_var", tk.StringVar(value="10.0")), 10.0),
            sift_sigma=_get_float(getattr(app, "sift_sigma_var", tk.StringVar(value="1.6")), 1.6),
            pnp_interval=pnp_interval,
            pnp_solver=pnp_solver,
        pnp_refinement=False if pnp_solver == "iterative" else app.pnp_refinement_var.get(),
        pnp_use_initial_guess=app.pnp_use_initial_guess_var.get(),
        pnp_min_inlier_ratio=_get_float(app.pnp_min_inlier_ratio_var, 0.55),
        pnp_multi_inlier_gate=_get_float(app.pnp_multi_inlier_gate_var, 0.8),
        pnp_reprojection_error=_get_float(app.pnp_reprojection_error_var, 3.0),
        pnp_adaptive_gate=app.pnp_adaptive_gate_var.get(),
        pnp_adaptive_gate_factor=_get_float(app.pnp_adaptive_gate_factor_var, 0.2),
        ba_interval=_get_int(app.ba_interval_var, 30),
        ba_window_size=_get_int(app.ba_window_size_var, 12),
        ba_mode=app.ba_mode_var.get() if getattr(app, "ba_mode_var", None) is not None else "sliding_window",
        keyframe_local_topk=_get_int(app.keyframe_local_topk_var, 5),
        keyframe_local_min_shared_points=_get_int(app.keyframe_local_min_shared_points_var, 20),
        ba_max_points=_get_int(app.ba_max_points_var, 250),
        ba_max_allowed_step_factor=_get_float(app.ba_max_allowed_step_factor_var, 3.0),
        ba_bad_reproj_ratio=_get_float(app.ba_bad_reproj_ratio_var, 1.2),
        use_ba=use_ba,
        use_homography=app.use_homography_var.get(),
        use_relative_scale=app.use_relative_scale_var.get(),
        ba_method=ba_method,
        ba_use_analytical_jacobian=(app.ba_jacobian_var.get() == "analytical"),
        ba_min_point_observations=_get_int(app.ba_min_point_observations_var, 3),
        ba_point_sanity_check=app.ba_point_sanity_check_var.get(),
        prune_interval=_get_int(app.prune_interval_var, 30),
        prune_high_reproj=app.prune_high_reproj_var.get(),
        prune_numeric_corruption=app.prune_numeric_corruption_var.get(),
        prune_outlier_hits=app.prune_outlier_hits_var.get(),
        prune_outlier_hit_count=_get_int(app.prune_outlier_hit_count_var, 3),
        prune_outlier_last_seen_age=_get_int(app.prune_outlier_last_seen_age_var, 20),
        local_map_window=_get_int(app.local_map_window_var, 0),
        log_file=_get_str(app.log_file_var),
        no_display=no_display,
        save_estimated=_get_str(app.save_estimated_var),
        timestamp_file=_get_str(app.timestamp_file_var),
        t_norm_log=None,
        default_k=app.default_k_var.get(),
        camera_spacing=_get_int(app.camera_spacing_var) if not no_display else None,
        ground_truth=_get_str(app.ground_truth_var),
        viewer_fps=_get_float(app.viewer_fps_var, 30.0),
        tracker_delay_ms=_get_float(app.tracker_delay_ms_var, 0.0),
        show_grid=app.show_grid_var.get() if not no_display else True,
        use_loop_closure=(app.loop_closure_var.get() != "No loop closure"),
        loop_closure_strategy=(
            "pose_graph" if app.loop_closure_var.get() == "Pose graph optimization" else "linear"
        ),
        loop_detection_strategy=app.loop_detection_strategy_var.get(),
        vocabulary_path=_get_str(app.vocabulary_path_var),
        num_spaced_candidates=_get_int(app.num_spaced_candidates_var, 5),
        min_gap=_get_int(app.min_gap_var, 100),
        keyframe_strategy=app.keyframe_strategy_var.get(),
        keyframe_interval=_get_int(app.keyframe_interval_var, 20),
        keyframe_min_spacing=_get_int(app.keyframe_min_spacing_var, 5),
        keyframe_max_spacing=_get_int(app.keyframe_max_spacing_var, 50),
        keyframe_ratio_threshold=_get_float(app.keyframe_ratio_threshold_var, 0.80),
        use_relocalization=app.use_relocalization_var.get(),
        relocalization_match_threshold=_get_int(app.relocalization_match_threshold_var, 25),
        relocalization_min_2d3d_pairs=_get_int(app.relocalization_min_2d3d_pairs_var, 10),
        relocalization_min_pnp_inliers=_get_int(app.relocalization_min_pnp_inliers_var, 10),
        relocalization_ratio_threshold=_get_float(app.relocalization_ratio_threshold_var, 0.85),
        relocalization_sanity_check_enabled=app.relocalization_sanity_check_var.get(),
        relocalization_max_distance_from_keyframe=_get_float(app.relocalization_max_distance_from_keyframe_var, 2.0),
        )
        # So the log file gets a CLI-style invocation line at the top (like when running from terminal)
        args._invocation = build_invocation_string(args)
    except (ValueError, TypeError) as e:
        messagebox.showerror("Invalid value", f"Please check numeric and choice fields.\n{e}")
        return

    # Disable Run and enable Stop; run pipeline in a background thread
    app.run_btn.config(state="disabled")
    app.stop_btn.config(state="normal")

    def run_in_thread():
        try:
            run_pipeline(args)
        finally:
            # Re-enable Run and disable Stop when done (on main thread)
            app.root.after(0, lambda: on_run_finished(app))

    app.pipeline_thread = threading.Thread(target=run_in_thread, daemon=False)
    app.pipeline_thread.start()


def on_run_finished(app):
    """Called when the pipeline run finishes (on main thread)."""
    app.run_btn.config(state="normal")
    app.stop_btn.config(state="disabled")
    # Optionally run playback error viewer after pipeline
    if getattr(app, "run_playback_after_var", None) and app.run_playback_after_var.get():
        est = (app.save_estimated_var.get() or "").strip()
        gt = (app.ground_truth_var.get() or "").strip()
        dataset = (app.dataset_var.get() or "rgb").strip()
        if est and gt:
            base = Path(__file__).resolve().parent
            script = base / "scripts" / "playback_error_viewer.py"
            if script.is_file():
                cmd = [
                    sys.executable,
                    str(script),
                    "-e", est,
                    "-g", gt,
                    "-i", dataset,
                ]
                try:
                    subprocess.Popen(cmd, cwd=str(base))
                except Exception as e:
                    messagebox.showerror("Playback viewer", f"Failed to start playback viewer: {e}")
            else:
                messagebox.showwarning("Playback viewer", f"Script not found: {script}")


def request_stop():
    """Set the pipeline stop flag (same as typing 'q' or 'quit' in terminal)."""
    pipeline_module._stop_requested = True


def on_pnp_adaptive_gate_change(app):
    """Show pnp_adaptive_gate_factor row only when adaptive gate is enabled; enable/disable factor entry."""
    is_no_pnp = app.pnp_solver_var.get() == "no_pnp"
    adaptive_on = app.pnp_adaptive_gate_var.get()
    if is_no_pnp or not adaptive_on:
        app.pnp_adaptive_gate_factor_row.grid_remove()
        app.pnp_adaptive_gate_factor_entry.config(state="disabled")
    else:
        app.pnp_adaptive_gate_factor_row.grid()
        app.pnp_adaptive_gate_factor_entry.config(state="normal")


def on_pnp_choice_change(app):
    """When no_pnp: disable all other PnP options. Otherwise enable them; gate multi_inlier on multi; grey out refinement when iterative. Factor row only when adaptive gate on."""
    is_no_pnp = app.pnp_solver_var.get() == "no_pnp"
    state = "disabled" if is_no_pnp else "normal"
    for w in (app.pnp_interval_entry, app.pnp_refinement_cb, app.pnp_use_initial_guess_cb,
              app.pnp_min_inlier_ratio_entry, app.pnp_multi_inlier_entry,
              app.pnp_reprojection_error_entry,
              app.pnp_adaptive_gate_cb):
        w.config(state=state)
    if is_no_pnp:
        app.pnp_adaptive_gate_factor_row.grid_remove()
        app.pnp_adaptive_gate_factor_entry.config(state="disabled")
    else:
        is_multi = app.pnp_solver_var.get() == "multi"
        app.pnp_multi_inlier_entry.config(state="normal" if is_multi else "disabled")
        is_iterative = app.pnp_solver_var.get() == "iterative"
        app.pnp_refinement_cb.config(state="disabled" if is_iterative else "normal")
        on_pnp_adaptive_gate_change(app)


def on_ba_choice_change(app):
    """When no_ba: disable BA widgets. Otherwise enable them; BA-mode UI also follows BA enablement."""
    is_no_ba = app.ba_var.get() == "no_ba"
    state = "disabled" if is_no_ba else "normal"
    for w in (app.ba_interval_entry, app.ba_window_size_entry, app.ba_max_points_entry,
              app.ba_max_allowed_step_factor_entry, app.ba_bad_reproj_ratio_entry,
              app.ba_jacobian_cb,
              getattr(app, "ba_mode_cb", None),
              getattr(app, "keyframe_local_topk_entry", None),
              getattr(app, "keyframe_local_min_shared_points_entry", None)):
        if w is None:
            continue
        w.config(state=state)
    # BA-related map-point options (only editable when BA enabled)
    for w in (getattr(app, "ba_min_point_observations_entry", None),
              getattr(app, "ba_point_sanity_check_cb", None),
              getattr(app, "prune_high_reproj_cb", None)):
        if w is not None:
            w.config(state=state)


def on_ba_mode_change(app):
    """Show/hide BA-specific parameters based on ba_mode selection."""
    mode = app.ba_mode_var.get()
    is_sliding = mode == "sliding_window"

    if is_sliding:
        app.ba_interval_label.grid()
        app.ba_interval_entry.grid()
        app.ba_window_size_label.grid()
        app.ba_window_size_entry.grid()
        app.keyframe_local_topk_label.grid_remove()
        app.keyframe_local_topk_entry.grid_remove()
        app.keyframe_local_min_shared_points_label.grid_remove()
        app.keyframe_local_min_shared_points_entry.grid_remove()
    else:
        app.ba_interval_label.grid_remove()
        app.ba_interval_entry.grid_remove()
        app.ba_window_size_label.grid_remove()
        app.ba_window_size_entry.grid_remove()
        app.keyframe_local_topk_label.grid()
        app.keyframe_local_topk_entry.grid()
        app.keyframe_local_min_shared_points_label.grid()
        app.keyframe_local_min_shared_points_entry.grid()


def on_display_change(app):
    """Show/hide display-only options based on Display dropdown."""
    show = app.display_var.get() == "Display"
    if show:
        app.display_options_frame.grid()
    else:
        app.display_options_frame.grid_remove()


def main():
    defaults = get_defaults()

    root = tk.Tk()
    root.title("VSLAM Configuration")
    root.geometry("520x780")
    root.minsize(400, 500)

    _scroll_canvas, scroll_inner = _make_scrollable(root)
    main_frame = ttk.Frame(scroll_inner, padding=10)
    main_frame.pack(fill=tk.X)

    # --- Dataset & method ---
    sec = make_section(main_frame, "Dataset & method")
    sec.pack(fill=tk.X, pady=2)
    ttk.Label(sec, text="Dataset folder:").grid(row=0, column=0, sticky=tk.W, pady=1)
    app = SimpleNamespace()
    app.dataset_var = tk.StringVar(value=defaults.dataset)
    ttk.Entry(sec, textvariable=app.dataset_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(sec, text="Image extension:").grid(row=1, column=0, sticky=tk.W, pady=1)
    app.image_ext_var = tk.StringVar(value=getattr(defaults, "image_ext", ".png"))
    ttk.Entry(sec, textvariable=app.image_ext_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(sec, text="(e.g. .png, .jpg)").grid(row=1, column=2, sticky=tk.W, padx=2)
    ttk.Label(sec, text="Method:").grid(row=2, column=0, sticky=tk.W, pady=1)
    method_cb, app.method_var = make_readonly_combobox(sec, ["ORB", "SIFT"], default_index=["ORB", "SIFT"].index(defaults.method))
    method_cb.grid(row=2, column=1, sticky=tk.W, padx=5, pady=1)
    method_cb.bind("<<ComboboxSelected>>", lambda e: on_method_change(app))

    # --- Multi-scale (preset + manual tuning) ---
    ttk.Label(sec, text="Multi-scale preset:").grid(row=3, column=0, sticky=tk.W, pady=1)
    ms_choices = ["none", "fast", "balanced", "aggressive"]
    ms_default = getattr(defaults, "multiscale_preset", "balanced")
    ms_default_idx = ms_choices.index(ms_default) if ms_default in ms_choices else 1
    app.multiscale_preset_cb, app.multiscale_preset_var = make_readonly_combobox(sec, ms_choices, default_index=ms_default_idx, width=14)
    app.multiscale_preset_cb.grid(row=3, column=1, sticky=tk.W, padx=5, pady=1)

    app.multiscale_manual_var = tk.BooleanVar(value=getattr(defaults, "multiscale_manual", False))
    app.multiscale_manual_cb = ttk.Checkbutton(
        sec,
        text="Manually tune multiscale parameters",
        variable=app.multiscale_manual_var,
        command=lambda: on_multiscale_manual_change(app),
    )
    app.multiscale_manual_cb.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=1)

    # Manual parameter panel (hidden unless checkbox enabled)
    app.multiscale_manual_frame = ttk.Frame(sec)
    app.multiscale_manual_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=(20, 0), pady=(2, 0))

    # ORB manual params
    app.multiscale_orb_frame = ttk.Frame(app.multiscale_manual_frame)
    app.multiscale_orb_frame.grid(row=0, column=0, sticky=tk.W)
    d = _balanced_multiscale_defaults()
    app.orb_nfeatures_var = tk.StringVar(value=str(getattr(defaults, "orb_nfeatures", d["orb_nfeatures"])))
    app.orb_scale_factor_var = tk.StringVar(value=str(getattr(defaults, "orb_scale_factor", d["orb_scale_factor"])))
    app.orb_nlevels_var = tk.StringVar(value=str(getattr(defaults, "orb_nlevels", d["orb_nlevels"])))
    ttk.Label(app.multiscale_orb_frame, text="ORB nfeatures:").grid(row=0, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_orb_frame, textvariable=app.orb_nfeatures_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(app.multiscale_orb_frame, text="ORB scaleFactor:").grid(row=1, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_orb_frame, textvariable=app.orb_scale_factor_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(app.multiscale_orb_frame, text="ORB nlevels:").grid(row=2, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_orb_frame, textvariable=app.orb_nlevels_var, width=8).grid(row=2, column=1, sticky=tk.W, padx=5, pady=1)

    # SIFT manual params
    app.multiscale_sift_frame = ttk.Frame(app.multiscale_manual_frame)
    app.multiscale_sift_frame.grid(row=0, column=0, sticky=tk.W)
    app.sift_nfeatures_var = tk.StringVar(value=str(getattr(defaults, "sift_nfeatures", d["sift_nfeatures"])))
    app.sift_n_octave_layers_var = tk.StringVar(value=str(getattr(defaults, "sift_n_octave_layers", d["sift_n_octave_layers"])))
    app.sift_contrast_threshold_var = tk.StringVar(value=str(getattr(defaults, "sift_contrast_threshold", d["sift_contrast_threshold"])))
    app.sift_edge_threshold_var = tk.StringVar(value=str(getattr(defaults, "sift_edge_threshold", d["sift_edge_threshold"])))
    app.sift_sigma_var = tk.StringVar(value=str(getattr(defaults, "sift_sigma", d["sift_sigma"])))
    ttk.Label(app.multiscale_sift_frame, text="SIFT nfeatures (0=unlimited):").grid(row=0, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_sift_frame, textvariable=app.sift_nfeatures_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(app.multiscale_sift_frame, text="SIFT nOctaveLayers:").grid(row=1, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_sift_frame, textvariable=app.sift_n_octave_layers_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(app.multiscale_sift_frame, text="SIFT contrastThreshold:").grid(row=2, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_sift_frame, textvariable=app.sift_contrast_threshold_var, width=8).grid(row=2, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(app.multiscale_sift_frame, text="SIFT edgeThreshold:").grid(row=3, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_sift_frame, textvariable=app.sift_edge_threshold_var, width=8).grid(row=3, column=1, sticky=tk.W, padx=5, pady=1)
    ttk.Label(app.multiscale_sift_frame, text="SIFT sigma:").grid(row=4, column=0, sticky=tk.W, pady=1)
    ttk.Entry(app.multiscale_sift_frame, textvariable=app.sift_sigma_var, width=8).grid(row=4, column=1, sticky=tk.W, padx=5, pady=1)

    # Initial state: manual panel hidden unless defaults.multiscale_manual; show method-specific fields.
    if not app.multiscale_manual_var.get():
        app.multiscale_manual_frame.grid_remove()
    else:
        on_multiscale_manual_change(app)

    # --- PnP ---
    sec = make_section(main_frame, "PnP")
    sec.pack(fill=tk.X, pady=2)
    row = 0
    ttk.Label(sec, text="PnP (solver or off):").grid(row=row, column=0, sticky=tk.W, pady=1)
    pnp_choices = ["no_pnp", "iterative", "multi", "ap3p", "epnp", "sqpnp"]
    if defaults.pnp_interval == 0:
        pnp_default_idx = 0  # no_pnp
    else:
        pnp_default_idx = pnp_choices.index(defaults.pnp_solver) if defaults.pnp_solver in pnp_choices else 1
    pnp_cb, app.pnp_solver_var = make_readonly_combobox(sec, pnp_choices, default_index=pnp_default_idx, width=14)
    pnp_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    pnp_cb.bind("<<ComboboxSelected>>", lambda e: on_pnp_choice_change(app))
    row += 1
    ttk.Label(sec, text="pnp_interval:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.pnp_interval_var = tk.StringVar(value=str(defaults.pnp_interval if defaults.pnp_interval else 10))
    app.pnp_interval_entry = ttk.Entry(sec, textvariable=app.pnp_interval_var, width=8)
    app.pnp_interval_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    app.pnp_refinement_var = tk.BooleanVar(value=defaults.pnp_refinement)
    app.pnp_refinement_cb = ttk.Checkbutton(sec, text="pnp_refinement", variable=app.pnp_refinement_var)
    app.pnp_refinement_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
    row += 1
    app.pnp_use_initial_guess_var = tk.BooleanVar(value=defaults.pnp_use_initial_guess)
    app.pnp_use_initial_guess_cb = ttk.Checkbutton(sec, text="pnp_use_initial_guess", variable=app.pnp_use_initial_guess_var)
    app.pnp_use_initial_guess_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
    row += 1
    ttk.Label(sec, text="pnp_min_inlier_ratio:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.pnp_min_inlier_ratio_var = tk.StringVar(value=str(defaults.pnp_min_inlier_ratio))
    app.pnp_min_inlier_ratio_entry = ttk.Entry(sec, textvariable=app.pnp_min_inlier_ratio_var, width=8)
    app.pnp_min_inlier_ratio_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="pnp_multi_inlier_gate:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.pnp_multi_inlier_gate_var = tk.StringVar(value=str(defaults.pnp_multi_inlier_gate))
    app.pnp_multi_inlier_entry = ttk.Entry(sec, textvariable=app.pnp_multi_inlier_gate_var, width=8)
    app.pnp_multi_inlier_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="pnp_reprojection_error:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.pnp_reprojection_error_var = tk.StringVar(value=str(getattr(defaults, "pnp_reprojection_error", 3.0)))
    app.pnp_reprojection_error_entry = ttk.Entry(sec, textvariable=app.pnp_reprojection_error_var, width=8)
    app.pnp_reprojection_error_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    app.pnp_adaptive_gate_var = tk.BooleanVar(value=getattr(defaults, "pnp_adaptive_gate", False))
    app.pnp_adaptive_gate_cb = ttk.Checkbutton(
        sec, text="pnp_adaptive_gate (widen gate after consecutive PnP failures)",
        variable=app.pnp_adaptive_gate_var,
        command=lambda: on_pnp_adaptive_gate_change(app)
    )
    app.pnp_adaptive_gate_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
    row += 1
    app.pnp_adaptive_gate_factor_row = ttk.Frame(sec)
    app.pnp_adaptive_gate_factor_row.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
    ttk.Label(app.pnp_adaptive_gate_factor_row, text="pnp_adaptive_gate_factor:").grid(row=0, column=0, sticky=tk.W, pady=1)
    app.pnp_adaptive_gate_factor_var = tk.StringVar(value=str(getattr(defaults, "pnp_adaptive_gate_factor", 0.2)))
    app.pnp_adaptive_gate_factor_entry = ttk.Entry(app.pnp_adaptive_gate_factor_row, textvariable=app.pnp_adaptive_gate_factor_var, width=8)
    app.pnp_adaptive_gate_factor_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    on_pnp_choice_change(app)  # set initial enabled/disabled state

    # --- BA ---
    sec = make_section(main_frame, "Bundle Adjustment")
    sec.pack(fill=tk.X, pady=2)
    row = 0
    ttk.Label(sec, text="BA (method or off):").grid(row=row, column=0, sticky=tk.W, pady=1)
    ba_choices = ["no_ba", "lm", "gradient_descent", "newton"]
    ba_default = "lm" if defaults.use_ba else "no_ba"
    ba_default_idx = ba_choices.index(ba_default) if ba_default in ba_choices else 1
    if defaults.use_ba and defaults.ba_method in ba_choices:
        ba_default_idx = ba_choices.index(defaults.ba_method)
    ba_cb, app.ba_var = make_readonly_combobox(sec, ba_choices, default_index=ba_default_idx, width=16)
    ba_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    ba_cb.bind("<<ComboboxSelected>>", lambda e: on_ba_choice_change(app))
    row += 1

    # BA mode selector: sliding-window vs keyframe-local covisibility BA.
    ttk.Label(sec, text="ba_mode:").grid(row=row, column=0, sticky=tk.W, pady=1)
    ba_mode_choices = ["sliding_window", "keyframe_local_covisibility"]
    ba_mode_default = getattr(defaults, "ba_mode", "sliding_window")
    ba_mode_default_idx = ba_mode_choices.index(ba_mode_default) if ba_mode_default in ba_mode_choices else 0
    app.ba_mode_cb, app.ba_mode_var = make_readonly_combobox(
        sec, ba_mode_choices, default_index=ba_mode_default_idx, width=24
    )
    app.ba_mode_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    app.ba_mode_cb.bind("<<ComboboxSelected>>", lambda e: on_ba_mode_change(app))
    row += 1

    # Sliding-window BA controls (hidden when ba_mode == keyframe_local_covisibility)
    app.ba_interval_label = ttk.Label(sec, text="ba_interval:")
    app.ba_interval_label.grid(row=row, column=0, sticky=tk.W, pady=1)
    app.ba_interval_var = tk.StringVar(value=str(defaults.ba_interval))
    app.ba_interval_entry = ttk.Entry(sec, textvariable=app.ba_interval_var, width=8)
    app.ba_interval_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1

    app.ba_window_size_label = ttk.Label(sec, text="ba_window_size:")
    app.ba_window_size_label.grid(row=row, column=0, sticky=tk.W, pady=1)
    app.ba_window_size_var = tk.StringVar(value=str(defaults.ba_window_size))
    app.ba_window_size_entry = ttk.Entry(sec, textvariable=app.ba_window_size_var, width=8)
    app.ba_window_size_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1

    # Keyframe-local covisibility BA controls (hidden when sliding-window)
    app.keyframe_local_topk_label = ttk.Label(sec, text="keyframe_local_topk:")
    app.keyframe_local_topk_label.grid(row=row, column=0, sticky=tk.W, pady=1)
    app.keyframe_local_topk_var = tk.StringVar(value=str(getattr(defaults, "keyframe_local_topk", 5)))
    app.keyframe_local_topk_entry = ttk.Entry(sec, textvariable=app.keyframe_local_topk_var, width=8)
    app.keyframe_local_topk_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1

    app.keyframe_local_min_shared_points_label = ttk.Label(sec, text="keyframe_local_min_shared_points:")
    app.keyframe_local_min_shared_points_label.grid(row=row, column=0, sticky=tk.W, pady=1)
    app.keyframe_local_min_shared_points_var = tk.StringVar(
        value=str(getattr(defaults, "keyframe_local_min_shared_points", 20))
    )
    app.keyframe_local_min_shared_points_entry = ttk.Entry(sec, textvariable=app.keyframe_local_min_shared_points_var, width=8)
    app.keyframe_local_min_shared_points_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1

    # Apply initial visibility based on ba_mode defaults.
    on_ba_mode_change(app)
    ttk.Label(sec, text="ba_max_points:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.ba_max_points_var = tk.StringVar(value=str(defaults.ba_max_points))
    app.ba_max_points_entry = ttk.Entry(sec, textvariable=app.ba_max_points_var, width=8)
    app.ba_max_points_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="ba_max_allowed_step_factor:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.ba_max_allowed_step_factor_var = tk.StringVar(value=str(getattr(defaults, "ba_max_allowed_step_factor", 3.0)))
    app.ba_max_allowed_step_factor_entry = ttk.Entry(sec, textvariable=app.ba_max_allowed_step_factor_var, width=8)
    app.ba_max_allowed_step_factor_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="ba_bad_reproj_ratio:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.ba_bad_reproj_ratio_var = tk.StringVar(value=str(getattr(defaults, "ba_bad_reproj_ratio", 1.2)))
    app.ba_bad_reproj_ratio_entry = ttk.Entry(sec, textvariable=app.ba_bad_reproj_ratio_var, width=8)
    app.ba_bad_reproj_ratio_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="BA Jacobian:").grid(row=row, column=0, sticky=tk.W, pady=1)
    jac_choices = ["analytical", "numerical"]
    jac_default = "analytical" if getattr(defaults, "ba_use_analytical_jacobian", True) else "numerical"
    app.ba_jacobian_cb, app.ba_jacobian_var = make_readonly_combobox(sec, jac_choices, default_index=jac_choices.index(jac_default), width=12)
    app.ba_jacobian_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1

    # BA-related map-point gating (only meaningful when BA is enabled)
    ttk.Label(sec, text="BA min point observations:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.ba_min_point_observations_var = tk.StringVar(value=str(getattr(defaults, "ba_min_point_observations", 3)))
    app.ba_min_point_observations_entry = ttk.Entry(sec, textvariable=app.ba_min_point_observations_var, width=8)
    app.ba_min_point_observations_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    app.ba_point_sanity_check_var = tk.BooleanVar(value=getattr(defaults, "ba_point_sanity_check", True))
    app.ba_point_sanity_check_cb = ttk.Checkbutton(sec, text="BA point sanity check (finite coords)", variable=app.ba_point_sanity_check_var)
    app.ba_point_sanity_check_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
    row += 1
    app.prune_high_reproj_var = tk.BooleanVar(value=getattr(defaults, "prune_high_reproj", False))
    app.prune_high_reproj_cb = ttk.Checkbutton(sec, text="Prune high reprojection error (policy B)", variable=app.prune_high_reproj_var)
    app.prune_high_reproj_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)

    on_ba_choice_change(app)  # set initial enabled/disabled state

    # --- Map points (BA + pruning) ---
    sec = make_section(main_frame, "Map points (BA + pruning)")
    sec.pack(fill=tk.X, pady=2)
    prow = 0
    ttk.Label(sec, text="Prune interval (frames; 0=disable):").grid(row=prow, column=0, sticky=tk.W, pady=1)
    app.prune_interval_var = tk.StringVar(value=str(getattr(defaults, "prune_interval", 30)))
    ttk.Entry(sec, textvariable=app.prune_interval_var, width=8).grid(row=prow, column=1, sticky=tk.W, padx=5, pady=1)
    prow += 1

    # Stage 04-1a: policy A (low-observation age prune) retired; the local mapper culls by
    # keyframe observation count (--lm_cull_after_keyframes / --lm_min_keyframe_obs).
    ttk.Label(sec, text="Landmark culling: local mapper (keyframe observation count)").grid(
        row=prow, column=0, columnspan=2, sticky=tk.W, pady=1
    )
    prow += 1
    app.prune_numeric_corruption_var = tk.BooleanVar(value=getattr(defaults, "prune_numeric_corruption", False))
    ttk.Checkbutton(sec, text="Prune numeric corruption (policy D)", variable=app.prune_numeric_corruption_var).grid(
        row=prow, column=0, columnspan=2, sticky=tk.W, pady=1
    )
    prow += 1

    app.prune_outlier_hits_var = tk.BooleanVar(value=getattr(defaults, "prune_outlier_hits", False))
    ttk.Checkbutton(sec, text="Prune by outlier hits + last-seen (policy E)", variable=app.prune_outlier_hits_var).grid(
        row=prow, column=0, columnspan=2, sticky=tk.W, pady=1
    )
    prow += 1

    # Nested options for policy E (hidden unless enabled)
    app.prune_outlier_opts_frame = ttk.Frame(sec)
    app.prune_outlier_opts_frame.grid(row=prow, column=0, columnspan=2, sticky=tk.W, padx=(20, 0))
    orow = 0
    ttk.Label(app.prune_outlier_opts_frame, text="Outlier hit count:").grid(row=orow, column=0, sticky=tk.W, pady=1)
    app.prune_outlier_hit_count_var = tk.StringVar(value=str(getattr(defaults, "prune_outlier_hit_count", 3)))
    ttk.Entry(app.prune_outlier_opts_frame, textvariable=app.prune_outlier_hit_count_var, width=8).grid(row=orow, column=1, sticky=tk.W, padx=5, pady=1)
    orow += 1
    ttk.Label(app.prune_outlier_opts_frame, text="Last-seen age (frame ids):").grid(row=orow, column=0, sticky=tk.W, pady=1)
    app.prune_outlier_last_seen_age_var = tk.StringVar(value=str(getattr(defaults, "prune_outlier_last_seen_age", 20)))
    ttk.Entry(app.prune_outlier_opts_frame, textvariable=app.prune_outlier_last_seen_age_var, width=8).grid(row=orow, column=1, sticky=tk.W, padx=5, pady=1)
    prow += 1

    def on_prune_outlier_hits_change(*args):
        if app.prune_outlier_hits_var.get():
            app.prune_outlier_opts_frame.grid()
        else:
            app.prune_outlier_opts_frame.grid_remove()

    app.prune_outlier_hits_var.trace_add("write", lambda *a: on_prune_outlier_hits_change())
    on_prune_outlier_hits_change()

    # --- Homography / default_k / relative scale ---
    sec = make_section(main_frame, "Options")
    sec.pack(fill=tk.X, pady=2)
    app.use_homography_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(sec, text="use_homography", variable=app.use_homography_var).pack(anchor=tk.W)
    app.use_relative_scale_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        sec, text="use_relative_scale (estimate translation scale from map)",
        variable=app.use_relative_scale_var
    ).pack(anchor=tk.W)
    app.default_k_var = tk.BooleanVar(value=defaults.default_k)
    ttk.Checkbutton(sec, text="default_k", variable=app.default_k_var).pack(anchor=tk.W)
    ttk.Label(sec, text="local_map_window:").pack(anchor=tk.W)
    app.local_map_window_var = tk.StringVar(value="0")
    ttk.Entry(sec, textvariable=app.local_map_window_var, width=8).pack(anchor=tk.W, fill=tk.X)
    # Stage 04-1a: landmarks come from the keyframe-pair source (local_mapping.py) with its
    # --lm_* defaults; the per-frame triangulation controls were retired with their flags.
    ttk.Label(sec, text="Point source: keyframe_pair (local mapper defaults)").pack(anchor=tk.W)

    # --- Keyframes ---
    sec = make_section(main_frame, "Keyframes")
    sec.pack(fill=tk.X, pady=2)
    row = 0
    loop_closure_choices = ["Pose graph optimization", "Linear interpolation", "No loop closure"]
    if not getattr(defaults, "use_loop_closure", True):
        lc_default_idx = 2
    elif getattr(defaults, "loop_closure_strategy", "pose_graph") == "linear":
        lc_default_idx = 1
    else:
        lc_default_idx = 0
    ttk.Label(sec, text="Loop closure:").grid(row=row, column=0, sticky=tk.W, pady=1)
    lc_cb, app.loop_closure_var = make_readonly_combobox(
        sec, loop_closure_choices, default_index=lc_default_idx, width=22
    )
    lc_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    app.use_relocalization_var = tk.BooleanVar(value=getattr(defaults, "use_relocalization", True))
    ttk.Checkbutton(sec, text="Relocalization on VO fail", variable=app.use_relocalization_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
    row += 1
    # Relocalization options (shown only when relocalization is enabled)
    app.relocalization_options_frame = ttk.Frame(sec)
    app.relocalization_options_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(20, 0))
    rrow = 0
    ttk.Label(app.relocalization_options_frame, text="relocalization_match_threshold:").grid(row=rrow, column=0, sticky=tk.W, pady=1)
    app.relocalization_match_threshold_var = tk.StringVar(value=str(getattr(defaults, "relocalization_match_threshold", 25)))
    ttk.Entry(app.relocalization_options_frame, textvariable=app.relocalization_match_threshold_var, width=8).grid(row=rrow, column=1, sticky=tk.W, padx=5, pady=1)
    rrow += 1
    ttk.Label(app.relocalization_options_frame, text="relocalization_min_2d3d_pairs:").grid(row=rrow, column=0, sticky=tk.W, pady=1)
    app.relocalization_min_2d3d_pairs_var = tk.StringVar(value=str(getattr(defaults, "relocalization_min_2d3d_pairs", 10)))
    ttk.Entry(app.relocalization_options_frame, textvariable=app.relocalization_min_2d3d_pairs_var, width=8).grid(row=rrow, column=1, sticky=tk.W, padx=5, pady=1)
    rrow += 1
    ttk.Label(app.relocalization_options_frame, text="relocalization_min_pnp_inliers:").grid(row=rrow, column=0, sticky=tk.W, pady=1)
    app.relocalization_min_pnp_inliers_var = tk.StringVar(value=str(getattr(defaults, "relocalization_min_pnp_inliers", 10)))
    ttk.Entry(app.relocalization_options_frame, textvariable=app.relocalization_min_pnp_inliers_var, width=8).grid(row=rrow, column=1, sticky=tk.W, padx=5, pady=1)
    rrow += 1
    ttk.Label(app.relocalization_options_frame, text="relocalization_ratio_threshold:").grid(row=rrow, column=0, sticky=tk.W, pady=1)
    app.relocalization_ratio_threshold_var = tk.StringVar(value=str(getattr(defaults, "relocalization_ratio_threshold", 0.85)))
    ttk.Entry(app.relocalization_options_frame, textvariable=app.relocalization_ratio_threshold_var, width=8).grid(row=rrow, column=1, sticky=tk.W, padx=5, pady=1)
    rrow += 1
    app.relocalization_sanity_check_var = tk.BooleanVar(value=getattr(defaults, "relocalization_sanity_check_enabled", getattr(defaults, "relocalization_sanity_check", False)))
    ttk.Checkbutton(
        app.relocalization_options_frame,
        text="Enable sanity check (reject if pose too far from keyframe)",
        variable=app.relocalization_sanity_check_var,
    ).grid(row=rrow, column=0, columnspan=2, sticky=tk.W, pady=1)
    rrow += 1
    app.relocalization_sanity_check_distance_frame = ttk.Frame(app.relocalization_options_frame)
    app.relocalization_sanity_check_distance_frame.grid(row=rrow, column=0, columnspan=2, sticky=tk.W, padx=(20, 0))
    ttk.Label(app.relocalization_sanity_check_distance_frame, text="relocalization_max_distance_from_keyframe:").grid(row=0, column=0, sticky=tk.W, pady=1)
    app.relocalization_max_distance_from_keyframe_var = tk.StringVar(value=str(getattr(defaults, "relocalization_max_distance_from_keyframe", 2.0)))
    ttk.Entry(app.relocalization_sanity_check_distance_frame, textvariable=app.relocalization_max_distance_from_keyframe_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=5, pady=1)
    rrow += 1
    row += 1

    def on_sanity_check_change(*args):
        if app.relocalization_sanity_check_var.get():
            app.relocalization_sanity_check_distance_frame.grid()
        else:
            app.relocalization_sanity_check_distance_frame.grid_remove()

    app.relocalization_sanity_check_var.trace_add("write", lambda *a: on_sanity_check_change())
    on_sanity_check_change()

    def on_relocalization_change(*args):
        if app.use_relocalization_var.get():
            app.relocalization_options_frame.grid()
        else:
            app.relocalization_options_frame.grid_remove()

    app.use_relocalization_var.trace_add("write", lambda *a: on_relocalization_change())
    on_relocalization_change()  # set initial visibility

    ttk.Label(sec, text="loop_detection_strategy:").grid(row=row, column=0, sticky=tk.W, pady=1)
    ld_choices = ["spaced", "bow"]
    ld_default_idx = ld_choices.index(getattr(defaults, "loop_detection_strategy", "spaced"))
    ld_cb, app.loop_detection_strategy_var = make_readonly_combobox(sec, ld_choices, default_index=ld_default_idx, width=12)
    ld_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    ld_cb.bind("<<ComboboxSelected>>", lambda e: on_loop_detection_strategy_change(app))
    row += 1
    ttk.Label(sec, text="vocabulary_path:").grid(row=row, column=0, sticky=tk.W, pady=1)
    # Default to a known working config unless user already set a path.
    bow_default_vocab = str((Path(__file__).parent / ".deps/loop-closure-toolbox/config/sthereo_01_rgb_4_3.yaml").resolve())
    vocab_default = getattr(defaults, "vocabulary_path", "") or ""
    if not str(vocab_default).strip():
        vocab_default = bow_default_vocab
    app.vocabulary_path_var = tk.StringVar(value=vocab_default)
    ttk.Entry(sec, textvariable=app.vocabulary_path_var, width=30).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="num_spaced_candidates:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.num_spaced_candidates_var = tk.StringVar(value=str(getattr(defaults, "num_spaced_candidates", 5)))
    ttk.Entry(sec, textvariable=app.num_spaced_candidates_var, width=8).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="min_gap:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.min_gap_var = tk.StringVar(value=str(getattr(defaults, "min_gap", 100)))
    ttk.Entry(sec, textvariable=app.min_gap_var, width=8).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="keyframe_strategy:").grid(row=row, column=0, sticky=tk.W, pady=1)
    kf_choices = ["fixed", "ratio", "hybrid"]
    kf_cb, app.keyframe_strategy_var = make_readonly_combobox(sec, kf_choices, default_index=kf_choices.index(defaults.keyframe_strategy), width=12)
    kf_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="keyframe_interval:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.keyframe_interval_var = tk.StringVar(value=str(defaults.keyframe_interval))
    ttk.Entry(sec, textvariable=app.keyframe_interval_var, width=8).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="keyframe_min_spacing:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.keyframe_min_spacing_var = tk.StringVar(value=str(defaults.keyframe_min_spacing))
    ttk.Entry(sec, textvariable=app.keyframe_min_spacing_var, width=8).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="keyframe_max_spacing:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.keyframe_max_spacing_var = tk.StringVar(value=str(defaults.keyframe_max_spacing))
    ttk.Entry(sec, textvariable=app.keyframe_max_spacing_var, width=8).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="keyframe_ratio_threshold:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.keyframe_ratio_threshold_var = tk.StringVar(value=str(defaults.keyframe_ratio_threshold))
    ttk.Entry(sec, textvariable=app.keyframe_ratio_threshold_var, width=8).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)

    # --- Output / Log ---
    sec = make_section(main_frame, "Output / Log")
    sec.pack(fill=tk.X, pady=2)
    row = 0
    ttk.Label(sec, text="log_file:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.log_file_var = tk.StringVar(value=defaults.log_file or "")
    ttk.Entry(sec, textvariable=app.log_file_var, width=30).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="save_estimated:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.save_estimated_var = tk.StringVar(value=defaults.save_estimated or "e.txt")
    ttk.Entry(sec, textvariable=app.save_estimated_var, width=30).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="timestamp_file:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.timestamp_file_var = tk.StringVar(value=defaults.timestamp_file or "rgb.txt")
    ttk.Entry(sec, textvariable=app.timestamp_file_var, width=30).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    ttk.Label(sec, text="ground_truth:").grid(row=row, column=0, sticky=tk.W, pady=1)
    app.ground_truth_var = tk.StringVar(value=defaults.ground_truth or "groundtruth.txt")
    ttk.Entry(sec, textvariable=app.ground_truth_var, width=30).grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    row += 1
    # Run playback error viewer after pipeline (only enable when both estimate and ground truth are set)
    app.run_playback_after_var = tk.BooleanVar(value=False)
    app.run_playback_after_cb = ttk.Checkbutton(
        sec, text="Run playback error viewer after pipeline", variable=app.run_playback_after_var
    )
    app.run_playback_after_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)

    def update_run_playback_after_state(*args):
        est = (app.save_estimated_var.get() or "").strip()
        gt = (app.ground_truth_var.get() or "").strip()
        if est and gt:
            app.run_playback_after_cb.config(state="normal")
        else:
            app.run_playback_after_cb.config(state="disabled")
            app.run_playback_after_var.set(False)

    app.save_estimated_var.trace_add("write", lambda *a: update_run_playback_after_state())
    app.ground_truth_var.trace_add("write", lambda *a: update_run_playback_after_state())
    update_run_playback_after_state()

    # --- Display ---
    sec = make_section(main_frame, "Display")
    sec.pack(fill=tk.X, pady=2)
    row = 0
    ttk.Label(sec, text="Display:").grid(row=row, column=0, sticky=tk.W, pady=1)
    display_choices = ["Display", "No display"]
    display_default_idx = 0 if not defaults.no_display else 1
    display_cb, app.display_var = make_readonly_combobox(sec, display_choices, default_index=display_default_idx, width=14)
    display_cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=1)
    display_cb.bind("<<ComboboxSelected>>", lambda e: on_display_change(app))
    row += 1

    # Display-only options (shown when Display selected)
    app.display_options_frame = ttk.Frame(sec)
    app.display_options_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
    sub = app.display_options_frame
    r = 0
    ttk.Label(sub, text="camera_spacing:").grid(row=r, column=0, sticky=tk.W, pady=1)
    app.camera_spacing_var = tk.StringVar(value=str(defaults.camera_spacing) if defaults.camera_spacing is not None else "")
    ttk.Entry(sub, textvariable=app.camera_spacing_var, width=8).grid(row=r, column=1, sticky=tk.W, padx=5, pady=1)
    r += 1
    app.show_grid_var = tk.BooleanVar(value=defaults.show_grid)
    ttk.Checkbutton(sub, text="show_grid", variable=app.show_grid_var).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=1)
    r += 1
    ttk.Label(sub, text="viewer_fps:").grid(row=r, column=0, sticky=tk.W, pady=1)
    app.viewer_fps_var = tk.StringVar(value=str(defaults.viewer_fps))
    ttk.Entry(sub, textvariable=app.viewer_fps_var, width=8).grid(row=r, column=1, sticky=tk.W, padx=5, pady=1)
    r += 1
    ttk.Label(sub, text="tracker_delay_ms:").grid(row=r, column=0, sticky=tk.W, pady=1)
    app.tracker_delay_ms_var = tk.StringVar(value=str(getattr(defaults, 'tracker_delay_ms', 0)))
    ttk.Entry(sub, textvariable=app.tracker_delay_ms_var, width=8).grid(row=r, column=1, sticky=tk.W, padx=5, pady=1)
    if defaults.no_display:
        app.display_options_frame.grid_remove()
    else:
        app.display_options_frame.grid()

    # --- Run / Stop ---
    app.root = root
    app.pipeline_thread = None
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(pady=10)
    app.run_btn = ttk.Button(btn_frame, text="Run", command=lambda: run_from_ui(app))
    app.run_btn.pack(side=tk.LEFT, padx=(0, 5))
    app.stop_btn = ttk.Button(btn_frame, text="Stop", command=request_stop, state="disabled")
    app.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

    def launch_playback_ui():
        playback_ui = Path(__file__).resolve().parent / "run_playback_ui.py"
        if playback_ui.is_file():
            try:
                subprocess.Popen([sys.executable, str(playback_ui)], cwd=str(playback_ui.parent))
            except Exception as e:
                messagebox.showerror("Playback viewer", f"Failed to start: {e}")
        else:
            messagebox.showwarning("Playback viewer", f"Not found: {playback_ui}")

    ttk.Button(btn_frame, text="Playback viewer (GT vs estimate)...", command=launch_playback_ui).pack(side=tk.LEFT, padx=5)

    def on_closing():
        """On window close: signal pipeline to stop and wait for thread before exiting."""
        pipeline_module._stop_requested = True
        if app.pipeline_thread is not None and app.pipeline_thread.is_alive():
            app.pipeline_thread.join(timeout=5.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    # Avoid GIL/tstate crash when Pangolin display was used (exit without full interpreter shutdown)
    os._exit(0)


if __name__ == "__main__":
    main()
