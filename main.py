

import argparse
import os
import sys
from pathlib import Path


import camera
import epipolar_refine
import front_end
from dataset import iter_frames, load_timestamps_for_images
from frame import DEFAULT_K
from depth_filter import DepthFilterParams
from depth_filter_arm import DepthFilterArmParams
from local_mapping import LocalMappingParams
import map_rotation
import map_tracker
from map_tracker import MapTracker, load_m1_map
import opening
from pipeline import show_trajectory
import rotation_only


def _multiscale_preset_params(preset: str):
    """
    Return (orb_params, sift_params) dictionaries for the given preset.

    - balanced: OpenCV defaults (documented defaults for ORB/SIFT)
    - fast: fewer levels/features for speed
    - aggressive: more levels/features for scale robustness
    - none: disable pyramid where possible (ORB nlevels=1; SIFT uses minimal scale-space)
    """
    preset = (preset or "balanced").lower().strip()
    if preset not in ("none", "fast", "balanced", "aggressive"):
        preset = "balanced"

    orb_balanced = {"nfeatures": 500, "scaleFactor": 1.2, "nlevels": 8}
    sift_balanced = {
        "nfeatures": 0,
        "nOctaveLayers": 3,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10.0,
        "sigma": 1.6,
    }

    if preset == "balanced":
        return orb_balanced, sift_balanced
    if preset == "none":
        # ORB: true single-level pyramid.
        # SIFT: OpenCV SIFT is intrinsically scale-space based; this is a minimal setting.
        return (
            {"nfeatures": 500, "scaleFactor": 1.2, "nlevels": 1},
            {
                "nfeatures": 0,
                "nOctaveLayers": 1,
                "contrastThreshold": 0.04,
                "edgeThreshold": 10.0,
                "sigma": 1.6,
            },
        )
    if preset == "fast":
        return (
            {"nfeatures": 500, "scaleFactor": 1.3, "nlevels": 6},
            {
                "nfeatures": 800,
                "nOctaveLayers": 3,
                "contrastThreshold": 0.06,
                "edgeThreshold": 10.0,
                "sigma": 1.6,
            },
        )
    # aggressive
    return (
        {"nfeatures": 2000, "scaleFactor": 1.15, "nlevels": 12},
        {
            "nfeatures": 0,
            "nOctaveLayers": 5,
            "contrastThreshold": 0.03,
            "edgeThreshold": 10.0,
            "sigma": 1.6,
        },
    )


def _resolve_detector_config(args):
    preset = getattr(args, "multiscale_preset", "balanced")
    manual = bool(getattr(args, "multiscale_manual", False))
    orb_preset, sift_preset = _multiscale_preset_params(preset)

    if not manual:
        orb = orb_preset
        sift = sift_preset
    else:
        # Manual mode: use explicit args. Defaults are the Balanced values.
        orb = {
            "nfeatures": int(getattr(args, "orb_nfeatures", 500)),
            "scaleFactor": float(getattr(args, "orb_scale_factor", 1.2)),
            "nlevels": int(getattr(args, "orb_nlevels", 8)),
        }
        sift = {
            "nfeatures": int(getattr(args, "sift_nfeatures", 0)),
            "nOctaveLayers": int(getattr(args, "sift_n_octave_layers", 3)),
            "contrastThreshold": float(getattr(args, "sift_contrast_threshold", 0.04)),
            "edgeThreshold": float(getattr(args, "sift_edge_threshold", 10.0)),
            "sigma": float(getattr(args, "sift_sigma", 1.6)),
        }

    return {
        "multiscale_preset": preset,
        "multiscale_manual": manual,
        "orb": orb,
        "sift": sift,
        "debug_features": bool(getattr(args, "debug_features", False)),
        "debug_features_every": int(getattr(args, "debug_features_every", 30)),
    }



def main():
    parser = get_parser()
    args = parser.parse_args(front_end.expand_argv(sys.argv[1:]))
    run_pipeline(args)
    # Exit immediately when display was used to avoid GIL/tstate crash on window close
    if not args.no_display:
        os._exit(0)


def write_resolved_settings(args, out_path):
    """Write every non-private argparse attribute of `args`, plus library
    versions and the invocation string, to `out_path` as JSON (Stage 02
    change B). Overwrites any existing file at out_path."""
    import json
    import numpy
    import cv2
    import scipy

    settings = {}
    for key, value in vars(args).items():
        if key.startswith('_'):
            continue
        if isinstance(value, Path):
            value = str(value)
        try:
            json.dumps(value)
        except TypeError:
            value = str(value)
        settings[key] = value

    settings["versions"] = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "cv2": cv2.__version__,
        "scipy": scipy.__version__,
    }
    settings["invocation"] = getattr(args, '_invocation', None) or " ".join(sys.argv)

    out_path = Path(out_path)
    with open(out_path, "w") as f:
        json.dump(settings, f, indent=2, sort_keys=True)
    print(f"  >> Wrote resolved settings to {out_path}")


def validate_tracking_args(args):
    """Stage 04l (spec A): refuse flag combinations under which --tracking map would not run what
    resolved_settings.json says. Under a perfect map nothing may edit the map or move a pose after
    the tracker has set it. Raises SystemExit naming the offending flag; returns None otherwise
    (every combination that does not use the three new flags included)."""
    # Stage 04-m0 (spec A2): the integrated system M0 has its own rules, checked before every other.
    if getattr(args, "live_map", "off") == "on":
        return _validate_live_map_args(args)
    tracking_mode = getattr(args, "tracking", "chain")
    m1_file = getattr(args, "m1_map", None)
    # Stage 04v (spec B2): the rotation-only state inside a map, checked before every other rule.
    # Stage 04x (spec B2): the same rules for every value other than off, the value in the message.
    rot_state = getattr(args, "map_rot_state", "off")
    if rot_state != "off":
        if tracking_mode != "map" or getattr(args, "map_reloc", "none") != "pnp":
            raise SystemExit(f"--map_rot_state {rot_state} is only valid with --tracking map and --map_reloc pnp")
        if getattr(args, "reloc_shadow", False):
            raise SystemExit(f"--map_rot_state {rot_state} refuses --reloc_shadow")
        if getattr(args, "reloc_solver", "epnp") != "epnp":
            raise SystemExit(f"--map_rot_state {rot_state} refuses --reloc_solver other than epnp")
    # Stage 04u (spec B3): --init twoview only under --tracking rotation, --init_refit only with it;
    # under --tracking rotation the 04r rules below apply whatever --init says.
    init_mode = getattr(args, "init", "none")
    init_refit = getattr(args, "init_refit", "off")
    if init_mode == "twoview" and tracking_mode != "rotation":
        raise SystemExit("--init twoview is only valid with --tracking rotation")
    if init_refit != "off" and init_mode != "twoview":
        raise SystemExit(f"--init_refit {init_refit} is only valid with --init twoview")
    if tracking_mode == "rotation":
        return _validate_rotation_args(args)
    if tracking_mode != "map":
        if m1_file is not None:
            raise SystemExit("--m1_map is only valid with --tracking map")
        if getattr(args, "map_reloc", "none") == "pnp":
            raise SystemExit("--map_reloc pnp is only valid with --tracking map")
        _validate_reloc_solver_args(args)
        return None
    if m1_file is None:
        raise SystemExit("--tracking map requires --m1_map (only the perfect-map rungs are built)")
    if not getattr(args, "default_k", False):
        raise SystemExit("--tracking map requires --default_k")
    if not getattr(args, "undistort", False):
        raise SystemExit("--tracking map requires --undistort (ruling G3)")
    if int(getattr(args, "pnp_interval", 10)) != 0:
        raise SystemExit("--tracking map requires --pnp_interval 0")
    for dest, flag in (("use_loop_closure", "--no_loop_closure"), ("use_relocalization", "--no_relocalization"),
                       ("use_ba", "--no_ba"), ("use_point_fusion", "--no_point_fusion")):
        if getattr(args, dest, True):
            raise SystemExit(f"--tracking map requires {flag}")
    for dest, flag in (("epi_refine", "--epi_refine"), ("lm_confirm", "--lm_confirm")):
        if getattr(args, dest, False):
            raise SystemExit(f"--tracking map refuses {flag}")
    _validate_reloc_solver_args(args)
    return None


def _validate_rotation_args(args):
    """Stage 04r (spec B): the flags --tracking rotation requires and refuses. Nothing may edit a
    map or move a pose after the mode has set it, and no ground truth enters the run. Raises
    SystemExit naming the flag; returns None otherwise. The front end is not constrained."""
    if not getattr(args, "default_k", False):
        raise SystemExit("--tracking rotation requires --default_k")
    if not getattr(args, "undistort", False):
        raise SystemExit("--tracking rotation requires --undistort")
    if int(getattr(args, "pnp_interval", 10)) != 0:
        raise SystemExit("--tracking rotation requires --pnp_interval 0")
    for dest, flag in (("use_loop_closure", "--no_loop_closure"), ("use_relocalization", "--no_relocalization"),
                       ("use_ba", "--no_ba"), ("use_point_fusion", "--no_point_fusion")):
        if getattr(args, dest, True):
            raise SystemExit(f"--tracking rotation requires {flag}")
    if getattr(args, "m1_map", None) is not None:
        raise SystemExit("--tracking rotation refuses --m1_map")
    if getattr(args, "map_reloc", "none") == "pnp":
        raise SystemExit("--tracking rotation refuses --map_reloc pnp")
    if getattr(args, "reloc_solver", "epnp") != "epnp":
        raise SystemExit("--tracking rotation refuses --reloc_solver other than epnp")
    if getattr(args, "reloc_shadow", False):
        raise SystemExit("--tracking rotation refuses --reloc_shadow")
    for dest, flag in (("epi_refine", "--epi_refine"), ("lm_confirm", "--lm_confirm")):
        if getattr(args, dest, False):
            raise SystemExit(f"--tracking rotation refuses {flag}")
    if getattr(args, "oracle", "none") != "none":
        raise SystemExit("--tracking rotation refuses --oracle other than none")
    if getattr(args, "point_source", "keyframe_pair") == "depth_filter":
        raise SystemExit("--tracking rotation refuses --point_source depth_filter")
    return None


def _validate_live_map_args(args):
    """Stage 04-m0 (spec A2): the flags --live_map on requires and refuses, in this order. M0 is the
    04u opening with the r52 fallback (or off) and the age lifetime; no ground truth enters the run.
    Raises SystemExit naming the flag; returns None otherwise (BA, point fusion, either point source,
    --epi_refine, --init_refit and every --ba_* flag are accepted; validate_ba_args still applies)."""
    if getattr(args, "tracking", "chain") != "rotation":
        raise SystemExit("--live_map on requires --tracking rotation")
    if getattr(args, "init", "none") != "twoview":
        raise SystemExit("--live_map on requires --init twoview")
    if not getattr(args, "default_k", False):
        raise SystemExit("--live_map on requires --default_k")
    if not getattr(args, "undistort", False):
        raise SystemExit("--live_map on requires --undistort")
    if int(getattr(args, "pnp_interval", 10)) != 0:
        raise SystemExit("--live_map on requires --pnp_interval 0")
    for dest, flag in (("use_loop_closure", "--no_loop_closure"), ("use_relocalization", "--no_relocalization")):
        if getattr(args, dest, True):
            raise SystemExit(f"--live_map on requires {flag}")
    if getattr(args, "m1_map", None) is not None:
        raise SystemExit("--live_map on refuses --m1_map")
    if getattr(args, "map_reloc", "none") == "pnp":
        raise SystemExit("--live_map on refuses --map_reloc pnp")
    if getattr(args, "reloc_solver", "epnp") != "epnp":
        raise SystemExit("--live_map on refuses --reloc_solver other than epnp")
    if getattr(args, "reloc_shadow", False):
        raise SystemExit("--live_map on refuses --reloc_shadow")
    if getattr(args, "oracle", "none") != "none":
        raise SystemExit("--live_map on refuses --oracle other than none")
    if getattr(args, "lm_confirm", False):
        raise SystemExit("--live_map on refuses --lm_confirm")
    if getattr(args, "df_lifetime", "age") == "r2":
        raise SystemExit("--live_map on refuses --df_lifetime r2")
    if getattr(args, "map_rot_state", "off") == "on":
        raise SystemExit("--live_map on refuses --map_rot_state on")
    return None


def _validate_reloc_solver_args(args):
    """Stage 04q (spec B2): --reloc_solver other than epnp and --reloc_shadow only with --tracking map
    and --map_reloc pnp. Raises SystemExit naming the flag."""
    usable = getattr(args, "tracking", "chain") == "map" and getattr(args, "map_reloc", "none") == "pnp"
    if usable:
        return None
    if getattr(args, "reloc_solver", "epnp") != "epnp":
        raise SystemExit("--reloc_solver other than epnp is only valid with --tracking map and --map_reloc pnp")
    if getattr(args, "reloc_shadow", False):
        raise SystemExit("--reloc_shadow is only valid with --tracking map and --map_reloc pnp")
    return None


def validate_ba_args(args):
    """Stage 04z (spec A2): the legacy jump gate is off wherever the participation rule is on
    (conventions.md#local-ba-participation). Raises SystemExit for --ba_free_cameras supported with
    --ba_jump_gate on; returns None for every other combination."""
    if getattr(args, "ba_free_cameras", "all") == "supported" and getattr(args, "ba_jump_gate", "on") == "on":
        raise SystemExit("--ba_free_cameras supported refuses --ba_jump_gate on")
    return None


def validate_residual_args(args):
    """Stage 04-m1 (spec A2): --residual_test log or on only where a map tracker accepts poses (--tracking
    map or --live_map on), and never on the 04v path (--map_rot_state on). Raises SystemExit naming the
    flag; returns None otherwise (every combination with --residual_test off included)."""
    v = getattr(args, "residual_test", "off")
    if v == "off":
        return None
    if not (getattr(args, "live_map", "off") == "on" or getattr(args, "tracking", "chain") == "map"):
        raise SystemExit(f"--residual_test {v} is only valid with --tracking map or --live_map on")
    if getattr(args, "map_rot_state", "off") == "on":
        raise SystemExit(f"--residual_test {v} refuses --map_rot_state on")
    return None


def validate_gap_args(args):
    """Stage 04-r04 (spec A2, ruling R65-4): --map_rot_gap lost acts on the r52 fallback only. Raises
    SystemExit naming the flag; returns None otherwise (every combination with --map_rot_gap carry included)."""
    if getattr(args, "map_rot_gap", "carry") != "lost":
        return None
    if getattr(args, "map_rot_state", "off") != "r52":
        raise SystemExit("--map_rot_gap lost requires --map_rot_state r52")
    return None


def run_pipeline(args):
    """Run the VO pipeline with the given args (argparse.Namespace or object with same attributes)."""
    validate_tracking_args(args)
    validate_ba_args(args)
    validate_residual_args(args)
    validate_gap_args(args)
    base = Path(__file__).parent
    method = args.method
    ''' in case default camera intrinsics are given'''
    K = DEFAULT_K.copy() if args.default_k else None
    image_folder = base / args.dataset
    image_ext = getattr(args, 'image_ext', '.png')
    detector_config = _resolve_detector_config(args)
    if getattr(args, 'undistort', False) and not args.default_k:
        raise SystemExit(
            "--undistort requires --default_k: the fr2 coefficients are calibrated against DEFAULT_K"
        )
    distortion = camera.DISTORTION_FR2 if getattr(args, 'undistort', False) else None
    # Stage 04l (spec E): the perfect-map tracker, built and header-checked before any frame.
    m1_tracker = None
    if getattr(args, 'tracking', 'chain') == 'map':
        try:
            m1_map = load_m1_map(args.m1_map)
        except (OSError, ValueError) as e:
            raise SystemExit(f"--m1_map: {e}")
        # Stage 04v (spec B3): the rotation-only state behind --map_rot_state (off: the 04m tracker)
        # Stage 04x (spec B3): r52 is the same class with variant "r52"
        if getattr(args, 'map_rot_state', 'off') in ('on', 'r52'):
            tracker_cls = map_rotation.MapRotationTracker
        else:
            tracker_cls = MapTracker
        m1_tracker = tracker_cls(m1_map, args.m1_mode, None)
        # Stage 04m (spec B): the relocalizing state, off unless --map_reloc pnp
        m1_tracker.reloc = getattr(args, 'map_reloc', 'none')
        m1_tracker.reloc_params = map_tracker.MapRelocParams()
        # Stage 04q (spec B3): the relocalization solver and the shadow mode
        m1_tracker.reloc_solver = args.reloc_solver
        m1_tracker.reloc_shadow = bool(args.reloc_shadow)
        # Stage 04-m1 (spec A3): the residual test of every count acceptance (off, log, on)
        m1_tracker.residual_test = getattr(args, "residual_test", "off")
        # Stage 04x (spec B3 (i)): the fallback structure of rulings R52-1 to R52-3
        if getattr(args, 'map_rot_state', 'off') == 'r52':
            m1_tracker.variant = "r52"
            # Stage 04-r04 (spec A3, R65-4): the post-gap label rule of the r52 fallback
            m1_tracker.gap_rule = getattr(args, "map_rot_gap", "carry")
        if not m1_tracker.check_header(getattr(args, 'front_end', 'orb'), method, args.undistort,
                                       args.keyframe_interval):
            raise SystemExit("--m1_map: the map file's header does not match this run")
    # Stage 04r (spec F): the pre-map rotation-only mode (None under chain and map).
    rot_tracker = None
    # Stage 04u (spec B4): under --init twoview the opening driver replaces rot_tracker.
    open_tracker = None
    if getattr(args, 'tracking', 'chain') == 'rotation':
        if getattr(args, 'init', 'none') == 'twoview':
            open_tracker = opening.OpeningTracker(None, refit=getattr(args, 'init_refit', 'off'))
        else:
            rot_tracker = rotation_only.RotationOnlyTracker()
    K_used, n_total, frame_iter = iter_frames(image_folder, method, K=K, ext=image_ext, detector_config=detector_config, distortion=distortion)

    show_realtime = not args.no_display # if no display is requested, show_realtime is False
    save_estimated_path = Path(args.save_estimated) if args.save_estimated else None
    if save_estimated_path is not None:
        resolved_settings_path = save_estimated_path.parent / "resolved_settings.json"
        write_resolved_settings(args, resolved_settings_path)
    timestamps = None # if no timestamps are given, timestamps is None
    if save_estimated_path is not None:
        # Timestamps: order must match load_frames (image_folder). Optionally use custom rgb.txt path.
        if args.timestamp_file:
            ts_source = (base / args.timestamp_file) if not Path(args.timestamp_file).is_absolute() else Path(args.timestamp_file)
        else:
            ts_source = image_folder
        if ts_source.is_file(): #if no timestamp file is given, ts_source is a directory, and fails this if.
            ts_list = load_timestamps_for_images(image_folder, ext=image_ext, rgb_file=ts_source) #load the timestamps from the timestamp file.
        else:#if no timestamp file is given.
            ts_list = load_timestamps_for_images(ts_source if ts_source.is_dir() else image_folder, ext=image_ext)#load the timestamps from the image folder.
        if ts_list is not None and len(ts_list) >= n_total:#
            timestamps = [t for t, _ in ts_list][: n_total]  #get the timestamps for the frames.
            print(f"  >> Using {len(timestamps)} timestamps for saved trajectory")
        else: #if the timestamps are not found or the number of timestamps is less than the number of frames, print a warning.
            print("  >> No rgb.txt or mismatch; saving trajectory with frame indices as timestamps")
            if args.timestamp_file:
                print("  >> WARNING: --timestamp_file was set but not used (file missing or row count/order mismatch). evo_ape will not align with ground truth.")

    t_norm_log_path = Path(args.t_norm_log) if args.t_norm_log else None
    log_file_path = Path(args.log_file) if args.log_file else None
    log_invocation = getattr(args, '_invocation', None) or " ".join(sys.argv)
    ground_truth_path = Path(args.ground_truth) if args.ground_truth else None
    # Stage 04x (spec B3 (ii)): the r52 fallback reads per-frame times (the history and the velocity window)
    if m1_tracker is not None and getattr(args, 'map_rot_state', 'off') == 'r52':
        m1_tracker.timestamps = timestamps
        if timestamps is None or len(timestamps) != n_total:
            raise SystemExit("--map_rot_state r52 requires per-frame timestamps: give --save_estimated and a usable "
                             "--timestamp_file (rgb.txt)")
    # Stage 04-m0 (spec A3, C3): under --live_map on the opening hands its map to a tracker that tracks
    # the live map: the r52 fallback (with the run's timestamps, checked as 04x does) or the 04m tracker.
    if getattr(args, 'live_map', 'off') == 'on' and open_tracker is not None:
        m0_rot_state = getattr(args, 'map_rot_state', 'off')
        if m0_rot_state == 'r52' and (timestamps is None or len(timestamps) != n_total):
            raise SystemExit("--map_rot_state r52 requires per-frame timestamps: give --save_estimated and a usable "
                             "--timestamp_file (rgb.txt)")

        def _m0_tracker_factory(m1map):
            if m0_rot_state == 'r52':
                tracker = map_rotation.MapRotationTracker(m1map, "m1a", None)
                tracker.variant = "r52"
                tracker.timestamps = timestamps
                # Stage 04-r04 (spec A3, R65-4): the post-gap label rule of the r52 fallback
                tracker.gap_rule = getattr(args, "map_rot_gap", "carry")
            else:
                tracker = MapTracker(m1map, "m1a", None)
            tracker.live_map = True
            # Stage 04-m1 (spec A3): the residual test, on the tracker the opening hands its map to
            tracker.residual_test = getattr(args, "residual_test", "off")
            return tracker

        _m0_tracker_factory.map_rot_state = m0_rot_state
        open_tracker.tracker_factory = _m0_tracker_factory
    oracle = _build_oracle(args, ground_truth_path, timestamps, n_total)
    show_trajectory(frame_iter,
                    n_frames=n_total,
                    camera_spacing=args.camera_spacing,
                    show_grid=args.show_grid,
                    show_traj=True,
                    pnp_interval=args.pnp_interval,
                    pnp_solver=args.pnp_solver,
                    pnp_refinement=args.pnp_refinement,
                    pnp_use_initial_guess=args.pnp_use_initial_guess,
                    pnp_multi_inlier_gate=args.pnp_multi_inlier_gate,
                    ba_interval=args.ba_interval,
                    ba_window_size=args.ba_window_size,
                    ba_max_points=args.ba_max_points,
                    ba_max_allowed_step_factor=getattr(args, 'ba_max_allowed_step_factor', 3.0),
                    ba_bad_reproj_ratio=getattr(args, 'ba_bad_reproj_ratio', 1.2),
                    use_bundle_adjustment=args.use_ba,
                    ba_method=args.ba_method,
                    ba_use_analytical_jacobian=getattr(args, 'ba_use_analytical_jacobian', True),
                    use_homography=args.use_homography,
                    show_realtime=show_realtime,
                    save_estimated_path=save_estimated_path,
                    timestamps=timestamps,
                    t_norm_log_path=t_norm_log_path,
                    local_map_window=args.local_map_window,
                    log_file_path=log_file_path,
                    log_invocation=log_invocation,
                    pnp_min_inlier_ratio=args.pnp_min_inlier_ratio,
                    pnp_adaptive_gate=getattr(args, 'pnp_adaptive_gate', False),
                    pnp_adaptive_gate_factor=getattr(args, 'pnp_adaptive_gate_factor', 0.2),
                    pnp_reprojection_error=getattr(args, 'pnp_reprojection_error', 3.0),
                    use_relative_scale=getattr(args, 'use_relative_scale', False),
                    ground_truth_path=ground_truth_path,
                    viewer_fps=args.viewer_fps,
                    tracker_delay_ms=getattr(args, 'tracker_delay_ms', 0),
                    use_loop_closure=args.use_loop_closure,
                    loop_closure_strategy=args.loop_closure_strategy,
                    loop_detection_strategy=getattr(args, 'loop_detection_strategy', 'spaced'),
                    vocabulary_path=getattr(args, 'vocabulary_path', None),
                    num_spaced_candidates=getattr(args, 'num_spaced_candidates', 5),
                    min_gap=getattr(args, 'min_gap', 100),
                    keyframe_strategy=args.keyframe_strategy,
                    keyframe_interval=args.keyframe_interval,
                    keyframe_min_spacing=args.keyframe_min_spacing,
                    keyframe_max_spacing=args.keyframe_max_spacing,
                    keyframe_ratio_threshold=args.keyframe_ratio_threshold,
                    use_relocalization=getattr(args, 'use_relocalization', True),
                    relocalization_match_threshold=getattr(args, 'relocalization_match_threshold', 25),
                    relocalization_min_2d3d_pairs=getattr(args, 'relocalization_min_2d3d_pairs', 10),
                    relocalization_min_pnp_inliers=getattr(args, 'relocalization_min_pnp_inliers', 10),
                    relocalization_ratio_threshold=getattr(args, 'relocalization_ratio_threshold', 0.85),
                    relocalization_sanity_check_enabled=getattr(args, 'relocalization_sanity_check_enabled', getattr(args, 'relocalization_sanity_check', False)),
                    relocalization_max_distance_from_keyframe=getattr(args, 'relocalization_max_distance_from_keyframe', 2.0),
                    ba_min_point_observations=getattr(args, 'ba_min_point_observations', 3),
                    ba_point_sanity_check=getattr(args, 'ba_point_sanity_check', True),
                    ba_max_fixed_observers=getattr(args, 'ba_max_fixed_observers', 0),
                    ba_huber_delta_px=getattr(args, 'ba_huber_delta_px', 2.4477),
                    ba_min_anchor_points=getattr(args, 'ba_min_anchor_points', 10),
                    ba_max_iterations=getattr(args, 'ba_max_iterations', 10),
                    ba_outlier_reject=getattr(args, 'ba_outlier_reject', 'off'),
                    ba_free_cameras=getattr(args, 'ba_free_cameras', 'all'),
                    ba_jump_gate=getattr(args, 'ba_jump_gate', 'on'),
                    prune_interval=getattr(args, 'prune_interval', 30),
                    prune_high_reproj=getattr(args, 'prune_high_reproj', False),
                    prune_numeric_corruption=getattr(args, 'prune_numeric_corruption', False),
                    prune_outlier_hits=getattr(args, 'prune_outlier_hits', False),
                    prune_outlier_hit_count=getattr(args, 'prune_outlier_hit_count', 3),
                    prune_outlier_last_seen_age=getattr(args, 'prune_outlier_last_seen_age', 20),

                    # BA mode + neighborhood (keyframe-local covisibility BA)
                    ba_mode=getattr(args, 'ba_mode', 'sliding_window'),
                    keyframe_local_topk=getattr(args, 'keyframe_local_topk', 5),
                    keyframe_local_min_shared_points=getattr(args, 'keyframe_local_min_shared_points', 20),
                    local_map_search_radius=getattr(args, 'local_map_search_radius', 25.0),
                    local_map_max_desc_dist=getattr(args, 'local_map_max_desc_dist', None),
                    local_map_max_depth_ratio=getattr(args, 'local_map_max_depth_ratio', 5.0),
                    use_point_fusion=getattr(args, 'use_point_fusion', True),
                    fusion_interval=getattr(args, 'fusion_interval', 30),
                    fusion_max_dist=getattr(args, 'fusion_max_dist', 0.05),
                    fusion_max_desc_dist=getattr(args, 'fusion_max_desc_dist', 40.0),
                    map_metrics_path=getattr(args, 'map_metrics', None),
                    step_lengths_path=getattr(args, 'step_lengths', None),
                    oracle=oracle,
                    # Stage 04-1a: the point-source arm and its constants (--lm_* flags)
                    point_source=getattr(args, 'point_source', 'keyframe_pair'),
                    lm_params=LocalMappingParams.from_args(args),
                    # Stage 04g: the depth-filter arm's constants (--df_* flags; used only with
                    # --point_source depth_filter)
                    df_params=DepthFilterParams.from_args(args),
                    df_arm_params=DepthFilterArmParams.from_args(args),
                    # Stage 04k: the refinement switch (getattr: some tests build the namespace by hand)
                    refine_params=(epipolar_refine.RefineParams() if getattr(args, "epi_refine", False) else None),
                    # Stage 04l: the map-based tracker (None under --tracking chain)
                    m1_tracker=m1_tracker,
                    # Stage 04r: the pre-map rotation-only mode (None under chain and map)
                    rot_tracker=rot_tracker,
                    # Stage 04u: the opening driver (None unless --tracking rotation --init twoview)
                    open_tracker=open_tracker,
                    # Stage 04-m0: the integrated system M0 (--live_map on)
                    live_map=getattr(args, "live_map", "off"))


def _build_oracle(args, ground_truth_path, timestamps, n_total):
    """Stage 03c (spec A1/A6): the Oracle for --oracle other than none, else None.

    With --oracle none nothing here runs: oracle.py is not even imported. Otherwise the
    ground-truth file and per-frame timestamps are required; any failure prints one
    ">> ORACLE ERROR:" line and exits with code 2 before any frame is processed."""
    rung = str(getattr(args, 'oracle', 'none') or 'none')
    if rung == 'none':
        return None

    def _fail(msg):
        print(f">> ORACLE ERROR: {msg}")
        sys.exit(2)

    if ground_truth_path is None:
        _fail(f"--oracle {rung} requires --ground_truth")
    if not Path(ground_truth_path).is_file():
        _fail(f"--ground_truth file not found: {ground_truth_path}")
    if timestamps is None or len(timestamps) != int(n_total):
        _fail(f"--oracle {rung} requires per-frame timestamps: give --save_estimated and a usable "
              f"--timestamp_file (rgb.txt) with one row per image")
    try:
        from oracle import Oracle
        return Oracle(rung, str(ground_truth_path), list(timestamps),
                      stationary=getattr(args, 'oracle_stationary', 'estimator'))
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")


def get_parser():
    """Return the argument parser (for CLI or UI to get defaults)."""
    parser = argparse.ArgumentParser(description='Visual Odometry with ORB or SIFT')
    parser.add_argument('--dataset', type=str, default='rgb', metavar='FOLDER',
                        help='Image folder name under project base (default: rgb). Use e.g. rgb2 for a different sequence.')
    parser.add_argument('--image_ext', type=str, default='.png', metavar='EXT',
                        help='Image file extension for glob (default: .png). Use e.g. .jpg for JPEG images.')
    parser.add_argument('--method', type=str, choices=['ORB', 'SIFT'], default='ORB',
                        help='Feature detection method: ORB or SIFT (default: ORB)')
    parser.add_argument('--front_end', type=str, choices=['orb', 'sift'], default='orb',
                        help='Front-end preset (stage 04j): sift expands to --method SIFT and the frozen '
                             'rate-matched protocol of front_end.PROTOCOL (the descriptor bound of '
                             '--lm_max_desc_dist and --df_max_desc_dist, the relocalization ratio); flags '
                             'given explicitly win. orb (no expansion) in the reference configuration.')
    # --- Feature extraction / multi-scale ---
    parser.add_argument(
        '--multiscale_preset',
        type=str,
        choices=['none', 'fast', 'balanced', 'aggressive'],
        default='balanced',
        help='Multi-scale feature preset controlling detector scale-space (default: balanced).',
    )
    parser.add_argument(
        '--multiscale_manual',
        action='store_true',
        default=False,
        help='Enable manual tuning of detailed ORB/SIFT scale-space parameters (default: False).',
    )
    # ORB detailed parameters (used when --multiscale_manual is set)
    parser.add_argument('--orb_nfeatures', type=int, default=500,
                        help='ORB: max number of features (manual tuning; default: 500).')
    parser.add_argument('--orb_scale_factor', type=float, default=1.2,
                        help='ORB: pyramid scale factor between levels (manual tuning; default: 1.2).')
    parser.add_argument('--orb_nlevels', type=int, default=8,
                        help='ORB: number of pyramid levels (manual tuning; default: 8).')
    # SIFT detailed parameters (used when --multiscale_manual is set)
    parser.add_argument('--sift_nfeatures', type=int, default=0,
                        help='SIFT: max number of features (0=unlimited; manual tuning; default: 0).')
    parser.add_argument('--sift_n_octave_layers', type=int, default=3,
                        help='SIFT: number of layers per octave (manual tuning; default: 3).')
    parser.add_argument('--sift_contrast_threshold', type=float, default=0.04,
                        help='SIFT: contrast threshold (manual tuning; default: 0.04).')
    parser.add_argument('--sift_edge_threshold', type=float, default=10.0,
                        help='SIFT: edge threshold (manual tuning; default: 10).')
    parser.add_argument('--sift_sigma', type=float, default=1.6,
                        help='SIFT: sigma at octave 0 (manual tuning; default: 1.6).')
    parser.add_argument('--debug_features', action='store_true', default=False,
                        help='Print feature debug info (keypoint count and size stats).')
    parser.add_argument('--debug_features_every', type=int, default=30,
                        help='When --debug_features: print every N frames (default: 30).')
    parser.add_argument('--pnp_interval', type=int, default=10,
                        help='Run PnP re-localization every N frames (0 to disable, default: 10)')
    parser.add_argument('--pnp_solver', type=str,
                        choices=['iterative', 'multi', 'ap3p', 'epnp', 'sqpnp'],
                        default='iterative',
                        help='PnP solver: iterative (default, RANSAC+LM; timing logged; refinement skipped), multi (run all, pick best), or ap3p, epnp, sqpnp')
    parser.add_argument('--pnp_refinement', action='store_true', default=True,
                        help='Refine PnP pose with Levenberg-Marquardt on inliers (default: True)')
    parser.add_argument('--no_pnp_refinement', action='store_false', dest='pnp_refinement',
                        help='Disable PnP refinement (no solvePnPRefineLM)')
    parser.add_argument('--pnp_use_initial_guess', action='store_true', default=True,
                        help='Use chained VO pose as initial guess for iterative solver (default: True)')
    parser.add_argument('--no_pnp_initial_guess', action='store_false', dest='pnp_use_initial_guess',
                        help='Do not use initial guess (for solver comparison)')
    parser.add_argument('--pnp_min_inlier_ratio', type=float, default=0.55,
                        help='Minimum inlier ratio to accept PnP pose (0.0-1.0, default: 0.55; stricter e.g. 0.65 rejects more)')
    parser.add_argument('--pnp_multi_inlier_gate', type=float, default=0.8,
                        help='When --pnp_solver multi: only consider solvers with inliers >= this fraction of max (default 0.8); then pick by lowest reprojection error')
    parser.add_argument('--pnp_adaptive_gate', action='store_true', default=False,
                        help='Widen PnP step/correction safety gate proportionally to consecutive PnP failures (reduces unfair rejections after VO drift)')
    parser.add_argument('--pnp_adaptive_gate_factor', type=float, default=0.2,
                        help='When --pnp_adaptive_gate: multiply gate by 1 + factor*consecutive_failures (default 0.2; e.g. 5 failures -> 2x gate)')
    parser.add_argument('--pnp_reprojection_error', type=float, default=3.0,
                        help='PnP RANSAC max reprojection error in pixels for inliers (default 3.0)')
    parser.add_argument('--use_relative_scale', action='store_true', default=False,
                        help='Estimate translation scale from map points for Essential path (default: False)')
    parser.add_argument('--no_relative_scale', action='store_false', dest='use_relative_scale',
                        help='Use unit translation (scale=1); test PnP without scale recovery')
    parser.add_argument('--ba_interval', type=int, default=30,
                        help='Run Bundle Adjustment every N frames (0 to disable, default: 30)')
    parser.add_argument('--ba_window_size', type=int, default=12,
                        help='Bundle Adjustment sliding-window size in frames (default: 12)')
    parser.add_argument('--ba_max_points', type=int, default=250,
                        help='Maximum map points per BA run (default: 250)')
    parser.add_argument('--ba_max_allowed_step_factor', type=float, default=3.0,
                        help='BA sanity check: max_allowed_step = max(this * p95_step, pre_max_step+2) (default: 3.0)')
    parser.add_argument('--ba_bad_reproj_ratio', type=float, default=1.2,
                        help='BA sanity check: reject if err_after > this * err_before (default: 1.2)')
    parser.add_argument('--use_ba', action='store_true', default=True,
                        help='Run Bundle Adjustment when ba_interval > 0 (default: True)')
    parser.add_argument('--no_ba', action='store_false', dest='use_ba',
                        help='Disable Bundle Adjustment completely')
    parser.add_argument('--use_homography', action='store_true', default=False,
                        help='When True, choose between homography and essential matrix by threshold; when False, use essential matrix only (default: False)')
    parser.add_argument('--no_homography', action='store_false', dest='use_homography',
                        help='Use essential matrix only (disable homography path)')
    parser.add_argument('--ba_method', type=str,
                        choices=['lm', 'gradient_descent', 'newton'], default='lm',
                        help='BA optimizer: lm (Levenberg-Marquardt), gradient_descent (CG), newton (Gauss-Newton) (default: lm)')
    parser.add_argument('--ba_use_analytical_jacobian', action='store_true', default=True,
                        help='Use analytical Jacobian for BA (default: True)')
    parser.add_argument('--ba_use_numerical_jacobian', action='store_false', dest='ba_use_analytical_jacobian',
                        help='Use numerical (finite-difference) Jacobian for BA')

    # BA scheduling / neighborhood selection mode.
    parser.add_argument(
        '--ba_mode',
        type=str,
        choices=['sliding_window', 'keyframe_local_covisibility'],
        default='sliding_window',
        help='Bundle Adjustment mode: sliding_window (default) or keyframe_local_covisibility.',
    )
    parser.add_argument(
        '--keyframe_local_topk',
        type=int,
        default=5,
        help='When ba_mode=keyframe_local_covisibility: number of neighbor keyframes to include (default: 5).',
    )
    parser.add_argument(
        '--keyframe_local_min_shared_points',
        type=int,
        default=20,
        help='When ba_mode=keyframe_local_covisibility: minimum shared map points to treat a keyframe as neighbor (default: 20).',
    )

    # --- BA point gating / map-point pruning ---
    parser.add_argument('--ba_min_point_observations', type=int, default=3,
                        help='BA: minimum number of observations for a map point to be included (default: 3).')
    parser.add_argument('--ba_point_sanity_check', action='store_true', default=True,
                        help='BA: enable basic point sanity checks (finite coords) (default: True).')
    parser.add_argument('--no_ba_point_sanity_check', action='store_false', dest='ba_point_sanity_check',
                        help='BA: disable point sanity checks.')
    # Stage 01a: back-end disposition (gauge, robust loss, solve length).
    parser.add_argument('--ba_max_fixed_observers', type=int, default=0,
                        help='BA: keep only the most recent N fixed outer observer frames (0 = unlimited, default: 0).')
    parser.add_argument('--ba_huber_delta_px', type=float, default=2.4477,
                        help='BA: Huber loss threshold on the reprojection residual norm in px (default: 2.4477 = sqrt(5.991)).')
    parser.add_argument('--ba_min_anchor_points', type=int, default=10,
                        help='BA: points seen by >= 2 fixed cameras needed to anchor scale with outer observers; otherwise a second window frame is fixed (default: 10).')
    parser.add_argument('--ba_max_iterations', type=int, default=10,
                        help='BA: maximum Levenberg-Marquardt iterations per solve (default: 10).')
    # Stage 04y: the local-BA outlier step of ORB-SLAM section VI.D (chi-square 5.991 on sigma-normalized
    # residuals at the middle and at the end of the solve; final outliers unlinked on an accepted solve).
    parser.add_argument('--ba_outlier_reject', choices=['off', 'on'], default='off',
                        help='Stage 04y: local-BA outlier rejection as in ORB-SLAM (Mur-Artal et al. 2015) '
                             'section VI.D: observations whose sigma-normalized squared residual exceeds 5.991 '
                             '(or whose point is behind the camera) are dropped at the middle of the solve and '
                             'the final outliers are unlinked from the map when the solve is accepted. Inert '
                             'with --no_ba (default: off).')
    # Stage 04z (ruling R57-2, conventions.md#local-ba-participation): supported cameras only in local
    # BA, and the legacy jump gate as a switch.
    parser.add_argument('--ba_free_cameras', choices=['all', 'supported'], default='all',
                        help='Stage 04z, ruling R57-2 (conventions.md#local-ba-participation): "supported" '
                             'holds fixed, like an outer observer, every local-BA window camera that observes '
                             'fewer than 30 points in the BA problem (tested again on the inlier rows after the '
                             'outlier step\'s mid classification); a solve with no free camera is skipped. '
                             'Refuses --ba_jump_gate on. Inert with --no_ba (default: all).')
    parser.add_argument('--ba_jump_gate', choices=['on', 'off'], default='on',
                        help='Stage 04z: the local-BA jump gate, a threshold in absolute map units (window '
                             'step + 2.0 units); kept only for byte identity of verified runs. The reprojection '
                             'gate stays either way. Inert with --no_ba (default: on).')

    # Pruning policy A (low-observation age prune) was retired in stage 04-1a: landmark
    # culling is local_mapping.cull_points (--lm_cull_after_keyframes / --lm_min_keyframe_obs).
    parser.add_argument('--prune_interval', type=int, default=30,
                        help='Run pruning every N frames (Frame.id clock). 0 disables pruning entirely (default: 30).')

    # Pruning policy B: persistently high reprojection error (disabled by default)
    parser.add_argument('--prune_high_reproj', action='store_true', default=False,
                        help='Prune map points with persistently high reprojection error (default: False).')
    parser.add_argument('--no_prune_high_reproj', action='store_false', dest='prune_high_reproj',
                        help='Disable pruning of high-reprojection-error points.')

    # Pruning policy D: numerical / geometric corruption (disabled by default)
    parser.add_argument('--prune_numeric_corruption', action='store_true', default=False,
                        help='Prune numerically corrupt map points (NaN/inf) (default: False).')
    parser.add_argument('--no_prune_numeric_corruption', action='store_false', dest='prune_numeric_corruption',
                        help='Disable pruning of numerically corrupt points.')

    # Pruning policy E: repeated outlier hits + last-seen age (disabled by default)
    parser.add_argument('--prune_outlier_hits', action='store_true', default=False,
                        help='Prune points with many outlier hits that have not been seen recently (default: False).')
    parser.add_argument('--no_prune_outlier_hits', action='store_false', dest='prune_outlier_hits',
                        help='Disable pruning based on outlier-hit counters.')
    parser.add_argument('--prune_outlier_hit_count', type=int, default=3,
                        help='Prune (policy E): minimum outlier hits to consider pruning (default: 3).')
    parser.add_argument('--prune_outlier_last_seen_age', type=int, default=20,
                        help='Prune (policy E): minimum age since last seen (in frame ids) (default: 20).')
    parser.add_argument('--local_map_window', type=int, default=0,
                        help='Number of recent frames for local map tracking (0 to disable, default: 0)')
    parser.add_argument('--local_map_search_radius', type=float, default=25.0,
                        help='Local map: pixel search radius around projection (Stage 3 default 25; was 40).')
    parser.add_argument('--local_map_max_desc_dist', type=float, default=None,
                        help='Local map: max descriptor distance (None = ORB 48 / SIFT 150 Stage 3 defaults).')
    parser.add_argument('--local_map_max_depth_ratio', type=float, default=5.0,
                        help='Local map: reject matches whose depth differs from median by more than this factor (0 disables).')
    parser.add_argument('--use_point_fusion', action='store_true', default=True,
                        help='Merge near-duplicate map points periodically (Stage 3; default: True).')
    parser.add_argument('--no_point_fusion', action='store_false', dest='use_point_fusion',
                        help='Disable point fusion / duplicate merge.')
    parser.add_argument('--fusion_interval', type=int, default=30,
                        help='Run point fusion every N frames (default: 30).')
    parser.add_argument('--fusion_max_dist', type=float, default=0.05,
                        help='Fusion: max 3D distance to consider duplicates (default: 0.05).')
    parser.add_argument('--fusion_max_desc_dist', type=float, default=40.0,
                        help='Fusion: max descriptor distance for merge (ORB Hamming; default: 40).')
    # --- Stage 04-1a: point source (design.md slot "Point source") and the local mapper's constants ---
    parser.add_argument('--point_source', type=str, choices=['keyframe_pair', 'depth_filter'], default='keyframe_pair',
                        help='Landmark source arm: keyframe_pair (default; triangulation between the new keyframe '
                             'and its neighbours at keyframe insertion, local_mapping.py) or depth_filter (stage 04g; '
                             'the SVO per-feature depth filter of depth_filter.py, driven by depth_filter_arm.py, '
                             'tested alternative of the Point-source slot: seeds per keyframe, one measurement per '
                             'posed frame, promotion to a landmark with its inlier measurements as observations).')
    _lm = LocalMappingParams()
    parser.add_argument('--lm_min_shared_points', type=int, default=_lm.min_shared_points,
                        help=f'Local mapping: shared landmarks for an earlier keyframe to be a neighbour (default: {_lm.min_shared_points}).')
    parser.add_argument('--lm_max_neighbours', type=int, default=_lm.max_neighbours,
                        help=f'Local mapping: at most this many neighbour keyframes per new keyframe (default: {_lm.max_neighbours}).')
    parser.add_argument('--lm_min_baseline_depth_ratio', type=float, default=_lm.min_baseline_depth_ratio,
                        help=f'Local mapping: skip a pair whose baseline / median depth is below this (default: {_lm.min_baseline_depth_ratio}).')
    parser.add_argument('--lm_epipolar_chi2', type=float, default=_lm.epipolar_chi2,
                        help=f'Local mapping: epipolar band, squared distance <= this * sigma^2 (default: {_lm.epipolar_chi2}).')
    parser.add_argument('--lm_epipole_radius_px', type=float, default=_lm.epipole_radius_px,
                        help=f'Local mapping: exclude candidates within this * sqrt(sigma) px of the epipole (default: {_lm.epipole_radius_px}).')
    parser.add_argument('--lm_max_desc_dist', type=float, default=_lm.max_desc_dist,
                        help=f'Local mapping: max descriptor distance for a pair match / insertion (default: {_lm.max_desc_dist}).')
    parser.add_argument('--lm_min_parallax_deg', type=float, default=_lm.min_parallax_deg,
                        help=f'Local mapping: min parallax between the two viewing rays in degrees (default: {_lm.min_parallax_deg}).')
    parser.add_argument('--lm_reproj_chi2', type=float, default=_lm.reproj_chi2,
                        help=f'Local mapping: squared reprojection error in each image <= this * sigma^2 (default: {_lm.reproj_chi2}).')
    parser.add_argument('--lm_scale_ratio_factor', type=float, default=_lm.scale_ratio_factor,
                        help=f'Local mapping: scale-consistency factor (default: {_lm.scale_ratio_factor} = 1.5 * scale factor 1.2).')
    parser.add_argument('--lm_insert_search_radius_px', type=float, default=_lm.insert_search_radius_px,
                        help=f'Local mapping: observation-insertion search radius in px (default: {_lm.insert_search_radius_px}).')
    parser.add_argument('--lm_cull_after_keyframes', type=int, default=_lm.cull_after_keyframes,
                        help=f'Local mapping: judge a landmark once this many keyframes have passed since its creation (default: {_lm.cull_after_keyframes}).')
    parser.add_argument('--lm_min_keyframe_obs', type=int, default=_lm.min_keyframe_obs,
                        help=f'Local mapping: keyframe observations a judged landmark needs to survive (default: {_lm.min_keyframe_obs}).')
    parser.add_argument('--lm_confirm', action='store_true', default=False, dest='lm_confirm',
                        help='Local mapping (stage 04e): a gate-accepted candidate becomes a landmark only after it is '
                             're-found within chi-square in a further neighbour keyframe at a disparity ratio r with '
                             'r >= ratio and |r - 1| >= ratio (--lm_confirm_disparity_ratio). Off in the reference '
                             'configuration.')
    parser.add_argument('--lm_confirm_disparity_ratio', type=float, default=_lm.confirm_disparity_ratio,
                        help=f'Local mapping (stage 04e): eligibility ratio of a confirming third view, both sides '
                             f'(default: {_lm.confirm_disparity_ratio} = 3 sigma_L / p, sigma_L = 1.73 px at level 3, '
                             f'p = 10 to 20 px repeat pitch).')
    # --- Stage 04k: sub-pixel refinement along the epipolar line (ruling S5; epipolar_refine.py) ---
    parser.add_argument('--epi_refine', action='store_true', default=False, dest='epi_refine',
                        help='Stage 04k: refine every descriptor match of the point source (pair mapper and depth '
                             'filter) to the sub-pixel maximum of a patch correlation along the epipolar line, '
                             'dropping ambiguous matches (epipolar_refine.py). Off in the reference configuration.')
    # --- Stage 04g: the depth-filter arm (--point_source depth_filter); defaults read from the module ---
    _df = DepthFilterParams()
    _dfa = DepthFilterArmParams()
    parser.add_argument('--df_min_parallax_deg', type=float, default=_df.min_parallax_deg,
                        help=f'Depth filter: theta_min in degrees, entering both promotion conditions (default: '
                             f'{_df.min_parallax_deg}, the map-ladder constant: a 10 percent single-pixel range error at f = 521 px).')
    parser.add_argument('--df_init_depth_ratio_min', type=float, default=_df.init_depth_ratio_min,
                        help=f'Depth filter: the prior on inverse range spans from this fraction of the seed\'s initial range to '
                             f'infinity (default: {_df.init_depth_ratio_min}; SVO seeds with a depth_min at a fraction of the mean depth).')
    parser.add_argument('--df_search_band_sigmas', type=float, default=_df.search_band_sigmas,
                        help=f'Depth filter: half-width of the epipolar search band in posterior sigmas (default: '
                             f'{_df.search_band_sigmas}, the paper\'s +-2 sigma band).')
    parser.add_argument('--df_epipolar_chi2', type=float, default=_df.epipolar_chi2,
                        help=f'Depth filter: a candidate keypoint is in the band when its squared distance to the segment is <= this '
                             f'* sigma^2 (default: {_df.epipolar_chi2}, chi-square 1 dof 95 percent, as the pair matcher).')
    parser.add_argument('--df_max_desc_dist', type=float, default=_df.max_desc_dist,
                        help=f'Depth filter: max descriptor distance of a measurement (default: {_df.max_desc_dist}, ORB-SLAM\'s TH_LOW).')
    parser.add_argument('--df_beta_init', type=float, default=_df.beta_init,
                        help=f'Depth filter: a = b at seeding, the weight of the inlier-ratio prior (default: {_df.beta_init}, '
                             f'SVO\'s Seed constructor).')
    parser.add_argument('--df_max_seed_keyframes', type=int, default=_df.max_seed_keyframes,
                        help=f'Depth filter: a seed is retired once its reference keyframe is this many keyframes old (default: '
                             f'{_df.max_seed_keyframes}, the local mapper\'s neighbourhood window). Age-lifetime retirement only; '
                             f'no effect under --df_lifetime r2 (stage 04i; the bootstrap window is --df_bootstrap_window).')
    parser.add_argument('--df_responsibility_inlier', type=float, default=_df.responsibility_inlier,
                        help=f'Depth filter: a measurement with mixture responsibility >= this is an inlier, counted for promotion and '
                             f'linked as an observation (default: {_df.responsibility_inlier}, the two-component threshold).')
    parser.add_argument('--df_promote_factor', type=float, default=_dfa.promote_factor,
                        help=f'Depth filter arm (stage 04g): promote a converged seed only once theta_acc >= this * theta_min '
                             f'(default: {_dfa.promote_factor}, the paper\'s first-convergence timing; the tested alternative runs '
                             f'9.5 = the 04d along-line matching error p50 of 9.51 px over the 1 px the variance assumes).')
    parser.add_argument('--df_bootstrap_depth_ratio', type=float, default=_dfa.bootstrap_depth_ratio,
                        help=f'Depth filter arm (stage 04g): scene range of a keyframe with no visible landmark, as a ratio to the '
                             f'widest baseline of the previous bootstrap_window keyframes (default: {_dfa.bootstrap_depth_ratio}; '
                             f'the 04d creating pairs sat at 0.66 m against 2.5 to 3 m of scene depth, ratio 4 to 4.5).')
    # --- Stage 04i: the seed lifetime (ruling R2, 2026-09-22); defaults read from the arm's dataclass ---
    parser.add_argument('--df_lifetime', type=str, choices=['age', 'r2'], default=_dfa.lifetime,
                        help=f'Depth filter arm (stage 04i): seed lifetime. "age" is the verified 04h driver (retire by '
                             f'max_seed_keyframes, promote at convergence and theta_acc >= k theta_min); "r2" is the per-feature '
                             f'state machine of ruling R2 (dormancy, disparity skip, descriptor death, the V&H failure test, '
                             f'N distinct baselines, tentative points, the live-seed cap) (default: {_dfa.lifetime}).')
    parser.add_argument('--df_bootstrap_window', type=int, default=_dfa.bootstrap_window,
                        help=f'Depth filter arm (stage 04i): the bootstrap depth window of scene_range (source "window") is the '
                             f'previous this-many keyframes, under both lifetimes (default: {_dfa.bootstrap_window}, the value '
                             f'max_seed_keyframes gave it in 04h; R2 decouples the window from the seed lifetime).')
    parser.add_argument('--df_match_error_px', type=float, default=_dfa.match_error_px,
                        help=f'Depth filter arm (stage 04i, r2): epsilon_match, the matcher\'s along-line error scale; a predicted '
                             f'disparity below it is skipped, baselines are distinct when their disparities differ by at least it '
                             f'(default: {_dfa.match_error_px}, the 04d und_attr_o3plus_match_err_px_p50 of 9.51 px).')
    parser.add_argument('--df_death_travel_ratio', type=float, default=_dfa.death_travel_ratio,
                        help=f'Depth filter arm (stage 04i, r2): descriptor death; an unmatched seed is dead once the camera has '
                             f'travelled this * z_init since its last match, measured at eligible attempts only (default: '
                             f'{_dfa.death_travel_ratio} = 0.84 m of matchable ORB baseline (04b) over the hall\'s 2.75 m scene range).')
    parser.add_argument('--df_outlier_eta', type=float, default=_dfa.outlier_eta,
                        help=f'Depth filter arm (stage 04i, r2): eta of the Vogiatzis-Hernandez failure test P(pi < eta) >= conf '
                             f'(default: {_dfa.outlier_eta}, section 4.3 of the paper).')
    parser.add_argument('--df_outlier_conf', type=float, default=_dfa.outlier_conf,
                        help=f'Depth filter arm (stage 04i, r2): the confidence of the same test (default: {_dfa.outlier_conf}, '
                             f'the paper\'s 99 percent).')
    parser.add_argument('--df_distinct_baselines', type=int, default=_dfa.distinct_baselines,
                        help=f'Depth filter arm (stage 04i, r2): convergence needs inlier views at this many baselines whose predicted '
                             f'disparities differ pairwise by >= match_error_px (default: {_dfa.distinct_baselines}, the route\'s '
                             f'third view: the first that can contradict two along-line-erroneous measurements).')
    parser.add_argument('--df_seed_cap', type=int, default=_dfa.seed_cap,
                        help=f'Depth filter arm (stage 04i, r2): cap on live seeds, dormant-longest evicted first (default: '
                             f'{_dfa.seed_cap}: SVO\'s 50 retained keyframes times 04h\'s 467 seeds per keyframe, rounded).')
    parser.add_argument('--df_trace', type=str, default=_dfa.trace_path, dest='df_trace_path', metavar='FILE',
                        help='Depth filter arm (stage 04i, r2): append one line per matched measurement to FILE (the harness\'s '
                             'per-measurement instrument); off when empty (default).')
    parser.add_argument('--map_metrics', type=str, default=None, metavar='FILE',
                        help='Write end-of-run map metrics (obs/point, duplicates, KF links) to FILE.')
    parser.add_argument('--step_lengths', type=str, default=None, metavar='FILE',
                        help='Write per-frame step lengths CSV to FILE (default: next to --save_estimated).')
    parser.add_argument('--log_file', type=str, default='log_file.txt', metavar='FILE',
                        help='Write pipeline log (epipolar, PnP, BA, etc.) to FILE (default: log_file.txt). Use --log_file "" to print to terminal only.')
    parser.add_argument('--no_display', action='store_true',
                        help='Run without Pangolin/OpenCV (headless); use with --save_estimated')
    parser.add_argument('--save_estimated', type=str, default=None, metavar='FILE',
                        help='Save estimated trajectory to FILE in TUM format (timestamp tx ty tz qx qy qz qw), like groundtruth.txt')
    parser.add_argument('--timestamp_file', type=str, default=None,
                        help='Only for --save_estimated: path to rgb.txt if it is not next to your images. If omitted, timestamps are read from rgb.txt in the image folder or its parent.')
    parser.add_argument('--t_norm_log', type=str, default=None, metavar='FILE',
                        help='Log ||t|| (norm of translation from recoverPose) per frame to FILE for scale/debugging (one line per frame: frame_id norm_t).')
    parser.add_argument('--default_k', action='store_true',
                        help='Use default camera intrinsics (DEFAULT_K in frame.py) instead of inferring from image size.')
    parser.add_argument('--undistort', action='store_true', default=False, dest='undistort',
                        help='Undistort keypoint coordinates once at extraction with the frozen TUM '
                             'fr2 coefficients (camera.DISTORTION_FR2; conventions.md#camera-model-and-distortion). '
                             'Requires --default_k. Off in the reference configuration until R04.')
    # Stage 04l (milestone 04-2): the map-based tracker, run against a perfect map only.
    parser.add_argument('--tracking', type=str, choices=['chain', 'map', 'rotation'], default='chain',
                        help='Pose source: chain (default; the frame-to-frame chain with its PnP, hold and '
                             'relocalization policies) or map (stage 04l: prediction, guided search against the '
                             'map, motion-only BA; requires --m1_map, --default_k, --undistort, --pnp_interval 0, '
                             '--no_loop_closure, --no_relocalization, --no_ba, --no_point_fusion) or '
                             'rotation (stage 04r): the pre-map rotation-only mode; every frame\'s orientation from '
                             'a rotation-only fit against the current panorama keyframe, translation held and '
                             'unsupported, no map (requires --default_k, --undistort, --pnp_interval 0, '
                             '--no_loop_closure, --no_relocalization, --no_ba, --no_point_fusion).')
    parser.add_argument('--m1_map', type=str, default=None, metavar='FILE',
                        help='Stage 04l: the perfect-map file written by the harness builder (rungs M1a / M1b); '
                             'only with --tracking map.')
    parser.add_argument('--m1_mode', type=str, choices=['m1a', 'm1b'], default='m1a',
                        help='Stage 04l: m1a (the tracker associates keypoints to the map, default) or m1b (the '
                             'map file\'s 2D-3D correspondence lines are used).')
    parser.add_argument('--map_reloc', type=str, choices=['none', 'pnp'], default='none',
                        help='Stage 04m: none (default; a LOST frame carries its prediction, the 04l tracker) or '
                             'pnp (a frame after a LOST frame is relocalized without a prior by EPnP inside RANSAC '
                             'and motion-only BA; the motion model restarts after a loss); only with --tracking map.')
    parser.add_argument('--reloc_solver', type=str, choices=['epnp', 'sqpnp', 'newton'], default='epnp',
                        help='stage 04q, D1 part C: the solver that refits the RANSAC consensus set of a '
                             'relocalization: epnp (default; the 04m relocalization), sqpnp (OpenCV SQPnP) or newton '
                             '(Newton-PnP, npnp.py); only with --tracking map and --map_reloc pnp.')
    parser.add_argument('--reloc_shadow', action='store_true', default=False,
                        help='stage 04q, D1 part C: after every tracked frame, run the relocalization of that frame '
                             'with each of the three solvers and log it without using it; only with --tracking map '
                             'and --map_reloc pnp.')
    # Stage 04v (milestone 04-3b): the rotation-only state inside a map.
    parser.add_argument('--map_rot_state', type=str, choices=['off', 'on', 'r52'], default='off',
                        help='Stage 04v, rulings B1-B7, B6 and T2: off (default; the 04m map tracker as it was) or on '
                             '(when the map pose fails while the rotation-only fit succeeds, the orientation comes from '
                             'the fit against a panorama keyframe anchored at the last tracked frame, the position is '
                             'bridged by the constant body twist, and the state ends only by normal tracking or '
                             'relocalization at 30 inliers); only with --tracking map and --map_reloc pnp. '
                             'r52 = stage 04x, rulings R52-1 to R52-3: entry only on detected rotation without parallax '
                             '(Pirchheim 4.2), position held at the anchor (label rotation), a panorama keyframe switch '
                             'also needs r > 0.2 (4.4), lost frames predicted with the windowed body velocity '
                             '(R50-2, R50-3).')
    # Stage 04u (milestone 04-3a): the two-view initializer wired into the rotation-only mode.
    parser.add_argument('--init', type=str, choices=['none', 'twoview'], default='none',
                        help='Stage 04u: none (default; the pre-map rotation-only mode of 04r as it was) or twoview '
                             '(the newest frame is tried against the two most recent panorama keyframes with the '
                             'two-view initializer; an accepted map is handed to the map tracker with its '
                             'relocalizing state and must survive 10 tracked frames); only with --tracking rotation.')
    parser.add_argument('--init_refit', type=str, choices=['off', 'shadow', 'live'], default='off',
                        help='Stage 04u, ruling R41-1: the initializer\'s refit switch. off (default); shadow = the '
                             'refit is computed on every attempted pair and logged, never used; live = the refit is '
                             'the initializer\'s setting. Only with --init twoview.')
    # Stage 04-m0 (milestone 04-4): the integrated system M0.
    parser.add_argument('--live_map', type=str, choices=['off', 'on'], default='off',
                        help='the integrated system M0 (design.md 04-4 row): the map tracker tracks the map the '
                             'mapper builds')
    # Stage 04-m1 (milestone 04-4): registered risk 1's residual-consistency test.
    parser.add_argument('--residual_test', type=str, choices=['off', 'log', 'on'], default='off',
                        help='Stage 04-m1, ruling R63-1 (registered risk 1): the residual-consistency test of every '
                             'pose the 30-inlier count accepts (tracking, relocalization, rotation-state exits, the '
                             '04q shadow): the mean of the final inliers\' squared sigma-normalized residuals against '
                             'the exact one-sided 95 percent point of the mean of truncated chi-square(2) draws '
                             '(R63-7). off (default; nothing computed or printed), log (computed and logged, no '
                             'decision changes) or on (a count-accepted pose is accepted only if it also passes). '
                             'Only with --tracking map or --live_map on; refuses --map_rot_state on.')
    # Stage 04-r04 (milestone 04-4, ruling R65-4): the post-gap label rule of the r52 fallback.
    parser.add_argument('--map_rot_gap', type=str, choices=['carry', 'lost'], default='carry',
                        help='Stage 04-r04, ruling R65-4: carry (default; every run as before) or lost (in the r52 '
                             'rotation-only state, a frame after a recording gap (rgb interval over 150 ms) is '
                             'labelled lost, never orientation-supported; its pose is unchanged). lost requires '
                             '--map_rot_state r52.')
    parser.add_argument('--camera_spacing', type=int, default=None,
                        help='Draw camera frustum every N frames. Omit for keyframe-only mode (default).')
    parser.add_argument('--ground_truth', type=str, default=None, metavar='FILE',
                        help='Path to TUM-format ground truth file for trajectory overlay in visualizer.')
    # Stage 03c: the oracle ladder pose rungs (off by default; see oracle.py).
    parser.add_argument('--oracle', type=str, choices=['none', 'o1', 'o2', 'o3', 'o4'], default='none',
                        help='Oracle ladder rung: none (default; no ground truth enters the estimator), '
                             'o1 true step magnitude, o2 true rotation and direction, o3 full true relative '
                             'motion, o4 true absolute pose at PnP frames. Requires --ground_truth and timestamps.')
    parser.add_argument('--oracle_stationary', type=str, choices=['estimator', 'zero'], default='estimator',
                        help='Rung o2 only: direction injected on stationary ground-truth pairs (< 1 mm): '
                             'the estimator\'s own unit direction (default) or the zero vector.')
    parser.add_argument('--viewer_fps', type=float, default=30, metavar='HZ',
                        help='Viewer refresh rate in Hz when display is enabled (default: 30).')
    parser.add_argument('--tracker_delay_ms', type=float, default=0, metavar='MS',
                        help='Extra delay in ms after each tracker frame (default: 0; increase to slow trajectory building).')
    parser.add_argument('--show_grid', action='store_true', default=True,
                        help='Draw grid floor in visualizer (default: True).')
    parser.add_argument('--no_show_grid', action='store_false', dest='show_grid',
                        help='Do not draw grid in visualizer.')
    parser.add_argument('--use_loop_closure', action='store_true', default=True,
                        help='Run loop closure detection and trajectory correction (default: True).')
    parser.add_argument('--no_loop_closure', action='store_false', dest='use_loop_closure',
                        help='Disable loop closure; keyframes still used for visualization.')
    parser.add_argument('--loop_closure_strategy', type=str,
                        choices=['pose_graph', 'linear'], default='pose_graph',
                        help='When loop closure is enabled: pose_graph (default) or linear interpolation for trajectory correction.')
    parser.add_argument('--loop_detection_strategy', type=str,
                        choices=['spaced', 'bow'], default='spaced',
                        help='Loop candidate selection: spaced (N evenly-spaced keyframes, default) or bow (Bag-of-Words with ORBvoc).')
    parser.add_argument('--vocabulary_path', type=str, default=None, metavar='FILE',
                        help='Path to ORBvoc.txt (or DBoW3 vocabulary). Required when --loop_detection_strategy is bow.')
    parser.add_argument('--num_spaced_candidates', type=int, default=5,
                        help='When loop_detection_strategy is spaced: max number of keyframe candidates to try (default: 5).')
    parser.add_argument('--min_gap', type=int, default=100,
                        help='Min frames between current frame and loop candidate keyframe (default: 100).')
    parser.add_argument('--keyframe_strategy', type=str,
                        choices=['fixed', 'ratio', 'hybrid'], default='fixed',
                        help='Keyframe selection: fixed (every keyframe_interval, default), ratio (tracking ratio), hybrid (min/max spacing + ratio)')
    parser.add_argument('--keyframe_interval', type=int, default=20,
                        help='Keyframe interval for fixed strategy; also used as fallback and for loop-closure check rate (default: 20)')
    parser.add_argument('--keyframe_min_spacing', type=int, default=5,
                        help='Min frames between keyframes for ratio/hybrid (avoids burst, default: 5)')
    parser.add_argument('--keyframe_max_spacing', type=int, default=50,
                        help='Max frames between keyframes for ratio/hybrid (force keyframe after this, default: 50)')
    parser.add_argument('--keyframe_ratio_threshold', type=float, default=0.80,
                        help='Create keyframe when tracked-feature ratio < this for ratio/hybrid (default: 0.80)')
    parser.add_argument('--use_relocalization', action='store_true', default=True,
                        help='When VO fails (e.g. cut in sequence), try to relocalize by matching against keyframes (default: True).')
    parser.add_argument('--no_relocalization', action='store_false', dest='use_relocalization',
                        help='Disable relocalization on VO failure.')
    parser.add_argument('--relocalization_match_threshold', type=int, default=25,
                        help='Min descriptor matches for relocalization (default: 25).')
    parser.add_argument('--relocalization_min_2d3d_pairs', type=int, default=10,
                        help='Min 2D-3D pairs for relocalization PnP (default: 10).')
    parser.add_argument('--relocalization_min_pnp_inliers', type=int, default=10,
                        help='Min PnP RANSAC inliers to accept relocalization (default: 10).')
    parser.add_argument('--relocalization_ratio_threshold', type=float, default=0.85,
                        help='Lowe ratio for relocalization descriptor matching (default: 0.85, more permissive than 0.75).')
    parser.add_argument('--relocalization_sanity_check', action='store_true', default=True,
                        help='Reject relocalization if PnP pose is too far from matched keyframe (default: enabled).')
    parser.add_argument('--no_relocalization_sanity_check', action='store_false', dest='relocalization_sanity_check',
                        help='Disable relocalization distance sanity check.')
    parser.add_argument('--relocalization_max_distance_from_keyframe', type=float, default=5.0,
                        help='When relocalization sanity check is on: max distance from keyframe in world units (default: 5.0).')
    return parser


if __name__ == "__main__":
    main()
