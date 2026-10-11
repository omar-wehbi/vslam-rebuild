"""Stage 04d: the camera model behind --undistort (conventions.md#camera-model-and-
distortion). With the switch off nothing changes (T3); with it on, keypoint pixels
are undistorted once at extraction with cv2.undistortPoints(..., K, D, P=K) and
every other attribute of the keypoint is preserved (T1, T2); the flag is wired
through main.py and refused without --default_k (T4)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import camera  # noqa: E402
import frame as frame_mod  # noqa: E402
from frame import DEFAULT_K, Frame  # noqa: E402

D = camera.DISTORTION_FR2
CX, CY = float(DEFAULT_K[0, 2]), float(DEFAULT_K[1, 2])


# --------------------------------------------------------------------------- #
# T1. Round trip and magnitude
# --------------------------------------------------------------------------- #
def test_t1_round_trip_and_magnitude():
    xs = np.arange(0, 640, 40, dtype=np.float64)
    ys = np.arange(0, 480, 40, dtype=np.float64)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    und = camera.undistort_points(grid, DEFAULT_K, D)
    assert und.shape == grid.shape

    # Round trip: projecting the normalised undistorted points back with the same
    # (K, D) reproduces the raw grid within 0.05 px (the harness self-test's tolerance).
    norm = cv2.undistortPoints(grid.reshape(-1, 1, 2), DEFAULT_K, D)  # P absent -> normalised
    rvec = tvec = np.zeros(3, dtype=np.float64)
    reproj, _ = cv2.projectPoints(
        np.concatenate([norm.reshape(-1, 2), np.ones((norm.shape[0], 1))], axis=1),
        rvec, tvec, DEFAULT_K, D,
    )
    reproj = reproj.reshape(-1, 2)
    assert np.max(np.abs(reproj - grid)) <= 0.05

    # (600, 440) moves by more than 5 px.
    far = camera.undistort_points(np.array([[600.0, 440.0]]), DEFAULT_K, D)[0]
    assert np.linalg.norm(far - np.array([600.0, 440.0])) > 5.0

    # Every grid point farther than 100 px from the principal point moves toward it.
    c = np.array([CX, CY])
    for raw, u in zip(grid, und):
        r_raw = np.linalg.norm(raw - c)
        r_und = np.linalg.norm(u - c)
        if r_raw > 100.0:
            assert r_und <= r_raw + 0.1, (raw, u, r_raw, r_und)

    # Every undistorted grid point lies inside the image.
    assert np.all(und[:, 0] >= 0.0) and np.all(und[:, 0] < 640.0)
    assert np.all(und[:, 1] >= 0.0) and np.all(und[:, 1] < 480.0)

    # The principal point moves by less than 0.01 px.
    pp = camera.undistort_points(np.array([[CX, CY]]), DEFAULT_K, D)[0]
    assert np.linalg.norm(pp - np.array([CX, CY])) < 0.01

    # Empty input.
    empty = camera.undistort_points(np.zeros((0, 2)), DEFAULT_K, D)
    assert empty.shape == (0, 2)


# --------------------------------------------------------------------------- #
# T2 / T3. Frame with the switch on vs off
# --------------------------------------------------------------------------- #
def _textured_image():
    return np.asarray(
        np.random.default_rng(0).integers(0, 256, (480, 640), dtype=np.uint8)
    )


def test_t2_frame_switch_on_vs_off():
    img = _textured_image()
    K = DEFAULT_K.copy()
    off = Frame(img, K, "ORB", "BF")
    on = Frame(img, K, "ORB", "BF", distortion=D)

    assert len(off.KeyPoints) == len(on.KeyPoints)
    assert len(off.KeyPoints) > 100
    assert np.array_equal(off.Descriptors, on.Descriptors)

    n = len(off.KeyPoints)
    for i in range(n):
        ko, kn = off.KeyPoints[i], on.KeyPoints[i]
        assert ko.size == kn.size
        assert ko.angle == kn.angle
        assert ko.response == kn.response
        assert ko.octave == kn.octave
        assert ko.class_id == kn.class_id

    assert np.array_equal(off.keypoint_xy_raw, on.keypoint_xy_raw)
    off_pts = np.array([kp.pt for kp in off.KeyPoints], dtype=np.float64)
    assert np.array_equal(off.keypoint_xy_raw, off_pts)

    expected_und = camera.undistort_points(on.keypoint_xy_raw, K, D)
    on_pts = np.array([kp.pt for kp in on.KeyPoints], dtype=np.float64)
    assert np.max(np.abs(on_pts - expected_und)) < 1e-3

    diffs = np.linalg.norm(on_pts - off_pts, axis=1)
    assert np.max(diffs) > 2.0

    assert off.distortion is None
    assert on.distortion is D


def test_t3_switch_off_is_the_old_frame():
    img = _textured_image()
    K = DEFAULT_K.copy()
    default_kw = Frame(img, K, "ORB", "BF")
    explicit_none = Frame(img, K, "ORB", "BF", distortion=None)

    assert np.array_equal(
        np.array([kp.pt for kp in default_kw.KeyPoints]),
        np.array([kp.pt for kp in explicit_none.KeyPoints]),
    )
    assert np.array_equal(default_kw.Descriptors, explicit_none.Descriptors)

    detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    direct_kp, direct_desc = detector.detectAndCompute(img, None)

    assert len(direct_kp) == len(default_kw.KeyPoints)
    assert np.array_equal(
        np.array([kp.pt for kp in direct_kp]),
        np.array([kp.pt for kp in default_kw.KeyPoints]),
    )
    assert np.array_equal(direct_desc, default_kw.Descriptors)


# --------------------------------------------------------------------------- #
# T4. Parser and settings
# --------------------------------------------------------------------------- #
def test_t4_parser_and_settings(monkeypatch, tmp_path):
    import main as main_mod

    p = main_mod.get_parser()
    a = p.parse_args(["--dataset", "rgb"])
    assert a.undistort is False

    b = p.parse_args(["--dataset", "rgb", "--default_k", "--undistort"])
    assert b.undistort is True

    # --undistort without --default_k raises SystemExit before any frame is read.
    def _fail_if_called(*a, **kw):
        raise AssertionError("iter_frames was called before the --default_k check")

    monkeypatch.setattr(main_mod, "iter_frames", _fail_if_called)
    args = p.parse_args(["--dataset", "rgb", "--undistort"])
    with pytest.raises(SystemExit):
        main_mod.run_pipeline(args)

    out = tmp_path / "resolved_settings.json"
    main_mod.write_resolved_settings(b, out)
    import json
    data = json.loads(out.read_text())
    assert data["undistort"] is True
