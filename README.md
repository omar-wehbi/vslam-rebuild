# V-SLAM: repairing a monocular visual SLAM system

A Python monocular visual SLAM system for a camera on a wheeled robot, and the record of rebuilding it. It started
as a course project: a frame-to-frame odometry chain with SLAM parts attached. A review named four faults:
- the path jumps;
- the map is at the wrong depth;
- map points are invented while the camera only rotates;
- loop closure has no visible effect.

We diagnosed the causes, repaired the foundation, measured what the original architecture could and could not do,
replaced it with map-based tracking, and scored the final system once on a sequence it had never seen.

**The short answer:** the repairs and the components worked, each on its own. The integrated system did not reach
its goal. On the development sequence it is right where it holds a pose but cannot stay posed: it holds a full pose
on 22 % of the frames. On the test sequence it also cannot keep its map's scale. The full account is in
**[`theory/closing-report.md`](theory/closing-report.md)**.

---

## 1. What the system does

Given the colour images of a single camera, it estimates the camera's path and a sparse 3D map. It writes the path
in TUM format, labelled frame by frame:
- **full pose:** posed against the map;
- **orientation only:** turning, with no usable map;
- **lost.**

Monocular vision cannot observe absolute scale, so paths are compared to ground truth after a similarity alignment.
The depth images of the TUM datasets are used only to *score* map points, never as input.

Data: TUM RGB-D `fr2/pioneer_slam3` (development, 2,544 frames) and `fr2/pioneer_slam` (test, 2,921 frames). The
datasets are not shipped.

---

## 2. The final system

Each frame goes through the following steps (the final configuration, "R04"):

1. **Features.** SIFT keypoints on undistorted coordinates.
2. **Opening.** Until a map exists, the system tracks orientation only. It builds its first map from two views
   only when the geometry supports one: a homography/fundamental model check, with refusal under low parallax or
   ambiguity (after ORB-SLAM).
3. **Tracking against the map.** The rotation is predicted from two views and the translation from constant
   velocity. The pose is then solved against map points and accepted as a full pose with at least 30 inliers and a
   residual-consistency test.
4. **When the map is not enough.** A rotation-only state tracks orientation; it never writes to the map. Relocalization
   against the map uses EPnP. The first frame after a recording pause is labelled lost in the rotation-only state.
5. **Mapping.** Only on frames posed against the map: interval keyframes, with new points from a per-feature depth
   filter (after SVO), promoted at convergence.
6. **Local bundle adjustment.** It runs over the recent keyframes with outlier rejection. Only cameras with enough
   support are freed.

Loop closure is **off**. The original loop closer was repaired but accepts wrong loops on a dense map, and its
redesign was deferred (`theory/deferred-designs.md`, design 9).

---

## 3. Results

From the verifier's records of the final reference run and of the test sequence's single scoring (kept in the
private working repository). All numbers and their definitions are in
[`theory/expected-performance.md`](theory/expected-performance.md).

| | Development sequence | Test sequence (scored once) |
|---|---:|---:|
| Full-pose coverage (floor: 80 % of the same tracker on a perfect map) | **0.22** (floor 0.61) | **0.13** (floor 0.52) |
| Longest stretch without a full pose (frames) | 1,296 | 2,007 |
| Frame-to-frame rotation errors over 10° on posed pairs | 0 of 470 | 1 of 358 |
| Map-point depth, median relative error | 0.067 | 0.291 |
| Posed frames wrong (> 0.10 m or 2°, orientation-anchored alignment) | 0 of 483 | 370 of 372 |
| Reruns byte-identical | yes (3) | yes (2) |

**The four faults:**

| Fault | Verdict |
|---|---|
| Coverage (the precondition) | **Not met** on either sequence: the robot drives out of the small area it mapped, and nothing can extend the map from outside it |
| 1. Jumps | **Unavailable as a verdict** (coverage below the floor). None where it posed on the development sequence |
| 2. Map depth | **Unavailable as a verdict.** Accurate where mapped on the development sequence. On the test sequence the map's scale changed about 4 times |
| 3. Points under rotation | **Met as the bar was written**: nothing is mapped in the rotation-only state. But points made while turning in place *and still tracking* are worse, and on the test sequence the scale change followed them |
| 4. Revisit | **Not claimed** (scoped out) |

Why each verdict reads as it does: [`theory/closing-report.md`](theory/closing-report.md), section 4.

**For comparison** (not a ranking): ORB-SLAM3 monocular covers both sequences in 3–8 separate maps, each with its
own scale. With ground-truth depth (RGB-D mode) it holds one map for most of the development sequence, so the
difficulty is monocular geometry, not the scene (closing report, sections 5 and 6).

---

## 4. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned set: requirements.lock (adds evo + pytest)

DS=path/to/rgbd_dataset_freiburg2_pioneer_slam3

# The final configuration (R04)
python main.py --no_display --dataset $DS/rgb --default_k \
  --undistort --front_end sift --tracking rotation --init twoview \
  --live_map on --map_rot_state r52 --pnp_interval 0 \
  --no_loop_closure --no_relocalization \
  --point_source depth_filter --df_promote_factor 1.0 \
  --ba_outlier_reject on --ba_jump_gate off --ba_free_cameras supported \
  --residual_test on --map_rot_gap lost \
  --ground_truth $DS/groundtruth.txt --timestamp_file $DS/rgb.txt \
  --save_estimated est.txt --log_file run.log

# Score the path
evo_ape tum $DS/groundtruth.txt est.txt -va --align --correct_scale
```

- **Run the final configuration with every flag above.** Without them, `main.py` runs the *original*
  frame-to-frame chain (its defaults are kept so that old results reproduce). `--no_relocalization` turns off the
  chain's relocalizer; the map tracker's own relocalization stays on.
- `--default_k` uses the TUM freiburg2 intrinsics.
- A full run takes about 10 minutes on a laptop.
- Leave out `--no_display` for the 3D viewer, which needs Pangolin built from source.
- Unit tests: `pip install -r requirements.lock`, then `pytest tests/`. The two derivative checks the tests call are
  in `system/verify/checks/`.
- The project's own scorers (coverage, labels, J, map depth) belong to the evaluation harness, which is not part of
  this repository. Path accuracy is scored with evo as above.
- The ORB-SLAM3 reference runs are reproducible with
  `REF_DATASET_DIR=$DS bash scripts/reference_orbslam3.sh`.

---

## 5. How it was built

- **Staged contracts.** Every change went through a stage contract whose checks and thresholds were fixed *before*
  any code was written. A verifier script computed pass or fail.
- **Standing checks.** Finite-difference checks of the bundle-adjustment and pose-graph derivatives, and a
  map-consistency check, re-ran on every stage.
- **The record.** 45 contracts between 2026-09-17 and 2026-10-11: 33 verified, 12 failed and replanned.
- **The roles.** I built this with AI agents (Claude models, through Claude Code) inside a verification loop I
  designed. Three agents planned, reviewed and built each stage; two scripts verified and sequenced it; I ruled on
  every decision of substance. See [`WORKFLOW.md`](WORKFLOW.md) and the closing report, section 9.
- **What the original code got wrong.** See [`DIAGNOSIS.md`](DIAGNOSIS.md),
  [`PROJECT_EVALUATION.md`](PROJECT_EVALUATION.md) and the closing report, section 2. In short: wrong derivatives,
  an inflated success counter, inconsistent map tables, a loop closer seeded from the drifted pose, near-180°
  orientation flips, ignored lens distortion, and a unit-step chain whose error floor is 0.91 m even with perfect
  rotations and directions.

---

## 6. Repository map

| Path | What |
|---|---|
| `main.py`, `pipeline.py` | Entry point and per-frame loop (both the original chain and the final system) |
| `opening.py`, `initializer.py`, `rotation_only.py` | Start-up: orientation-only tracking and the two-view initializer |
| `map_tracker.py`, `map_rotation.py`, `residual_test.py`, `npnp.py` | Tracking against the map, the rotation-only state, the acceptance test, a Newton-PnP study solver |
| `local_mapping.py`, `depth_filter.py`, `epipolar_refine.py` | Keyframe mapping and the depth-filter point source |
| `optimization.py` | Bundle adjustment and pose graph (local-perturbation chart) |
| `front_end.py`, `camera.py`, `dataset.py` | Features, camera model and undistortion, dataset reading |
| `frame.py`, `tracking.py`, `mapping.py`, `localization.py`, `loop_closure.py` | The original chain's parts (kept, off in the final configuration) |
| `Map.py`, `Point.py`, `geometry.py`, `reprojection.py`, `map_export.py`, `pose_log.py` | Data structures, helpers, exports |
| `oracle.py` | Ground-truth injection for the oracle ladder (off by default; checked not to leak) |
| `visualizer.py`, `run_ui.py`, `run_playback_ui.py`, `scripts/` | Viewer, GUIs, evaluation helpers |
| `tests/` | Unit tests |
| `system/verify/checks/` | The two finite-difference derivative checks (bundle adjustment, pose graph) |
| `theory/closing-report.md` | The full account: what was done, why, and what was found |
| `theory/expected-performance.md` | Every reference number and its definition |
| `theory/deferred-designs.md` | Designs that were not built, each with the prediction it would have to meet |
| `DIAGNOSIS.md`, `PROJECT_EVALUATION.md` | What was wrong with the original code |
| `WORKFLOW.md` | How the work was run: the agent loop and its checks |

---

## License

See [`LICENSE`](LICENSE).
