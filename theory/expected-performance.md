# Expected performance

Every reference number of the project and its definition. Numbers were recorded here before each stage was dispatched, so that its thresholds were set before its results; nothing here was filled from the stage being assessed.

## Primary sequence: TUM fr2 pioneer_slam3

Known properties: robot-mounted camera, long sequence (2544 frames), low texture, sustained near-static and rotation segments, one revisit near frames 640 to 900.

`[paper: orb-slam-2015.pdf §VIII.B, Table III]` ORB-SLAM reports 16 TUM RGB-D sequences (keyframe RMSE 0.30–3.81 cm after 7 DoF alignment) and states it discarded sequences "not suitable for pure monocular SLAM systems, as they contain strong rotations, no texture or no motion". **No fr2/pioneer sequence appears in Table III.** `[reasoning]` pioneer_slam3 matches the discard criteria; no published monocular result for it is known to the workshop. Thresholds for this sequence must come from this harness, and the acceptance bar should be behavioural (no jumps, no map under pure rotation, a loop closure that measurably corrects a synthetic drift) rather than a centimetre target.

## Reproduced baseline on this machine

Filled by stage 02 and re-measured by stage 03b: the frozen reference is **R03b** (`#reference-run-r03b-stage-03`), which replaced R02 (`#reference-run-r02`) as the comparator on 2026-09-18 after the orientation-flip fix; R02 stays recorded as the pre-fix reference; the pre-foundation full-sequence numbers under `#harness-measurements-on-this-machine` are context only. Historical Windows numbers (not reproduced here, not comparable): Sim(3) APE RMSE 1.346 m, median 1.160 m, max 3.862 m; without scale correction 16.35 m.

## Repeatability and per-metric tolerances (stage 02)

Three plain reruns of the frozen reference, nothing varied (`design.md#experimental-method` rule 2). `[reasoning]` The pipeline is expected to be deterministic (legacy OpenCV RANSAC with a fixed-seed local RNG; no other randomness), so the reruns are expected to be byte-identical; if so, record that as the result. Do not vary algorithms to manufacture a nonzero floor. Tolerances are per metric, each combining the observed repeat range with a pre-declared practical tolerance, fixed before stage 03 reports:

**Result (stage 02, verified 2026-09-18 02:32; tag `stage-02-reference-and-repeatability-verified`, commit 887b8df).** `[measured: runs/02-reference-and-repeatability/attempt-1/verifier.json]` Three plain reruns of the frozen reference on the full 2544-frame sequence were **byte-identical**: `repeat_identical = 1`, `repeat_runs_completed = 3`, and every range metric (`repeat_range_ape_m`, `repeat_range_rpe_t_1f_m`, `repeat_range_rpe_t_1s_m`, `repeat_range_rpe_rot_1f_deg`) exactly 0. The pipeline is deterministic under the reference configuration, as `design.md#experimental-method` rule 2 predicted. The tolerances below are therefore **practical margins, not noise** (human ruling 2026-09-18, record `workshop/09-pre-03-rulings.md`); each is about 3 to 5 percent of the R02 value.

| Metric | Repeat range (3 reruns) `[measured: R02]` | R02 value `[measured: R02]` | Practical tolerance (human, 2026-09-18) | τ used in the decision rule |
|---|---|---|---|---|
| APE RMSE, Sim(3)-aligned (m) | 0 | 1.477 | 0.05 m | τ_APE = 0.05 m |
| RPE translation, Δ = 1 frame, RMSE (m) | 0 | 0.0521 | 0.003 m | τ_RPE1 = 0.003 m |
| RPE translation, Δ = 1 s (30 frames), RMSE (m) | 0 | 0.402 | 0.02 m | τ_RPE1s = 0.02 m |
| RPE rotation, Δ = 1 frame, **median** (deg) | 0 | 0.363 | 0.05 deg | τ_rot = 0.05 deg |

τ_rot is defined on the **median** rotation RPE at 1 frame, not the RMSE: `[measured: R02]` the RMSE is 54.6 deg against a 0.363 deg median, dominated by a few near-180-degree orientation flips (`#reference-run-r02`, finding), and is unusable for comparison until they are fixed. Three samples give a range, not a spread that supports statistical claims.

### Reference run R02

Stage 02 (`runs/02-reference-and-repeatability/`, attempt 1, verified 2026-09-18 02:32, verifier PASS on all 14 checks; frozen configuration: product defaults, `--default_k`, `--use_relative_scale` off, `pnp_interval` 10; resolved settings saved next to the trajectory). All numbers `[measured: runs/02-reference-and-repeatability/attempt-1/verifier.json]`. **R02 is the comparator from now on**; the pre-foundation 1.751 m (`#harness-measurements-on-this-machine`, same harness, old code) is context only.

| Quantity | R02 (O0, frozen reference) | S1 (unit-step chain from ground truth) |
|---|---|---|
| Frames | 2544 estimated | 2544 (1666 translating / 305 near-stationary / 543 stationary / 29 no truth; 27 interpolated) |
| APE RMSE, Sim(3)-aligned (m) | 1.477 (median 1.436, max 2.638) | 0.907 (median 0.638, max 1.893); **1.040** under the alternative undefined-direction choice |
| Sim(3) scale | 0.052 | 0.0072 |
| RPE translation, Δ = 1 frame, RMSE (m) | 0.0521 (median 0.0514) | 0.0116 (alternative choice 0.0122) |
| RPE translation, Δ = 1 s = 30 frames, RMSE (m) | 0.402 | 0.161 |
| RPE rotation, Δ = 1 frame (deg) | RMSE **54.6**, median **0.363** | RMSE 0.0047 |
| RPE rotation, Δ = 1 s (deg) | RMSE 127.5 | RMSE 0.0051 |
| Max step (map units) | 2.67 (median 1.0, p99 1.16) | – |
| PnP accepted / failed | 26 / 228 | – |
| BA completed (counter after stage 01) | 54 (23 success, 31 fail, as the log counts them) | – |
| Loop detections; VO failures (held) | 0; 2 | – |
| Run time (s) | 107 | – |

S1 conventions `[measured: system/verify/checks/oracle_s1.py docstring]`: relative rotation and translation direction from GT_i⁻¹·GT_{i+1}, every step forced to unit length; on stationary frames (ground-truth step < 1 mm, direction undefined) the primary choice is a **zero step** (rotation only); the alternative choice repeats the previous frame's direction and is what the 1.040 m figure reports; near-stationary is < 3 mm. `[reasoning]` S1 is the structural floor of any unit-step chain given perfect epipolar estimates: scale unobservability alone costs about 0.9 m APE on this sequence, and the class threshold moves it at the 0.1 m level.

**Gap convention (ruled 2026-09-18, later; `design.md#open-questions`, "no jumps" follow-up).** The R02 and S1 numbers above, including every Δ = 1 frame RPE, **include every consecutive pair that has ground truth**; they are not recomputed. From H(J) onward, J and the Δ = 1 frame translation RPE exclude consecutive `rgb.txt` pairs whose timestamp gap exceeds 150 ms (`[measured: workshop 2026-09-18, rgb.txt]` 5 pause pairs on this sequence: frames 1694, 1695, 1720, 2485, 2490, gaps 0.27 to 1.71 s; the 570 dropped-frame pairs at 55–110 ms stay in), report the excluded pairs' J as a labelled diagnostic (count over J_max, max), and S1 reports the same count; Δ = 1 s RPE is unaffected; tracking loss across any gap appears in the supported-coverage report with the preceding gap length. Context for J_max `[measured: workshop 2026-09-18, groundtruth.txt at rgb.txt timestamps, gaps ≤ 45 ms]`: true per-frame rotation median 0.57 deg, p99 2.13 deg, max 7.05 deg (adopted by me as the context figure in place of the brief's 1.6 deg).

Relative-scale estimator audit (one extra run with `--use_relative_scale`, scored against ground-truth baselines) `[measured: same verifier.json]`: 2543 frames logged (2541 essential-matrix); the estimator **fired on 7.08 percent** of frames (180); 159 scored after excluding 20 stationary frames; **median relative error 3.66, p90 11.4**; that run's APE RMSE 1.599 m, Sim(3) scale 0.021 (worse than the reference's 1.477 m). `[reasoning]` The estimator is not a usable scale source; this closes the no-veto branch of `design.md#oracle-ladder-pre-registered-decision-rule` and is part of the evidence for the 2026-09-18 architecture ruling.

**Finding for the record (rotation flips).** `[measured: R02]` Rotation RPE at 1 frame: RMSE 54.6 deg, median 0.363 deg; at 1 s: RMSE 127.5 deg. `[reasoning]` A median of a third of a degree with an RMSE of fifty-plus means a small number of frames carry near-180-degree orientation flips in the estimated trajectory. This is a defect in the current pipeline, to be located and fixed by a contract before or inside stage 03 (the orchestrator decides; `design.md#proposed-stage-ordering`, 03 row); the oracle ladder must not have to look through it, and O0 is re-measured after the fix. Until then rotation comparisons use the median only (τ_rot). Frames with a rotation RPE above J_max = 10 deg (`design.md#open-questions`) also mean O0 fails the no-jumps bar today.

Standing checks on R02 `[measured: same verifier.json]`: `fd_jacobian_ba` rotation block relative error 6.9e-11, `fd_jacobian_posegraph` 1.2e-10, `map_invariants` 0 violations, `loop_pnp_unseeded` 1, BA accounting identity holds (attempted 4, skipped 2, completed 2, accepted 2 in the probe).

### Reference run R03b (stage 03)

**Camera model caveat (added 2026-09-20).** R02, R03b, the ladder and every depth table below were measured on raw distorted pixels with `--default_k` (`conventions.md#camera-model-and-distortion`). They remain the reference numbers of that configuration. 04d re-ran O0, O2, O3 and the direction breakdown with undistortion on (`#r03b-u-and-ladder-u-04d-re-verification`, 2026-09-20): the 03c direction finding is **re-verified** (34.0° median, 31.3 percent reversed on translating frames); the architecture ruling stands.

Stage 03b (`runs/03b-essential-hypothesis-selection/`, attempt 1, verified 2026-09-18; tag `stage-03b-essential-hypothesis-selection-verified`, commit d092166): the essential-matrix hypothesis is selected by a cutoff-free cheirality vote (`cv2.recoverPose` retired). Same frozen configuration as R02; three plain reruns byte-identical. **R03b is the comparator from 2026-09-18**; every 03c rung ran on it. All numbers `[measured: runs/03b-essential-hypothesis-selection/attempt-1/verifier.json]`.

| Quantity | R02 | **R03b** |
|---|---|---|
| APE RMSE, Sim(3)-aligned (m) | 1.477 | **1.404** (median 1.133, max 3.153; scale 0.0171) |
| RPE translation, Δ = 1 frame, RMSE (m), all pairs with truth | 0.0521 | 0.0203 (median 0.0165) |
| RPE translation, Δ = 1 frame, RMSE (m), 150 ms gap exclusion, online / final | – | 0.0187 / 0.0167 |
| RPE translation, Δ = 1 s, RMSE (m) | 0.402 | 0.208 |
| RPE rotation, Δ = 1 frame, RMSE / median (deg) | 54.6 / 0.363 | **5.36 / 0.313** |
| RPE rotation, Δ = 1 s, RMSE (deg) | 127.5 | 28.4 |
| J online: pairs over J_max = 10 deg, of 2507 evaluated (5 excluded by gap, 31 without truth) | 236 | **11** (max 150.3, p99 5.72, median 0.314; all 11 sourced from the two-view step; excluded pairs: 1 over J_max, max 24.0) |
| J final | – | 11 (max 150.3, p99 5.38, median 0.311) |
| Supported coverage; held frames | 0.9992; 2 | 0.9992; 2 (both after a gap) |
| PnP accepted / failed | 26 / 228 | 33 / 221 |
| BA completed / accepted / rejected | 54 | 53 / 33 / 20 |
| Loop detections | 0 | 0 |
| Map points at end of run | – | 13,473 (89,657 observations) |

**What the flip fix was** `[measured: runs/03a-online-stream-and-jump-instrument/attempt-2/artifacts/jumps/]`: of R02's 236 pairs over J_max, 225 were exact half-turns, each within one frame of a two-view estimate whose own rotation was about 180 deg; on every such frame the decomposition chosen by `recoverPose` had zero points passing its 50-unit far-point cutoff, while the cutoff-free vote identified the right hypothesis (frame 3: 386 votes against 4). The 11 that remain are genuine two-view failures on frames with 6 to 114 matches and are the front-end's own. `[reasoning]` The zero-count pass condition is the finished system's bar (stage 04), not this repair's.

**The oracle ladder** (stage 03c, `runs/03c-oracle-pose-rungs/attempt-1/artifacts/oracle_ladder/ladder.json`; injection at the two-view step in the camera frame, s = 331.5 map units per metre fixed before any frame was processed from the first consecutive pair with a true step of at least 3 mm, which on this sequence is frames 243–244 (`[measured: oracle.py:209–222; 04b O3+ [ORACLE-SETUP] scale_pair=243,244 gt_step_m=0.003017]`; the first 243 frames are the opening rotation in place, max true step 2.1 mm; corrected 2026-09-19 evening, the earlier text said frames 1–2); O4 at PnP frames only) is tabulated in the oracle-ablation table below (section "Oracle ablation"). Readings, for the record `[reasoning unless marked]`:

- O3nd reproduces truth to 3.5 mm: the injector's frame and sign are right; every other row is trustworthy.
- O2z = S1 (0.907 m): the in-pipeline rung and the zero-code synthesis agree; the unit-step floor is confirmed by two routes.
- **Veto A fired** as the pre-registered rule predicted: RPE-t 1 frame (O0) − (O1) = 0.0005 m < τ_RPE1 = 0.003 m, and O1's APE is worse (1.641 vs 1.404). Supplying true step lengths to the chain buys nothing.
- Veto C not interpretable: O0 has 11 pairs over J_max (the rule's precondition). Recorded: J(O1) = 11, J(O3) = 0. Veto B unavailable (c < c_min, ruled).
- **O4 measured the PnP safety gate, not the correction**: 251 true absolute poses offered, 1 accepted. The pipeline as built cannot use a correct correction; the gate is retired in stage 04.
- **The chain's defect is translation direction, not scale and not rotation.** `[measured: [ORACLE] lines, all injected frames]` estimator rotation error median 0.313 deg; direction error median 50.8 deg. `[workshop measurement, 2026-09-19, same lines, 1,664 translating frames]` true step median 9.1 mm (p10 5 mm, p90 19 mm); rotation error median 0.41 deg, p99 7.9 deg; direction error median 39 deg, p75 103 deg, p90 162 deg; **31 percent of steps point backwards** (error > 90 deg); 10 percent within 10 deg; sign-folded median 22 deg, flat from 3 mm to 40 cm steps. `[reasoning]` At f ≈ 520 px a 9 mm baseline against a 2–3 m scene gives about 2 px of disparity, the size of keypoint noise, so the direction and its sign are near-unobservable per pair while rotation (about 5 px per half degree, global) stays observable. A motion-regime statement (forward motion at about 0.25 m/s at 30 fps in a room-sized scene), not a property of this recording.
- **Map points are the product of wrong motion** `[measured: ladder.json map_points]`: O0 13,473, O1 12,648, O2 357, O2z 317, O3 320. Point creation is driven by rotation and direction being estimated rather than true, not by scale. `[reasoning, from mapping.py:529–569, not instrumented]` The gates (1.5 deg ray parallax, 40-baseline depth cap) are keyframe-pair constants applied to consecutive frames: under true motion a real point sits at about 280 baselines with 0.2 deg parallax and fails both; under wrong motion the DLT lands the point near the cameras with fake parallax and passes. Loosening the gates is not a fix (2 px disparity gives about 50 percent depth error per pixel of noise); the point source changes in stage 04-1.

**Depth** (stage 03e, `runs/03e-map-export-and-depth-metric-redispatch/attempt-1/verifier.json`; protocol (a)+(c) as ruled; conventions: nearest depth frame within 20 ms, PNG/5000 m, valid 0.3–10 m, nearest pixel, `DEFAULT_K`, no distortion, per landmark, p99 tail, (c) median over observing keyframes first):

| Rung | Landmarks | Valid obs | (a) retained / coverage | (a) median signed rel. error | (a) median abs / p99 | (c) keyframes | (c) landmarks | (c) median dev / p99 | (c) local-scale median; p90/p10 |
|---|---|---|---|---|---|---|---|---|---|
| O0 | 13,473 | 36,080 (254 frames without depth; 89 behind camera) | 7,609 / 0.56 | **−0.907** | 0.909 / 0.991 | 34 | 1,549 | 0.084 / 13.6 | 0.131; **9.2** |
| O3 | 320 | 551 (186 behind camera) | 273 / 0.85 | −0.978 | 0.978 / 0.998 | 1 | 12 | 0.237 / 19.6 | 0.092; 1.0 (one keyframe) |

`[reasoning]` Map points sit at about a tenth of their true depth at the trajectory's scale on O0 and at 2 percent on O3; the local scale varies nine-fold across O0's keyframes, so no single map scale exists. Since O3's trajectory is correct to 1.2 cm, this is the mapping side; it is 04-1's bar to beat.

### R03b-u and ladder-u (04d re-verification)

`[measured: runs/harness/ladder-u/REPORT.md, ladder.json; runs/harness/ladder-u-off/ladder.json; run by hand 2026-09-20 on main at 04d verified, full sequence, --default_k]` The same three rungs with the `--undistort` switch off and on. The switch-off runs are on today's tree (the 04a–04d keyframe-pair mapper), so they are the like-for-like comparator; 03c's own numbers were the pre-04 per-frame mapper's.

| Quantity | Registered tolerance (`design.md#04-1-after-04c-the-mechanism-and-the-route`) | Off (raw, today's tree) | On (undistorted) | Verdict |
|---|---|---|---|---|
| Two-view translation-direction error, median over translating frames (`[ORACLE]` lines, 1664 frames) | within 2° of 39.4° | 39.4° (= 03c) | **34.0°** | outside tolerance by 5.4°, in the improving direction |
| same, median over all frames with truth (ladder.json `dir_err_median_deg`, includes near-stationary) | – | 50.8° | 44.6° | recorded |
| Reversed fraction (> 90°) | within 2 points of 0.309 | 0.309 | 0.313 | within |
| Within 10° fraction | – | 0.104 | 0.112 | recorded |
| Rotation error median / p99 (deg) | within τ_rot = 0.05 on the median | 0.410 / 7.91 | 0.393 / 7.74 | within |
| O0 APE (m) | within τ_APE = 0.05 | 1.4403 | 1.4506 | within (+0.010) |
| O2 APE (m) | – | 0.707 | 0.715 | unit-step floor unchanged |
| O0 J online count over J_max / max (deg) | – | 11 / 150 | 10 / 177 | recorded |
| Depth (a) medians within 0.02 of −0.907 / −0.978 | (those were the pre-04 mapper's; not comparable) | O0 −0.630 (04d verifier) | O3+ −0.277 (04d, `und_`) | not comparable; see `#milestone-04-1-on-o3-plus` |
| **O3 as built** (PnP, BA, loop closure on) APE (m) | – | 0.356 | 0.293 | **not 03c's O3 (0.0117)**: see below |

**O3 as built is no longer 03c's O3** `[measured: ladder-u/o3/summary.txt]`: PnP accepted 0, BA accepted 3, **loop-closure detections 3 (off) / 1 (on)**; the one on-run detection (frame 1200 ↔ 660, 11 PnP inliers) issues a `[CORRECTION]` touching 539 frames with max rotation 176.6° and max translation 199.9 map units (0.6 m), and the perfect trajectory ends 0.29 m off. `[reasoning]` The 04 mapper's map is dense enough for the loop detector and BA as built to act on it; 03c's per-frame map never triggered them. Comparators for 04-1 stay O3nd (0.0035 m) and O3+ (04a–04d: 0.003–0.023 m). This is a finding about loop closure and BA acting on a wrong-depth map, filed for 04-4 (the M0 gap table) and 05 (geometric verification of a loop: 11 inliers must not move a trajectory by 0.3 m), not about the lens.

## Smoke subset

The verifier's `smoke` profile uses the first N frames (N in `system/config.yaml`). Record the baseline smoke APE here once measured; smoke numbers are only comparable to other smoke numbers.

## Hold-out sequence

*(Status 2026-10-11: scored once on 2026-10-11; see `#the-hold-out-scored-once-fr2pioneer_slam-04-ho-verified-2026-10-11`.)*

Ruled 2026-09-17 (`design.md#goal`): **fr2/pioneer_slam** (same robot, same fault class). On disk since 2026-09-17 as `eval.holdout_root` in `system/config.yaml` `[measured: 2921 rgb, 2890 depth frames, same layout as the primary]`; a check selects it with `VSLAM_DATASET=holdout`, `eval_profile` alone never scores it, score once after the component stages. fr1/desk is an optional external anchor only: `[paper: orb-slam-2015.pdf Table III]` reports a monocular keyframe RMSE for it, so a number here can be read against a published one; it does not decide acceptance.

## Harness measurements on this machine (2026-09-08, NumPy 2.5.3, OpenCV 4.11, SciPy 1.17, evo 1.37)

**Pre-foundation code; context only since 2026-09-18.** R02 (below) is the comparator.

Smoke profile (first 300 frames, product defaults + `--default_k`): Sim(3) APE RMSE 0.101 m, median 0.068 m, max 0.313 m, Sim(3) scale 0.0085, run time 16 s. PnP 3/29 accepted, BA counter 7 success / 2 fail but only 4 completed solves, 0 loop detections. Smoke numbers are comparable only with other smoke numbers.

Full sequence, current `main` (commit 10256c0 = Stage 2 + Stage 3 code, product defaults + `--default_k`, no `--local_map_window`): Sim(3) APE RMSE **1.751 m**, median 1.249 m, max 3.315 m, scale 0.041, 346 s. PnP 27/254 accepted (10.6%), BA counter 62/22 but only 47 completed solves, loop detections 0, VO fails 2 (held, max step 3.09). Artefacts: `runs/harness/full/`.

Full sequence, pre-remediation code (commit 78c7041, the tree the historical Stage 0 was measured on), same machine and environment: Sim(3) APE RMSE **1.738 m**, median 1.470 m, max 5.122 m, scale 0.047, 346 s. PnP 25/254, BA counter 58/26 with 49 completed solves, loop detections 0, VO fails 2 with the origin teleport (max step 85.9). Artefacts: `runs/harness/stage0-code-78c7041/`.

### What the two full runs say together

- Same machine, pre-remediation vs current main: 1.738 m vs 1.751 m. The Stage 2 and 3 code at product defaults has no measurable effect on APE; it removes the origin teleport (max step 85.9 to 3.1) and nothing else.
- Same code, Windows vs this machine: 1.346 m vs 1.738 m. The environment (OpenCV 4.11 here, unknown there; feature detection and RANSAC nondeterminism) moves the number by about 0.4 m. **Historical Windows numbers are not comparable to anything measured here.** Anchor all thresholds to numbers from this harness.
- Loop detections: 0 on both runs here, 1 on Windows. The single historical loop closure was environment-dependent.
- Run-to-run noise floor on this machine is still unmeasured; do that (three repeats of one configuration) before setting any threshold tighter than about 0.1 m.

## Oracle ablation (adopted 2026-09-17, `design.md#adopted-ideas`)

O0 and S1 filled by stage 02 (2026-09-18); O0 re-measured as R03b and O1–O4 filled by stage 03 (2026-09-18, recorded 2026-09-19, `#reference-run-r03b-stage-03`); O5 superseded by O3+ (design ruling 2026-09-19); the map-ladder rungs O3+, M1a, M1b, K1, M0 (`design.md#oracle-ladder-rung-definitions`) are filled by stage 04. Rung definitions, per-rung protocol and the pre-registered decision rule are in `design.md#dependencies-and-decision-points`; since the 2026-09-18 ruling (architecture decided: 04a) the rule is a written prediction that these numbers confirm or embarrass, and this table only records numbers. RPE is always reported with its Δ, after the same Sim(3) alignment as APE, translation and rotation separately; rotation at 1 frame as RMSE and median (comparisons use the median, τ_rot = 0.05 deg, until the R02 flips are fixed); supported coverage (`design.md`, T5) is recorded for every rung, and for O3 both the full-trajectory APE and the APE at R's common timestamps. J (rotation RPE at 1 frame on the online stream: count of frames over J_max = 10 deg, max, p99, median; pass = count zero) is recorded for every in-pipeline rung once H(J) exists, over the evaluated pairs after the 150 ms gap exclusion (`#reference-run-r02`, gap convention), with the 5 excluded pairs' J as a separate diagnostic. Depth (O3, O5) follows the protocol ruled 2026-09-18: **(a)** accuracy at the trajectory's Sim(3) scale and **(c)** ground-truth-free consistency at the map's own local scale, two columns, never merged, each as median / tail (p99) with the count of retained landmarks, per-landmark aggregation (under (c): median over a landmark's observing keyframes first), scored on independent observations only.

| Rung | Supported coverage | APE RMSE (m) | APE at R's timestamps (m) | RPE-t Δ=1 frame (m) | RPE-t Δ=1 s (m) | RPE-rot Δ=1 frame (deg) RMSE / median | J: count > 10 deg / max / p99 / median (online) | Depth (a) at Sim(3) scale: median / tail; n | Depth (c) local consistency: median / tail; n | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| O0 reference **R03b** (R02 row kept below) | 0.9992; 2 held `[measured: R03b]` | **1.404** (median 1.133, max 3.153; scale 0.0171) | not computed (B unavailable) | 0.0203 (median 0.0165) | 0.208 | **5.36 / 0.313** (1 s: 28.4 RMSE) | **11** / 150.3 / 5.72 / 0.314 | −0.907 / 0.991; n 7,609 (cov 0.56) | 0.084 / 13.6; n 1,549; local scale p90/p10 9.2 | comparator since 2026-09-18; PnP 33/221, BA 33/20, 13,473 map points |
| O0 reference (R02, pre-flip-fix) | 2544 of 2544 frames emitted; 2 held (VO fail) `[measured: R02]` | **1.477** (median 1.436, max 2.638; scale 0.052) `[measured: R02]` | (to score at R's timestamps in 03) | 0.0521 (median 0.0514) | 0.402 (30 frames) | **54.6 / 0.363** (1 s: 127.5 RMSE) | (H(J) pending; count is nonzero by the RMSE/median gap) | – | – | frozen reference, three reruns identical; pre-foundation was 1.751 (context only); carries the orientation-flip defect, re-measure after its fix |
| S1 | – (synthesized) | **0.907** (median 0.638, max 1.893; scale 0.0072); alt 1.040 `[measured: R02]` | – | 0.0116 (alt 0.0122) | 0.161 | 0.0047 / – | – | – | – | zero-code, stage 02; zero step on 543 stationary frames (< 1 mm), 305 near-stationary (< 3 mm), 1666 translating, 29 no truth; alt = repeat previous direction |
| O1 | 0.9992 | 1.641 | – | 0.0198 | 0.193 | 5.48 / 0.311 | 11 / 150.3 / 5.69 / 0.314 | – | – | true step lengths; **veto A fires**: RPE-t 1f gap 0.0005 m < τ; 12,648 map points |
| O1 (placeholder, pre-03) | | | | | | | | – | – | feeds A, C |
| O2 | 0.9992 | 0.959 | – | 0.0122 | 0.168 | 0.031 / 0.000 | 0 / 0.000 | – | – | true rotation + direction, unit step, estimator direction on stationary frames; PnP 0/254; 357 map points |
| O2z | 0.9992 | **0.907** | – | 0.0117 | 0.161 | 0.031 / 0.000 | 0 / 0.000 | – | – | as O2, zero step on stationary frames; **equals S1** |
| O2 (placeholder, pre-03) | | | | | | | | – | – | direction; exhibits unit-step floor |
| O3 | 0.9992 | **0.0117** (median 0.0085, max 0.036) | – | 0.00116 | 0.0037 | 0.107 / 0.000 | 0 / 0.56 (final max 2.84) | −0.978 / 0.998; n 273 (cov 0.85) | 0.237 / 19.6; n 12; 1 keyframe | full true relative motion; PnP 0/254, BA 1/3; **320 map points**; downstream not destructive, but barely acted |
| O3nd | 0.9992 | 0.0035 | – | 0.00043 | 0.0022 | 0.031 / 0.000 | 0 / 0.000 | – | – | O3 with PnP, BA, loop closure, relocalization off: injector convention check |
| O3 (placeholder, pre-03) | | | | | | | | | | feeds B (unavailable on this sequence, `#goal` bar substitutes), C; injected vs corrected logged |
| O4 | 0.9992 | 1.417 | – | 0.0201 | 0.209 | 5.36 / 0.309 | 11 / 150.3 / 5.65 / 0.310 | – | – | true absolute pose at PnP frames: 251 injected, **1 accepted** by the gate; BA 32/21; measures the gate, not the correction |
| O4 (placeholder, pre-03) | | | | | | | | – | – | not a veto |
| O5 | – | – | – | – | – | – | – | – | – | **superseded by O3+** (2026-09-19); never run |
| **O3+** (default arm, 04b = 04c) | 1,382 exported (2,454 created, 1,072 culled) | 8,529 (254 frames without depth; 0 behind camera) | 1,149 / 0.83 | **−0.422** | 0.444 / 0.821 | 62 | 1,072 | 0.140 / 1.62 | 0.00412; **3.24** |
| M1a, M1b, K1, M0 | | | | | | | | | | stage 04 map ladder, `design.md#oracle-ladder-rung-definitions` |
| O5 (placeholder, pre-03) | | | | | | | | | | not a veto; independent observations |
| R (reference system) | `[measured: runs/harness/reference-orbslam3/summary.txt]` keyframe spans cover 63.5 / 78.2 / 86.2 s of 111.4 s (57 to 77 percent; 3 runs); nothing before frame 382; c < c_min = 0.8 | 0.039 / 0.057 / 0.277 (largest map per run, keyframes only, each map Sim(3)-aligned on its own) | (same set) | – | (waits on H(RPE) at shared timestamps) | | – | – | – | see next section: 3 to 6 disconnected maps per run, 0 loop closures; veto B unavailable (ruled 2026-09-18) |

Relative-scale estimator (no-veto branch, `design.md`) `[measured: R02]`: firing rate across frames = **7.08 percent** (180 of 2543); relative error vs ground-truth baseline on firing frames = **median 3.66, p90 11.4** (159 scored); APE with it on 1.599 m vs 1.477 m off (prior `[measured: runs/stage-2/REPORT.md]` ≈ 8.8% "ok" was the same order). Branch closed: no usable scale source on the chain.

## Milestone 04-1 on O3 plus

Rung O3+ = `--oracle o3 --pnp_interval 0 --no_loop_closure` (relocalization kept). Three contracts, all on the default keyframe-pair arm; 04b and 04c run the same mapper (04c adds a provenance line only) and reproduce each other exactly `[measured: runs/04a, 04b, 04c attempt-1 verifier.json]`.

| Contract | Neighbourhood | Created / culled / exported | Pair baseline p50 | (a) n / coverage | (a) signed p50 | (a) abs p50 / p99 | (c) local scale p90/p10 | O3+ APE | Sub-mm pairs | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 04a | covisibility only (never forms: every keyframe 1 neighbour) | 869 / 712 / 157 | 0.175 m | 147 / 0.94 | −0.536 | 0.536 / 0.859 | 3.25 | 0.0034 m | 2 | FAIL (counts, depth) |
| 04b | + recent-keyframe fill, widest first, pair-level parallax skip | 2,454 / 1,072 / 1,382 | 0.511 m | 1,149 / 0.83 | −0.422 | 0.444 / 0.821 | 3.24 | 0.0077 m | 0 | FAIL (2000-point floor, depth 0.122, spread 1.5) |
| 04c | as 04b, provenance line | 2,454 / 1,072 / 1,382 | 0.511 m | 1,149 / 0.83 | −0.422 | 0.444 / 0.821 | 3.24 | 0.0077 m | 0 | PASS (diagnostic; no depth bar) |
| 04d switch off | = 04c (byte-identical O0 and O3+ trajectories) | 2,454 / 1,072 / 1,382 | 0.511 m | 1,149 / 0.83 | −0.422 | 0.444 / 0.821 | 3.24 | 0.0077 m | 0 | PASS (foundation; no depth bar) |
| **04d switch on** (`--undistort`) | as 04b, keypoints undistorted at extraction | 1,853 / 900 / 953 | 0.657 m | 827 / 0.87 | **−0.277** | **0.321** / 0.807 | **1.86** | 0.0228 m | 0 | recorded; BA accepted 3 of 25 |
| **04e** (`--undistort --lm_confirm`, third-view χ² confirmation) | as 04d; a point is finite only after a χ² match in a further neighbour at ≥ 30 % different disparity | 472 landmarks (98 of 128 keyframes create nothing) | – | – | – | **0.249** (abs median) | **2.12** | – | 0 | FAIL (arm's finding: precision, not matches); attribution correct 0.76, mismatch 0.02, creation-depth ratio 0.82 `[measured: runs/04e attempt-1 verifier.json; orchestrator.json]` |
| **04g = 04h, reference arm** (`--undistort --point_source depth_filter --df_promote_factor 1.0`, SVO depth filter, promote at first convergence) | seeds at every unlinked keypoint of each keyframe, one epipolar descriptor measurement per later frame, Gaussian + uniform mixture; retired five keyframes after seeding | 3,271 promoted / 1,105 exported (35 of 127 keyframes promote nothing) | 0.406 m | 959 / 0.87 | **−0.188** | **0.225** / 0.835 | **1.758** (59 kf) | 0.0424 m (BA accepted 6) | 0 | FAIL on both bars; the best map measured. 04h reproduced 04g byte for byte (`same_build_*_identical = 1`) `[measured: runs/04h attempt-1 verifier.json dfarm_reference_*]` |
| **04g = 04h, deferred arm** (`--df_promote_factor 9.5`: promote at θ_acc ≥ 9.5·θ_min = 10.45°) | as above | 184 promoted / 98 exported (102 keyframes promote nothing); 3,909 seeds retired while converged | 0.707 m | 91 / 0.93 | −0.145 | 0.164 / 0.959 | 1.569 (17 kf) | 0.0054 m (BA accepted 1) | 0 | FAIL on both bars and **below the 100-landmark instrument floors**: not a rung measurement; starved by the five-keyframe seed lifetime (`workshop/19`) `[measured: dfarm_deferred_*]` |

Creation attribution (04c; each creating match classified against the depth image transferred through the ground-truth relative pose, classes at 4σ_L / 12σ_L with σ_L = 1.2^level) `[measured: runs/04c attempt-1 verifier.json attr_o3plus_*]`:

| Quantity | Value |
|---|---|
| provenance lines / scored / unscored (no depth pixel, no truth) | 2,454 / 1,360 / 1,074 + 20 |
| correct / mismatch / ambiguous | 0.351 / 0.153 / 0.496 |
| match error px p50 / p90 | 9.9 / 20.7 |
| creation depth signed p50: all / correct / mismatch | −0.503 / −0.135 / −0.593 |
| correct by parallax bin 1.1–1.5 / 1.5–2 / 2–3 / 3+ deg (n) | −0.101 (229) / −0.138 (116) / −0.198 (97) / −0.103 (35) |
| parallax ratio (mapper / true) p50, correct; true parallax p50, correct | 1.135; 1.42° |
| survivors correct fraction (scored n) / culled correct fraction (scored n) | 0.442 (785) / 0.226 (575) |
| hamming p50 correct / mismatch; hamming2 p50 correct / mismatch | 31 / 34; 53 / 53 |
| created keypoint pyramid level: median (new / partner); share at levels 6–7 | 3 / 3; 18 % |

The same instrument on the 04d switch-on run (`[measured: 04d verifier.json und_attr_o3plus_*]`, mapper and instrument both undistorted): scored 999 of 1,853; correct / mismatch / ambiguous **0.390 / 0.151 / 0.458** (registered prediction ≈ 0.53 / 0.11 / 0.36: missed; the offline re-scoring had corrected the instrument on the *same* raw-pixel matches, whereas the undistorted mapper makes different matches on wider pairs); match error px p50 / p90 9.5 / 22.4; creation depth signed p50 all / correct / mismatch −0.374 / **−0.104** / −0.550 (prediction "toward −0.10": met); parallax ratio 1.098 at true parallax 1.57°; survivors correct 0.550 (493 scored) vs culled 0.235; hamming2 p50 50 / 51; 86 of 128 keyframes created nothing (65 before) and the median created-pair baseline rose from 0.51 to 0.66 m.

Human-side follow-up `[measured: runs/04c/attempt-1/analysis/along_line_bias.md, not a verifier record]`: transfer error cross-line p50 1.9 px (99 % within 4σ); along-line p50 −9.4 px raw / −4.9 px undistorted, p10 −18.5, p90 +2.3, two thirds toward the epipole; undistorting both keypoints (fr2 k1..k3) moves the error p50 from 9.9 to 6.2 px and the split to 0.53 / 0.11 / 0.36; points within 100 px of the centre still ~10 px off; 40 % of created points under 10 px of disparity (the 1.1° gate ≈ 10 px at f = 521); along-line shift ≈ −4 px below 20 px of disparity and ≈ −11 px at 20–80 px; a mocap clock offset does not explain it (bias crosses zero near −70 ms while the cross-line error grows); mapper creation depth ratio z·s / z_meas p50 0.50 against ≈ 0.68 implied by the matched pixels alone.

**SIFT front-end pre-test (2026-09-22, outside the loop; `workshop/20`; `[measured: workshop pre-test, not a verifier record]`).** Same keyframe-pair mapper, same rung (O3+ `--undistort`), full sequence; only `--method SIFT` plus an L2 descriptor bound rate-matched to ORB's Hamming ≤ 50 (248). ORB rows are the 04d verifier records.

| Quantity | ORB (04d) | SIFT d ≤ 248 | SIFT d ≤ 150 |
|---|---:|---:|---:|
| (a) abs median / signed median | 0.321 / −0.277 | **0.184 / −0.062** | 0.184 / −0.096 |
| (c) spread (keyframes) | 1.86 | **1.63** (72) | 1.94 (39) |
| exported / created / culled | 953 / 1,853 / 900 | 1,051 / 2,627 / 1,574 | 607 / 1,513 / 906 |
| match error px p50 / p90 | 9.51 / 22.4 | 7.03 / 22.8 | 6.65 / 22.0 |
| attribution correct / mismatch / ambiguous | 0.390 / 0.151 / 0.458 | 0.314 / 0.310 / 0.376 | 0.341 / 0.297 / 0.362 |
| creation depth signed p50, correct class | −0.104 | **+0.035** | +0.001 |
| parallax ratio (mapper / true), correct class | 1.098 | **0.983** | 1.000 |
| O3+ APE RMSE (comparator 0.0535) | 0.0228 | 0.0764 (10 relocalization corrections) | – |

Reading in `workshop/20`: SIFT removes the one-signed creation bias (parallax inflation 1.098 → 0.983; correct-class creation depth −10.4 % → +3.5 %), the first intervention to move the signed error since undistortion, and doubles the mismatch class (0.151 → 0.310), which a tighter descriptor bound does not repair. Instrument caveat: `creation_attribution.py`'s `level_n=(\d+)` cannot parse SIFT's octave −1 (1,529 of 2,627 creations), so these numbers come from a workshop copy with the regex widened, validated against the official check on an ORB run; widening it is a task-H item before any SIFT stage.

**04j / 04j redispatch, SIFT front end (verifier records, 2026-09-22/23)** `[measured: runs/04j-sift-front-end-redispatch attempt-1 verifier.json fe_*]`, O3+ `--undistort --front_end sift` (frozen protocol: L2 bound 248, ratio tests 0.766 / 0.876):

| Arm | (a) median / signed | (c) (keyframes) | landmarks | mismatch | BA corrections | APE | fused |
|---|---|---|---:|---:|---:|---:|---:|
| pair | 0.197 / −0.087 | 1.62 (71) | 1,054 | 0.305 | 8 | 0.0758 | 1 |
| pair + `--lm_confirm` | 0.192 / −0.050 | 1.59 (34) | 511 | 0.083 | 0 | 0.00004 | 0 |
| depth filter, paper timing | **0.144 / −0.062** | **1.53** (91) | 2,229 | – | 33 | 0.0744 | 9 |
| depth filter, `--no_ba` (S7 diagnostic) | 0.169 / −0.107 | 1.56 (95) | 2,006 | – | 0 | 0.00004 | 9 |

Bars 0.122 / 1.5 (R4): no arm clears; SIFT first-order figure m_fo = 0.0954 recorded. S6 reading met by the depth-filter arm. S3 predictions 5 of 6.

**04k, epipolar refinement on SIFT (verifier record, 2026-09-23)** `[measured: runs/04k-epipolar-refinement attempt-1 verifier.json rf_*]`: pair + `--epi_refine` (a) 0.186 / signed −0.083 / (c) 1.70 on 924 landmarks, mismatch 0.263, APE 0.054 m; depth filter + `--epi_refine` 0.149 / −0.049 / 1.63 on 2,084, APE 0.088 m; with `--no_ba` 0.169 / −0.098 / 1.57 on 1,897. Along-line offset (correct class) −0.364 px refined vs −0.315 unrefined; creation depth +0.025 vs +0.021.

**Milestone 04-1 closed 2026-09-23 (`workshop/21`):** adopted point source SIFT + depth filter at the paper timing, no refinement — **(a) 0.144 with BA / 0.169 without, (c) 1.53 / 1.56**, against the bars 0.122 / 1.5 (recorded shortfall).

Depth-filter arm, promotion attributes and per-landmark error `[measured: runs/04h attempt-1, dfarm_* and [DF-PROMOTE] lines]` and `[measured: workshop analysis 2026-09-21, theory/workshop/19-analysis-perlm.py on the 04h artefacts; not a verifier record]`:

| Quantity | Reference (k = 1) | Deferred (k = 9.5) |
|---|---:|---:|
| θ_max read at the posterior p50 / θ_acc p50 | 1.68° / 1.76° | 4.99° / 10.69° |
| n_meas p50 / inlier ratio p50 / reference level p50 | 15 / 0.57 / 1 | 24.5 / 0.69 / 1 |
| time reference → first point p50 / p90 (frames) | 41 / 77 | 65.5 / 87.7 |
| signed (a) at promotion (z_ref vs depth at the reference keypoint), n | −0.241 (526) | −0.101 (60) |
| per-landmark final − promotion p50 (p10 / p90) | −0.019 (−0.063 / +0.012) | −0.028 (−0.075 / +0.009) |
| final: > 5 % too close / within 5 % / > 5 % too far | 0.70 / 0.14 / 0.16 | 0.86 / 0.10 / 0.04 |
| first-order true parallax at the widest inlier view, p50; fraction below θ_min | 1.31°; 0.38 | 2.57°; 0.00 |
| promoted at e < −0.30: read θ_max / true θ / n_meas (n) | 1.71° / 0.93° / 21 (199) | 4.54° / 2.20° / 31 (21) |
| promoted within ±0.10: read θ_max / true θ / n_meas (n) | 1.64° / 1.60° / 33 (128) | 3.63° / 3.48° / 44 (29) |
| final abs p50 by frames to promotion: 0–20 / 40–60 / 80–100 | 0.428 / 0.225 / 0.146 | – / 0.393 / 0.199 (small n) |
| final abs p50 by point depth: < 2.5 m / 2.5–4 m / 4–6 m / ≥ 6 m | 0.31 / **0.394** / 0.192 / 0.124 (signed +0.077) | 0.105 / **0.474** / 0.218 / 0.194 |
| final abs p50 keeping inlier ratio ≥ 0.7 (share of map kept) | 0.174 (0.24) | 0.132 (0.77); ≥ 0.8: 0.103 (0.26) |

Reading in `workshop/19`: the error is admitted at promotion; a relative convergence test (σ_ρ/μ and θ read at the posterior) passes too-close seeds with less true parallax than θ_min, so first-convergence promotion is biased too close by selection; deferral dilutes it; the five-keyframe lifetime cuts off the population (≥ 80 frames, ≥ 80 measurements) that approaches the bar. The 04g/04h numbers are the pinned baseline for the proposed 04i lifetime follow-up.

`[reasoning]` Reading and route in `design.md#04-1-after-04c-the-mechanism-and-the-route`: the map is shallow because the matcher's along-line error (5–10 px after undistortion: repeat pitch, coarse levels) is half the disparity the gate admits, and the gate plus cheirality keep only the sign that makes points closer; distortion adds a one-signed 2–4 % on top. 04d confirms both parts: undistortion alone took the (a) median from −0.42 to −0.28 and the (c) spread from 3.24 to 1.86, and left the matcher's error scale (9.5 px p50) and the ambiguous half (46 percent) in place, which is what 04e and 04f are for. The chain-era O3 row (320 points at −0.978) and the 04d row bracket what 04e–04f must beat: the bar is 0.122 on (a) and 1.5 on the (c) spread.

## 04-H perfect map (build only, 2026-09-23)

`[measured: [harness] 7816e2a, perfect_map_score.py --rung m1b --build-only, full primary; per-frame counts computed by the workshop from the same build's map file]` SIFT front end, `--undistort`, keyframe interval 20, depth-edge rejection and the widened M1b window of that commit.

| Quantity | Value |
|---|---:|
| keyframes / points | 115 / 21,927 |
| points per keyframe p10 / p50 / p90 | 75 / 163 / 334 |
| valid-depth fraction of keyframe keypoints | 0.344 |
| dropped at depth edges | 9,340 |
| truth floor: reprojection of map points into ground-truth-posed frames, p50 / p90 | 1.682 / 4.102 px (n 184,365) |
| M1b correspondences (C lines) | 191,874 |
| per frame p1 / p5 / p10 / p50 | 0 / 3 / 10 / 54 |
| frames with < 30 / < 40 / < 60 correspondences (of 2,543) | 691 / 987 / 1,351 |
| longest run under 30 | 96 frames, from frame 2105 |

The 30-correspondence count defines the map-supported frames of `design.md` G4. The truth floor confirms the ≈ 1.7 px estimate of `workshop/21`.

## Newton-PnP study, part A (04o, 2026-09-25)

Synthetic only; the solver is not in the product. Solver: the project's NPnP variant (D1b-1, barrier on the 14-dimensional face). Record: `workshop/29-04o-outcome-npnp-part-a.md`. Sources: `[measured: runs/04o-npnp-synthetic-reproduction/attempt-1/verifier.json]` for the bars; `[measured: theory/workshop/29-04o-reading.py on that attempt's per-instance CSVs]` for the D1b-4a reading.

| Bar (D1a as amended by D1b) | Value | Eligible |
|---|---|---|
| `npa_gamma_disagree` | 0 | 26,492 |
| `npa_cost_exceeds_sdp_certified` | 0 | 26,483 (0.983 of 26,950) |
| `npa_gap_inconsistent` (max error) | 0 (1.9e-19) | all converged |
| `npa_bf_certified_exceeds` | 0 | 24,250 (share 1.0) |
| converged / certified | 26,950 / 26,950 | |
| oracle not optimal (`npa_sdp_failed`) | 458 | |

Median rotation error (deg), in-front poses (D1b-4a):

| Regime | NPnP | EPnP | EPnP + LM | SQPnP |
|---|---|---|---|---|
| R1 SQPnP protocol | 0.304 | 0.350 | 0.285 | 0.301 |
| R2 noise sweep | 0.123 | 0.142 | 0.096 | 0.120 |
| R3 near-planar | 0.087 | 0.187 | 0.120 | 0.098 |
| R4 this hall | 0.243 | 0.271 | 0.201 | 0.244 |

Gross failure rate (behind the camera off the planar tie, or rotation error > 10°):

| Cell | NPnP | SQPnP | EPnP | EPnP + LM |
|---|---|---|---|---|
| R1, n = 4 | 1.7 % | 1.0 % | 26.5 % | 16.7 % |
| R1, n ≥ 5 (18,000) | 0 (0 %) | 2 (0.01 %) | 14 (0.08 %) | 10 (0.06 %) |
| R3, ρ = 0 | 0 % | 0 % | 39.8 % | 37.8 % |
| R3, ρ = 0.001 | 19.8 % | 0 % | 22.0 % | 20.9 % |
| R3, ρ = 0.01 | 0.7 % | 0 % | 1.3 % | 1.1 % |
| R4 hall, n = 6 | 1.7 % | 0.5 % | 2.3 % | 1.7 % |
| R4 hall, n > 6 | 0.2 % | 0.1 % | 0.1 % | 0.1 % |

**Known defect in the 04o record (found 2026-09-25 by the 04p harness session, commit 6079602):** `npnp_study_a.py`'s `ml_pose` passes a view of the instance's true translation to `cv2.solvePnPRefineLM`, which refines in place. In regime 1 this runs after the reference solvers are graded and before NPnP, so **04o's regime-1 NPnP translation errors (`npnp_trans_err_m`, and any aggregate of it) are measured against the ML translation, not the truth**. Rotations, costs, behind-camera flags, every bar and the D1b-4a reading here (rotation only) are unaffected. Part B passes copies.

Cost, recorded and never a speed claim: Newton iterations median 24–30 (max 78), flat in n = 4…300; Python 19–31 ms per solve; Clarabel 8–13 ms; OpenCV EPnP 0.12–0.18 ms, SQPnP 0.05–0.11 ms.

## Newton-PnP study, part B (04p, 2026-09-25)

Real correspondences without matching errors (M1b: real SIFT pixels, depth-image points, true poses). 10,406 instances: 2,365 full frames (n 6–420) and one subset per frame at n = 6, 10, 20, 50. Truth floor (input reprojection RMSE under the true pose): p50 2.51 px, p90 4.55 px. Record: `workshop/31-04p-outcome-npnp-part-b.md`. Sources: `[measured: runs/04p-npnp-real-correspondences/attempt-1/verifier.json]` for the bars; `[measured: theory/workshop/31-04p-reading.py]` for the reading.

Bars (part A's, carried by D1c-6): all met. γ disagreement 0 / 10,328; cost excess over the certified oracle 0; gap inconsistency 0; certified above the brute force 0 / 6,991. **NPnP certified on 100 % of real sets in every class (H4).** The oracle was not optimal on 78.

Median rotation error (deg), in-front poses (ML = `solvePnPRefineLM` from the true pose, the floor):

| Class | NPnP | SQPnP | EPnP | EPnP + LM | ML |
|---|---|---|---|---|---|
| n = 6 | 0.639 | 0.650 | 0.716 | 0.611 | 0.559 |
| n = 10 | 0.486 | 0.488 | 0.524 | 0.437 | 0.430 |
| n = 20 | 0.355 | 0.356 | 0.389 | 0.315 | 0.315 |
| n = 50 | 0.268 | 0.269 | 0.300 | 0.228 | 0.228 |
| full frames | 0.308 | 0.307 | 0.332 | 0.271 | 0.269 |

Gross failure rate (behind the camera, or rotation error > 10°):

| Class | NPnP | SQPnP | EPnP | EPnP + LM | ML |
|---|---|---|---|---|---|
| n = 6 | 3.20 % | 2.09 % | 4.05 % | 3.37 % | 0.47 % |
| n = 10 | 0.26 % | 0.22 % | 1.27 % | 0.70 % | 0.04 % |
| n = 20 | 0 % | 0 % | 0.19 % | 0.19 % | 0 % |
| n = 50 | 0 % | 0 % | 0 % | 0 % | 0 % |
| full frames | 0.25 % | 0.21 % | 0.42 % | 0.42 % | 0.04 % |
| all sets with n ≥ 20 (5,534) | 0 | 0 | 7 | 7 | 0 |

NPnP versus SQPnP: median |difference| of rotation error 0.004–0.009°, each better on about half. NPnP behind the camera: 34 instances (31 at n = 6), all the line cost's true optimum, none at n ≥ 20. Newton iterations: median 28, max 77. Python time: 19 ms per solve.

## Newton-PnP study, part C (04q, 2026-09-26)

The three solvers are in the system. 04m's relocalization is run in shadow mode on every tracked frame and logged, never used, plus one real run per arm. RANSAC hypotheses are the same in every arm; the arm's solver refits the consensus set, then motion-only BA and the guided search run as in 04m. Record: `workshop/33-04q-outcome-npnp-part-c.md`. Sources: `[measured: runs/04q-npnp-relocalization-solver/attempt-1/verifier.json]` for the verdict; `[measured: theory/workshop/33-04q-reading.py]` for the reading.

**Verdict (D1c-5 (2), pre-registered): 0 on M1a (deciding) and on M1b. EPnP stays.**

| M1a shadow, 2,078 events | EPnP | SQPnP | NPnP |
|---|---|---|---|
| Wrong accepted (> 0.10 m or > 2°) | 143 | 144 | 141 |
| Correct accepted | 1,570 | 1,567 | 1,565 |
| Paired wrong difference vs EPnP, 95 % block interval (43 blocks of 60 frames) | – | +0.05 pp [−0.10, +0.21] | −0.10 pp [−0.30, +0.09] |
| Paired correct difference, lower bound (guard 1 at −1 pp) | – | −0.54 pp | −0.64 pp |
| Cheirality rejections (shadow / real) | 3 / 2 | 0 / 0 | 0 / 0 |
| Refit translation error p50 (before BA) | 15.1 mm | 11.5 mm | 11.5 mm |
| Refit collapses > 5° | 21 | 15 | 15 |
| Final rotation error p50 (after BA) | 0.361° | 0.361° | 0.361° |
| Refit time p50 (Python) | 0.03 ms | 0.12 ms | 21 ms |

M1b (1,661 events): 25 wrong in every arm, all on the same frames. Refit translation p50 11.4 / 8.4 / 8.4 mm; final rotation p50 0.241° in every arm.

Mechanism:
- The arms are wrong on the same frames: 138 of EPnP's 143 on M1a, 25 of 25 on M1b.
- The final pose is bit-identical across arms on about 90 % of frames.
- M1a's wrong acceptances have consensus sets that fit the true pose badly (input RMSE p50 5.45 px against 2.41 px for correct ones).
- Wrong acceptances concentrate at a small image footprint: 86 of 143 (M1a) and 22 of 25 (M1b) are in the lowest hull-share tercile.

NPnP: converged and certified on 4,504 of 4,504 solves; gaps −1.9e-8 to −1.1e-8 ‖M‖_F (A5-2 reading: none below −1.4e-6). Newton iterations p50 28, p90 34–38, max 71.

## Pre-map rotation-only mode (04r, verified 2026-09-26)

Milestone 04-3a, first contract. `--tracking rotation` solves each frame's orientation against the current panorama keyframe using the pre-tested estimator: a 2-point RANSAC followed by an SO(3) least-squares fit on bearings. A new keyframe is taken when the 4×3-grid view coverage drops below 0.75. Translation is held at zero, and no points are made. The run uses the full primary sequence (SIFT, undistorted) with every correction switched off. Orientation is scored in the frame-0 gauge (`conventions.md#orientation-scoring`). Sources: `[measured: runs/04r-premap-rotation-only-mode-redispatch/attempt-1/verifier.json]`. Attempt 1 of 04r gave the same figures and the same trajectory sha256; it failed only on a carried check that could not pass. Record: `workshop/36`.

| Quantity | 04r | Chain era (o0, same scoring) | Bar |
|---|---|---|---|
| e_60 / e_120 / e_220 / e_381 | 1.01 / 1.85 / 3.47 / 4.47° | 4.14 / 10.15 / 7.09 / 20.22° | e_120, e_220 below chain: **met** |
| e_220 as a share of the true turn (117.17°) | 2.96 % | 6.05 % | within the 2–4 % truth floor |
| J over 10°, frames 0–381 | 0 of 373 pairs (max 4.95°, p99 3.33°) | – | 0: **met** |
| Availability, frames 1–220 / 1–381 / all | 1.000 / 0.990 / 0.987 | – | ≥ 0.90 on 1–220: **met** |
| Finite map points | 0 | – | 0: **met** |
| Panorama keyframes, 0–220 / 0–381 / all (frame 0 included) | 6 / 10 / 52 | – | recorded |
| Per-solve error over 0–220, p50 / p90 / max | 0.45 / 0.75 / 1.02° | – | recorded (pre-test S1: 0.40 / 0.81 / 1.04°) |
| Inliers over 0–220, p10 / p50 | 97 / 166 | – | recorded |
| True camera-centre displacement absorbed as rotation, p50 / max | 13.6 / 57.9 mm | – | recorded |
| Longest unsupported run | 24 frames (507–530, reason=inliers) | – | recorded |
| J over 10° after frame 381 | 6 (frames 1697, 2309–2312, 2492), max 25.6° | – | recorded |
| Orientation change error, frames 1090–1170 (04-3b's turn in place) | 0.66° | – | recorded |
| Run time, full sequence | 287 s | – | – |

`[reasoning]` Over the opening turn, the mode more than halves the chain's accumulated orientation error at frame 220 and cuts it more than fivefold at frame 120. The error at frame 220 is within the ground truth's own floor. Past the opening, the mode holds orientation with no map at 0.99 availability, but it has no answer to sustained translation (frames 501–530) and accepts a few wrong solves (the 6 jumps). Both are the initializer's and 04-4's concern, not this mode's.

## Two-view initializer module (04s, verified 2026-09-27)

Milestone 04-3a, second theme, first of two contracts. `initializer.py` is ORB-SLAM3's two-view H/F initializer. At σ = 1 it matches the source exactly: 0 relative error on S_H, S_F and R_H. On top of that it applies the design's publication rules (P1): one match per feature, a full BA with the first camera fixed, then a re-check of the BA'd geometry against θ_min = 1.1°, 2σ reprojection and cheirality, median depth 1, at least 50 points, and at least 2 rows and 2 columns of the 4×3 grid. It is **not wired in**. Everything below is synthetic: seven regimes of 100 scenes each. Source: `[measured: runs/04s-twoview-initializer-module/attempt-1/verifier.json]`. On every per-regime figure, the product equals the harness's to-the-letter scratch build (`workshop/37` addenda 1 and 3). Record: `workshop/38`.

| Regime | Accepted | Main refusal | H chosen | Rot. err p50 / max | Transl. dir. err p50 / max | Wrong | Outliers published | Published points with true parallax < 1.1° |
|---|---|---|---|---|---|---|---|---|
| gen | 0.91 | few_good 9 | 0.00 | 0.057 / 0.39° | 0.32 / 1.54° | 0 | 20 | 51 of 19,457 |
| plane | 0.10 | ambiguous 88 | 0.98 | 0.19 / 0.32° | 2.38 / 3.39° | 0 | 0 | 0 of 2,335 |
| rot (bar: 0) | 0.00 | ambiguous 100 | 1.00 | – | – | 0 | 0 | – |
| low (true parallax p50 0.50°) | 0.00 | parallax 64, points 14, ambiguous 12, few_good 10 | 0.00 | – | – | 0 | 0 | – |
| conc (bar: 0) | 0.00 | few_good 76, support 24 | 0.00 | – | – | 0 | 0 | – |
| dup | 0.89 | few_good 11 | 0.00 | 0.065 / 0.50° | 0.28 / 3.53° | 0 | 18 | 33 of 18,963 |
| sig (σ ∈ {1, 2, 4}) | 0.68 | few_good 32 | 0.00 | 0.16 / 0.66° | 0.85 / 5.51° | 0 | 16 | 85 of 11,929 |

Every published-point invariant is zero over all 700 scenes. The median-depth error is 2.2e-16. The benign interface probe (10°, 0.5 m) was refused as few_good: 176 good points out of 205, against ⌊0.9 × 205⌋ = 184. Run time was 84 s for 700 scenes.

`[reasoning]` What this says for the real opening:
- (1) Wrong acceptances are 0 in every regime. A refusal costs time, not correctness.
- (2) Two refusal modes will set the acceptance delay: few_good, from the unrefit 8-point F (`papers/orb-slam.md#initialization-what-the-paper-leaves-to-the-implementation`), and ambiguous, the planar twofold ambiguity. The second is what the paper intends, and the robot's floor-facing opening is exactly that geometry.
- (3) Two views publish outliers whose random pixel falls within 2σ of the epipolar line, at about 1 per 1,000 points. Later views must remove them.
- (4) The one-signed "too close" selection bias that `conventions.md#map-ladder-conventions` predicts is not tested here, because no low-parallax scene was accepted.

## Initializer refit arm (04t, verified 2026-09-28)

Milestone 04-3a, initializer theme, second of three contracts (ruling R37-2). `initialize(..., refit=True)` fits the selected H or F once more on all of its inliers, after model selection and before decomposition. The default `refit=False` reproduces 04s byte for byte. The switch is **not wired in**. The data are the same seven synthetic regimes of 100 scenes as 04s. Source: `[measured: runs/04t-initializer-refit-arm/attempt-1/verifier.json]`, commit 7582da5. Every per-regime figure equals the task H scratch build (b0ac9f7, `workshop/40` addendum 1), so the product is the scratch build.

| Regime | Accepted off → refit | Main refusal, refit | Rot. err p50 / max, refit (off p50) | Transl. dir. err p50 / max, refit (off p50) | Wrong | Outliers published off → refit | Published points with true parallax < 1.1°, refit |
|---|---|---|---|---|---|---|---|
| gen | 0.91 → 1.00 | none | 0.041 / 0.22° (0.057°) | 0.22 / 0.85° (0.32°) | 0 | 20 → 24 | 45 of 22,926 |
| plane | 0.10 → 0.08 | ambiguous 90 | 0.20 / 0.52° (0.19°) | 2.28 / 3.48° (2.38°) | 0 | 0 → 0 | 0 of 1,903 |
| rot (bar: 0) | 0 → 0 | ambiguous 100 | – | – | 0 | 0 | – |
| low | 0 → 0 | parallax 95, points 4 | – | – | 0 | 0 | – |
| conc (bar: 0) | 0 → 0 | support 65, few_good 35 | – | – | 0 | 0 | – |
| dup | 0.89 → 0.99 | few_good 1 | 0.040 / 0.25° | 0.22 / 1.03° (0.28°) | 0 | 18 → 20 | 31 of 22,715 |
| sig (σ ∈ {1, 2, 4}) | 0.68 → 0.91 | few_good 9 | 0.12 / 0.63° (0.16°) | 0.54 / 2.70° (0.85°) | 0 | 16 → 30 | 80 of 17,905 |

Paired over scenes: legitimate acceptances 258 → 298 (gen +9, dup +10, sig +23, plane −2; no other regime moves). Wrong acceptances 0 → 0. The synthetic half of the R37-2 adoption rule reads guard 1, gain 1 (`initcmp_syn_rule = 1`), **recorded, not an adoption**: the real-opening half and the hold-out confirmation are still to come. The benign probe (10°, 0.5 m) is refused off and accepted with the refit; pure rotation is refused in both. Every published-point invariant is zero. Run time 102 s for 700 scenes (84 s off).

`[reasoning]` What this adds for the real opening:
- (1) The few_good refusals of 04s were almost all the unrefit 8-point F (gen 9 → 0, dup 11 → 1, sig 32 → 9). The refit also tightens pose accuracy on the scenes it accepts (median rotation error about −25 %, translation direction about −30 %).
- (2) The refit does not touch the planar twofold ambiguity. It makes it slightly stricter (88 → 90 refused), because a better H also sharpens the competing Faugeras hypothesis. On a floor-facing opening, the refit is not expected to shorten a wait caused by `ambiguous` (R37-1).
- (3) Where the refit shifts a refusal, the refusal stays safe: conc moves from few_good to the spatial-support gate (24 → 65), and low moves from mixed reasons to parallax (64 → 95).
- (4) Epipolar-consistent outliers roughly double in sig (16 → 30, about 1.7 per 1,000 published points). A cleaner F lets more of them through. That is a survival-check load, not a wrong acceptance, and R37-2's guard does not count it.

## Opening wiring (04u, verified 2026-10-01)

Milestone 04-3a, third and last contract. `--tracking rotation --init twoview` (off by default) runs the 04r rotation-only mode on every frame and tries the 04s initializer between the newest frame and the last two candidates. The candidates are the panorama keyframes plus the R43-1 renewal frames. An accepted map is handed to the 04m map tracker (M1a path, EPnP relocalization). A map lost within 10 tracked frames is a false start: it is reset, and rotation-only tracking resumes. One surviving map per run. Run 1: refit off live, with the refit computed in shadow. Run 2: refit live. Full primary sequence, `--front_end sift --undistort`. `[measured: runs/04u-initializer-opening-wiring/attempt-2, checks/opening_init_run_off.log, opening_init_run_refit.log, opening_init_compare.log]`

| Quantity | Run 1 (refit off) | Run 2 (refit live) | Bar |
|---|---|---|---|
| Prefix identical to 04r up to the first acceptance | 1 | 1 | barred, run 1: **met** |
| e_120 / e_220 minus chain era | −8.30 / −3.63° | same | < 0, run 1: **met** |
| J over 10°, frames 0–381 / whole run | 0 of 373 / 0 (max 4.95° / 8.91°) | same | 0 on the opening: **met** |
| Turn availability (frames 1–220) | 1.000 | 1.000 | floor: **met** |
| Maps accepted under rotation (true parallax) / wrong maps | 0 / 0 | 0 / 0 | 0: **met** |
| Published-point invariant violations | 0 | 0 | 0: **met** |
| Initializer attempts / ambiguous / too few matches / parallax / few_good | 1,133 / 699 / 351 / 51 / 30 | 1,134 / 671 / 356 / 69 / 36 | recorded |
| H selected among attempts that reached selection | 0.826 | 0.824 | recorded |
| Renewal frames (R43-1) | 33 | 33 | recorded |
| First acceptance (a, b) / false starts / surviving map at | (571, 588) / 1 / 605 | (571, 593) / 0 / 593 | recorded; prediction (1′) "221–450" **missed** |
| Delay against the reference's first surviving map (382) | +223 frames | +211 frames | recorded |
| Surviving map: points / true baseline / true rotation | 55 / 0.093 m / 6.4° | 50 / 0.199 m / 14.1° | recorded |
| Surviving map: rotation err / translation-direction err | 0.20° / 1.79° | 0.66° / 2.87° | recorded |
| Surviving map: depth relerr p50 / true parallax p50 / points below θ_min | 0.032 / 1.69° / 10 | 0.036 / 1.99° / 2 | recorded |
| Survival inliers p50 / min (acceptance 30) | 40 / 33 | 37 / 32 | recorded |
| Frames tracked after the map / lost | 21 / 1,924 | 20 / 1,930 | recorded |
| Available share, whole run / longest unsupported run | 0.231 / 1,924 | 0.229 / 1,929 | recorded |
| Hand-off jump | 0.26° | 0.20° | recorded |
| Near-gate reading (R40-1), d_par [interval] | −999 (4 far points) | −0.16° [−0.27, +0.07] | recorded; inconclusive |
| Run time | 361 s | 328 s | – |

R37-2 real-opening reading (R41-1, R41-2): over 1,133 paired attempts, accepted off 2 and refit 1, gained 0 and lost 1, wrong 0 and 0, false starts 1 and 0; guard 1, gain 0, survival 1, so **`opencmp_real_rule` = 0**. Closed by R43-2: the refit is not adopted and stays in the code, off.

`[reasoning]` What this says:
- (1) The opening now behaves as the design requires. It builds no map during the turn in place and accepts no wrong map, and its first map is accurate. Orientation through the turn is 04r's, unchanged.
- (2) The first map is late: frame 605 against the reference's 382. The mechanism is in `workshop/43` addendum 2. On this robot every homography selection has R_H in (0.45, 0.50], so the paper's 0.45 sends floor-dominated driving views to H, which then refuses them as ambiguous. This is R37-1's registered risk, accepted under R43-3. `rh_select` stays 0.45; generalized GRIC is second in line after 04-4.
- (3) Without a mapper the two-view map carries the run for about 20 frames and is then lost for good, which gives an available share of 0.23. That number is **04-4's** comparator: growing the map is the mapper's job. It is not a 04-3b quantity, because 04-3b runs on M1a.

## Rotation-only state inside the map (04v, verified 2026-10-02)

Milestone 04-3b, a single contract. `--map_rot_state on` (off by default): when the M1a map tracker loses its pose, the 04r rotation-only fit is measured against a panorama keyframe at the last tracked frame. The position is bridged with the tracker's constant body twist and labelled "bridged" up to age 30, then "rotation". The state exits only through tracking or EPnP relocalization at 30 inliers. Map: the pinned perfect map with every keyframe inside a ground-truth turn in place withheld (B1: six intervals; 22 keyframes, 4,363 points removed; 93 keyframes kept). Full primary sequence. `[measured: runs/04v-rotation-state-in-map/attempt-1, artifacts/rotation_state_run_off, rotation_state_run_on, rotation_state_compare]`

The verified trajectories are byte-identical to the harness's dry run on its scratch build (`workshop/49`), and every metric except run time is equal.

| Quantity | Off | On | Bar |
|---|---|---|---|
| Clean turn 1095–1199 (60.0°): entry u / re-engagement r / exit path | – / 1177 / reloc | 1120 / 1177 / reloc (same pose: 0.026 m, 0.74°) | entry inside the turn: **met** |
| e_end (orientation change 1095 → 1199) / chain era | 0.564° / 10.581° | 0.891° / 10.581° | on below chain: **met** |
| J over 10° in W (1095–1177) / max | 0 | 0 / 0.634° | 0: **met** |
| Available share in W / state frames | 0.305 / 0 | 1.000 / 58 | floors: **met** |
| Labels at c0 / c1 | tracked / lost | tracked / bridged (age 10) | recorded |
| **In-stint rotation error 1120–1176, p50 / max** (R49-4) | 1.17° / 1.70° (04m prediction) | **0.25° / 0.52°** | recorded |
| **In-stint position error 1120–1176, p50 / max** (R49-4) | 0.048 / 0.056 m (held) | **0.350 / 0.668 m** (twist) | recorded |
| Re-entry discontinuity, clean turn | 0.069 m | 0.679 m | recorded; prediction (5) **missed** |
| Whole run: re-entry discontinuity p50 / p90 / max | – | 0.089 / 0.882 / 6.17 m | recorded |
| Whole run: bridge error at the 30-frame horizon, p50 | – | 0.449 m | recorded (B6 pre-test: 1–2 cm) |
| Whole run: orient_frac / full_pose_frac | 0.699 / 0.699 | 0.991 / 0.722 | recorded; prediction (6) full-pose part **missed** |
| Whole run: labels tracked / bridged / rotation / lost | 1,779 / 0 / 0 / 765 | 1,836 / 433 / 252 / 23 | recorded |
| **Tracked frames above 0.10 m or 2°** (R49-4) | 183 | 207 | recorded |
| **Bridged or rotation frames above 10° rotation error** | – | 18 (stint 357–401: 42° from frame 372) | recorded |
| Wrong exits / wrong relocalizations | 12 of 53 | 19 of 63 | recorded |
| Run time | 409 s | 458 s | – |

Per interval, e_end on / off / chain: 16–254 4.15 / 9.82 / 7.32°; 1095–1199 0.89 / 0.56 / 10.58°; 1605–1649 0.55 / 0.55 / 1.49°; 1665–1719 (gaps) 2.85 / 5.31 / 7.86°; 2222–2236 (frame drops) 4.71 / 4.72 / 2.32°; 2467–2499 (gaps) 2.70 / 1.73 / 2.55°. In 2222–2236 both arms inherit a tracked pose that was wrong by 5.3° at 2221. The state's anchor carries that error, so this row is not evidence about the fit `[measured, reasoning]`.

`[reasoning]` What this says (details in `workshop/49`):
- (1) **Orientation:** the state does what it was built for. In the turn in place its rotation stays within 0.52° of the truth, about 1° better than the tracker's own prediction. The barred e_end does not show this, because both arms relocalize at 1177, before the turn ends.
- (2) **Position:** one-step body-twist bridging fails in slow motion. The twist is the difference of two noisy tracked poses and is about 14 times the true step in the turn in place. B6 is reopened (R49-2).
- (3) **Trust in the label:** one wrong fit locked in, so 18 frames are labelled orientation-supported while more than 10° wrong. The diagnosis is owed (R49-3) before 04-4, whose goal bar is zero J over supported pairs.
- (4) **Full-pose frames:** the state's predictions let tracking resume earlier (+57 tracked frames). About 24 of those frames are wrong, which is the count-only acceptance of registered risk 1.

## Fallback restructured: Pirchheim entry, held position, lost-frame velocity (04x, verified 2026-10-04)

Milestone 04-3, the respecified stage after R52. `--map_rot_state r52`:
- The rotation-only state is entered only when three things hold: the map pose fails, Pirchheim's test (α < 5°, β/γ > 0.2) fires on the last 2 s of tracked history, and the fit succeeds.
- Inside the state, the position is held at the anchor.
- Frames lost outside the state get their translation from the windowed body velocity.

Same frozen map and sequence as 04v. The off and on values reproduce 04v attempt 1 byte for byte. Verified at merge 0c600d2 (`accept_pass` on a mechanical gate). `[measured: runs/04x-fallback-structure/attempt-1, artifacts/fallback_run_r52, fallback_compare]`

| Quantity | 04v off | 04v on | 04x r52 | Bar / prediction |
|---|---|---|---|---|
| Orientation-supported frames > 10° wrong, outside gaps | 0 | 16 | **0** | 0: **met** |
| J over 10° on orientation-supported pairs outside gaps / max | – | – | **0** / 9.62° | 0: **met**. The pair after the gap, (2491, 2492), is recorded at 25.7° (R54-1) |
| Clean turn: entries / e_end / chain era | – / 0.564° / 10.581° | 1120 / 0.891° / 10.581° | 1120 and 1179 / **0.269°** / 10.581° | below chain: **met** |
| Clean turn, in-stint rotation p50 / max | 1.17° / 1.70° | 0.25° / 0.52° | 0.25° / 0.52° | recorded |
| Clean turn, in-stint position p50 / max | 0.048 / 0.056 m | 0.350 / 0.668 m | **0.028 / 0.038 m** (held) | recorded |
| Entry attempts / detected / entered | – | 63 entered | 58 / 34 / 33 | about 20 entries: **missed** |
| Stint 357–401 | – | entered, 42° locked in | not entered | not entered: **held** |
| Entered by regime (turn in place / turn and drive / straight / slow / other) | – | – | 3 / 20 / 2 / 1 / 6 (plus 1 with no regime) | recorded |
| orient_frac / full_pose_frac | 0.699 / 0.699 | 0.991 / 0.722 | 0.910 / 0.712 | full pose within 0.01 of off: **missed** (+0.013) |
| Labels tracked / rotation / lost | 1,779 / 0 / 765 | 1,836 / 685 / 23 | 1,811 / 503 / 230 | recorded |
| Tracked frames above 0.10 m or 2° | 183 | 207 | 224 | recorded |
| Wrong exits | 12 of 53 | 19 of 63 | 8 of 33 | recorded |
| Lost runs with a fitted velocity | – | – | 7 of 25 | recorded |
| Run time | 409 s | 458 s | 444 s | – |

In the 04v on column, "rotation" counts bridged and rotation frames together (433 + 252).

`[reasoning; workshop/55]` What this says:
- (1) **The entry test does what the paper defines.** Pure rotation is detected. The 22 entries made while driving are "rotation without parallax" by Pirchheim's definition: α < 5° accepts travel up to about 9% of the scene depth, and the scenes here are about 4.5 m deep, so that is about 0.39 m.
- (2) **Most wrong exits were wrong before the state started.** In 7 of the 8, the anchor (the last tracked frame) was already off by 0.10–0.54 m. Its error grew over the last 2–8 tracked frames, so it is the count-only acceptance of registered risk 1, the same cause as the 224 tracked-but-wrong frames. That goes to 04-4's residual test (R55-2).
- (3) **Held position** beats the 04v twist on the clean turn by an order of magnitude: 0.028 m against 0.350 m at p50.

## Support forecast at true poses (R46-3, workshop measurement 2026-10-04)

These are the current-build maps from 04x's standing checks, projected through the oracle-o3 truth with the map tracker's visibility tests (`workshop/56`; files only, nothing re-run). 2,514 frames have truth. Supported means at least 30 expected correspondences. For O3+ and pair, early and late are the two ends of the promotion window. `[measured: workshop/56]`

| Map (points) | S, own yield (post hoc) | Median expected | S, raw visible (upper bound) | First supported frame (own yield) |
|---|---|---|---|---|
| M1a perfect (21,927) | 0.737 (actual correspondences) | 54 | 1.000 | – |
| O3+ adopted (2,229), early / late | 0.488 / 0.452 | 28.8 / 23.6 | 0.773 / 0.692 | 620 (20.7 s) / 639 |
| Pair triangulation (1,054), early / late | 0.110 / 0.082 | 7.7 / 4.8 | 0.519 / 0.481 | 1420 / 1439 |

Own yield is the share of visible points the product's mapper actually observed at its keyframes: 0.29 for O3+, 0.23 for pair. The pre-registered yield, borrowed from M1a (0.062), is invalid because 60% of M1a's visible points are near-duplicates. The pre-registered adjusted numbers are kept in `workshop/56` and flagged.

`[reasoning]` The adopted mapper does not meet the tracker's 30-point assumption at true poses on about half the frames. The pair map meets it on about a tenth. The opening, a 9 s turn in place, has no supported frame until about 20 s. Rulings R56-1 to R56-4 were ratified 2026-10-04.

## Mapper-side step on O3+: BA outlier step and the C6 lifetime measurement (04y, verified 2026-10-04)

The first contract of milestone 04-4 (ruling C6). `--ba_outlier_reject on` adds ORB-SLAM's local-BA outlier step (χ² 5.991 on σ-normalised residuals, mid-solve and at the end; final outliers unlinked on an accepted solve). Off reproduces every verified run byte for byte, including the 04i R2 pins after the pose-vintage fix. Rung O3+, `--front_end sift --point_source depth_filter --df_promote_factor 1.0 --undistort`, full sequence. `[measured: runs/04y-mapper-side-repairs/attempt-1, checks/mapper_side_*, mapper_side_compare]`

| Arm | Landmarks | (a) abs / signed | (c) (keyframes) | APE m | BA completed / accepted / rejected | Unlinked | Time to first point p50 |
|---|---:|---|---|---:|---|---:|---:|
| age + BA, outlier step on | 2,111 | 0.137 / −0.068 | 1.41 (89) | 0.042 | 78 / 35 / 43 | 6,144 (8.9 %) | 38 |
| age + BA, outlier step off (carried 04j arm) | 2,229 | 0.144 / −0.062 | 1.53 (91) | 0.074 | 78 / 33 / 45 | – | 37 (04j) |
| R2 + BA, outlier step on | 459 | 0.0998 / −0.069 | 1.14 (46) | 0.021 | 48 / 9 / 39 | 511 (6.9 %) | 66 |
| R2, no BA | 387 | 0.115 / −0.076 | 1.17 (43) | 0.00004 | – | – | 63 |
| pair + BA, outlier step on | 1,029 | 0.192 / −0.083 | 1.71 (68) | 0.076 | 38 / 8 / 30 | 542 (9.7 %) | – |

- The C6 rule gives **age stays**: R2's (a) gain is 0.037 with better (c), but it has 459 landmarks, under the 800 floor. Predictions: R2 not adopted, met. R2 slower, met. R2 (a) in 0.126–0.133, missed (better).
- Map invariants were zero on every `[BA-OUTLIERS]` line. A 400-frame invariant run removed 80 links.
- `[measured: grep of the arms' summary.txt]` Every rolled-back solve in these five arms was rejected by the jump gate, none by the reprojection gate. The gate's absolute term (`+ 2.0` map units) is 6 mm at O3+'s scale (331.5 units/m). At the M0 initializer's scale it is about 3.2 m (0.62 units/m, measured on 04u's published pair 571/593). See `workshop/57`, R57-2 (proposed).
- `[measured: workshop/57 pre-test, gate off via --ba_max_allowed_step_factor 1e9, same arm]` With the jump gate off, all 81 solves are accepted. On thin windows (≤ 30 points), 22 of 27 leave a camera more than 0.10 m or 2° off (median move 19 cm, max 5.4 m), against 16 of 54 on thicker windows. APE 0.385 m, depth (a) 0.158, (c) 1.97, 2,899 landmarks, 218 sub-mm creation pairs.

## Supported-camera rule on O3+: local BA frees only cameras with 30 points (04z, verified 2026-10-05)

The second mapper-side step of milestone 04-4 (R57-2 (c)). `--ba_free_cameras supported` frees a window camera only when it observes at least 30 points in the BA problem. After the 04y outlier step it is tested again on inlier rows, and a camera left under 30 keeps its phase-1 pose. A solve with no free camera is skipped (`no_supported_camera`). `--ba_jump_gate off` switches the legacy gate off, and main.py refuses the rule together with the gate. At the defaults every verified run is byte-identical, including the eight `mapper_side_same_as_04y` hashes. Rung O3+, age lifetime, 04y outlier step on, full sequence. Verified at merge 6096038. Every number equals the task H dry run (`workshop/58`). `[measured: runs/04z-ba-supported-cameras/attempt-1, verifier.json: bs_support_on_*, bs_gate_off_*, bs_cmp_*]`

| Reading | gate on, no rule (04y control) | gate off, no rule | **rule on, gate off** | Bar / prediction |
|---|---:|---:|---:|---|
| BA attempted / completed / accepted | – / 78 / 35 | 84 / 81 / 81 | 84 / **34 / 34** | recorded |
| Skipped, no supported camera | – | 0 | 45 | recorded |
| Bad share (a camera moved > 0.10 m or 2° by an accepted solve) | 3/35 = 0.086 | 38/81 = 0.469 | **0/34 = 0** | ≤ 0.1857: **met** |
| Thin (≤ 30 points) / thick bad share | – | 0.815 / 0.296 | – (none thin) / 0 | recorded |
| Camera move per accepted solve, p50 / max | – | 0.067 / 5.41 m | 0.023 / 0.076 m | recorded |
| Sub-mm creation pairs | – | 218 | **0** | 0: **met** |
| Smallest phase-2 row count of a free camera / free cameras under 10 rows | – | 0 / 325 | 23 / 0 | recorded (R57-3) |
| Cameras held, build / phase 2 (solves with a hold) | – | 0 / 0 | 171 / 15 (27) | recorded |
| Jump-gate rejections | (all 43 rollbacks) | 0 | 0 | 0: **met** |
| APE RMSE (m) | 0.0416 | 0.3849 | **0.0332** | 0.042–0.385, nearer the former: **missed on the good side** / met |
| Depth (a) abs / signed | 0.137 / −0.068 | 0.158 / −0.100 | **0.128** / −0.065 | within ±0.01 of 0.137: **met** |
| Depth (c) local scale p90/p10 (keyframes) | 1.41 (89) | 1.97 (94) | 1.48 (92) | recorded |
| Landmarks | 2,111 | 2,899 | 1,949 | recorded |
| Time to first point, p50 (frames) | 38 | 36 | 37 | recorded |
| Map invariants on `[BA-OUTLIERS]` lines | 0 | 0 | 0 | 0: **met** |

- `[reasoning; workshop/58]` The rule replaces the jump gate on its merits. It keeps about as many accepted solves as the gate (34 against 35) and leaves none bad. It does so by skipping thin solves beforehand, not by rolling back afterwards.
- The thin/thick split is empty under the rule by construction. Only the bad share carries the reading.
- Four solves (frames 480, 570, 1110, 2010) skipped phase 2 because their single free camera fell under 30 inlier rows. None is bad.
- Open, carried: the phase-2 fixed-observer anchor count (04y engineering note). The orchestrator's lesson: a comparator band for this rule should be centred on 0.033 or be one-sided, not centred on 04y.

## Integrated system M0: live map, map tracking, O3+ mapper, r52 fallback (04-m0, verified 2026-10-06)

The first contract of milestone 04-4. `--live_map on` joins three parts that had only run separately:
- the 04u opening, which hands its map over at the first acceptance;
- the map tracker with the r52 fallback, tracking every point the mapper creates;
- the 04z mapper (age lifetime, outlier step, supported rule, gate off), which runs only on frames the map poses.

Under M0, "supported" means posed by map tracking with at least 30 inliers. The opening up to the first acceptance (frame 588) is 04u's, row for row. Verified at merge 743d1ba, attempt 2. Attempt 1's product files are byte-identical; attempt 2 only removed a test import. `[measured: runs/04-m0-integration/attempt-2, verifier.json: m0_*, m0_cmp_*]`

**Mechanics (barred, all met).** No map writes, links, point changes or keyframes on frames the map did not pose: 0 / 0 / 0 / 0. Map invariants 0 on the 900-frame run. BA, outlier-step and `[BA-SUPPORT]` identities 1. Depth-filter identity in its R60-1 form 1 (22 exported points without a promote line == 22 initial points alive at the end; 1199 + 30 == 1223 culled + 6 fused). Jump-gate rejections 0. No error lines.

**Coverage over U = frames 382..2543 (2,162 frames).** The floor (R59-1) is 0.8 × 1653/2162 = 0.6117.

| Reading | M0 | M1a r52 (same build, perfect map) | M1a (04m configuration) | Opening only |
|---|---:|---:|---:|---:|
| Full-pose coverage | **0.222** | 0.765 | 0.807 | 0.011 |
| Floor ratio (to M1a r52) | **0.290**: floor **not met** | 1 | – | – |
| Orientation-only / lost | 0.764 / 0.014 | – | – | – |
| Longest loss (frames) | **1,296** (from 1248 to the end) | 60 | 55 | 1,924 |
| Loss / recovery transitions | 8 / 8 | – | – | – |
| Map points at the end | 656 | 17,564 | 21,927 | 55 |

The O3+ comparator (1,949 points) has coverage 0.9995. It is the oracle-pose rung, so its coverage is not a product comparator.

**The start-up row.** First acceptance and hand-off at 588, with 52 initial points. First promoted point at 594. First frame with an inlier on a new point at 595. First frame with 30 inliers on new points at 601. First loss after the hand-off at 709 (a 120-frame tracked run). Median `[MAP-TRACK]` observations on tracked frames: 84. 25 keyframes (588–1240). The mapper ran on 480 frames.

**Local BA on M0.** 17 attempted, 17 accepted, 0 skipped. 4 cameras held, none in phase 2. The smallest free-camera row count is 30. Bad share 0/17. **R59-3 (at least 45/84 = 0.536 skipped) is missed: 0/17.** `[reasoning]` Under M0, BA runs only on frames the map poses. Window cameras posed by map tracking at 30 or more inliers pass the supported test almost by construction.

**Errors over the 480 full-pose frames.** Below the floor, R46-2 and R59-1 make these **conditional metrics unavailable** for the verdict. They are recorded, not compared.

| Reading | M0 | M1a r52 | O3+ (oracle poses) |
|---|---:|---:|---:|
| J over 10° on full-pose pairs (472 pairs) / max | **0** / 0.64° | 0 | 0 |
| J on orientation pairs outside gaps / after a gap: count over 10°, max | 3, 13.7° / 2, 25.6° | – | – |
| APE RMSE (m), Sim(3) scale 0.503 | 0.0095 | 0.083 | 0.034 |
| RPE translation median 1 frame / 1 s / 5 s (m) | 0.0019 / 0.0071 / 0.0121 | 0.0071 / 0.0158 / – | 0.00005 / 0.0016 / – |
| RPE rotation median 1 frame / 1 s / 5 s (deg) | 0.130 / 0.405 / 0.544 | 0.239 / – / – | ≈ 0 / – / – |
| Depth (a) abs / signed / p99 (n) | 0.068 / −0.034 / 0.95 (484) | – | 0.129 |
| Depth (c) local scale p90/p10 (keyframes) | 1.05 (25) | – | 1.48 |
| Sub-mm creation pairs | 24 | – | 0 |
| Promotions with true parallax under θ_min = 1.1° / inside the opening's turn in place | 410 / 226 | – | – |

The O3+ column is m0_score's reading of the same-build 04z support_on run (APE 0.034 m, depth (a) 0.129). 04z's own scorer gives 0.033 m and 0.128. The M1a and O3+ errors are computed over their own, much longer, full-pose sets. A smaller error on 480 frames that cover 2.1 m of path does not make M0 more accurate than either.

**Honest labels.** 10 orientation-only frames are more than 10° off. The 5 post-gap frames include 3 supported frames over 10° (recorded, R59-2). 6 exits from the rotation-only state, 1 wrong. 2 relocalizations, 1 wrong. 7 rotation-only entries, 1,473 rotation-only frames, 3 lost. *Annotation 2026-10-09 (`workshop/66`): the wrong exit and the wrong relocalization are one event (frame 1164, 16 mm, 2.06°), wrong only through the 2° alignment offset described next; under an orientation-anchored alignment M0 has no wrong exit, relocalization or full-pose frame.*

**181 full-pose frames counted "wrong" are an alignment-rotation reading, not a position failure.** `[workshop measurement: workshop/61, Umeyama Sim(3) over the 480 full-pose frames, truth by Slerp]` The script counts 180 (the 1-frame difference is the known gap between Slerp and `gt_at_frames`). Every one of them is over 2° in rotation. **None** is over 0.10 m. Rotation error after the Sim(3): p50 1.95°, p90 2.46°, max 2.91°. 103 of the 180 lie in 588–699. `[reasoning]` A near-constant offset of about 2°, against a 1-frame rotation RPE median of 0.13°, points to the Sim(3) rotation being weakly fixed by a 2.1 m path. It does not point to frame-level tracking errors. The 0.10 m / 2° "wrong" test is therefore at its edge on M0. Read `m0_tracked_wrong_frames` with that in mind.

**The verdict (R60-2, ratified).** Loss of track. The robot drove out of the 2.1 m of path it had mapped while driving and turning (1225–1300). The depth filter takes about 40 frames to create a point, so the map thinned to 28 inliers at 1248. After that, no fallback state may map or exit (R52-1). This is recorded under R46-1 as adverse evidence for the oracle-based point-source choice. Repairs are later labelled arms and do not rewrite it: the R46-4 pair challenger first (04-m0b), then the mixed estimator (R52-4). *Annotation 2026-10-08 (R65-1, `workshop/65`): the mixed estimator is deferred and not built in the project, so this verdict is M0's final reading on the primary.* Mechanism and the ORB-SLAM3 comparison over 1248–2543: `workshop/60`.

## Point-source challenger on M0: keyframe pairs, and the epi run (04-m0b, verified 2026-10-07)

Measurement only; no file changed. Arms: m0 (M0 as built), pair (`--point_source keyframe_pair`, all else the same), epi (`--epi_refine`, recorded), o3pair (the pair source on oracle poses). `[measured: runs/04-m0b-pair-challenger/attempt-1, m0_arms_compare]`

| Reading | m0 | pair | epi |
|---|---:|---:|---:|
| Full-pose coverage over U (floor 0.6117) | 0.2220 | 0.0106 | 0.2216 |
| Longest loss (frames), from | 1296, 1248 | 1924, 620 | 1296, 1248 |
| Points created by the mapper | 1,833 | 0 | 1,686 |
| BA completed | 17 | 0 | 16 |
| APE RMSE (m), own full-pose frames | 0.0095 | 0.0051 (about 30 frames; not comparable) | 0.0071 |
| Run time (s) | 562 | 280 | 817 |

- m0 is byte-identical to 04-m0 attempt 2.
- `transfer_point_source` = 1, `guards_available` = 0. Under R62-2 the 1 is a **start-up failure**: the pair map (55 points) never reached its second keyframe slot (620). It is not a verdict on keyframe-pair mapping, and R46-4's latency question stays open. Four pair passes are vacuous (no points, no BA).
- epi leaves the stall unchanged (same onset and length). It is slightly better where it tracks, at +45 % wall time.
- o3pair: 980 points, APE 0.0019 m, depth (a) 0.162.
- Registered readings: workshop/60's "later onset" not graded (pair lost earlier); R56-2 holds; R59-3 missed again (0 skipped solves). Record: `workshop/63`.

## Residual-consistency test (04-m1, verified 2026-10-08)

`--residual_test {off,log,on}`: for every pose the 30-inlier count accepts, the mean normalized squared residual of the final inliers is compared with the exact 95 % point of the mean of n truncated χ²(2) draws (R63-7); `on` also rejects a pose that fails. Arms: M0 off, M0 log (byte-identical to off), M0 on, the M1a 04q shadow run with log (pinned to 04m), the M1a r52 run with on. `[measured: runs/04-m1-residual-test/attempt-1, resid_compare]`

| Question (R63-8) | Reading | Bar |
|---|---|---|
| Q1 false alarms, M0 | max per-stratum share 0.009 (2 of 479; 3 strata) | ≤ 0.10, met |
| Q1 false alarms, M1a correct shadow poses | max 0.070 (n 86; 5 strata); overall 51 of 1,570 | ≤ 0.10, met |
| Q2 catches, M1a paired shadow events (2,078) | wrong 143 → 108; d −0.0168 [−0.0376, −0.0031] | excludes 0, met |
| Q3 harm, M0 on | coverage over U +0.0014; J full-pose 0 | met |
| Non-vacuity, M1a | 7,214 evaluations, 356 rejections | met |

Recorded: AUC of stat/crit 0.80 (workshop/33 had 0.88 on the earlier sample); the count cut matched to the same false-alarm share catches 17 of 36, the test 35; catchable = 0 by construction; R55-2 not met (5 of 5 present events still wrong); the SIFT σ model is lenient (median stat 0.97 on M0, 1.25 on M1a, against a mean of 1.685). On M1a with r52 and the test on: full-pose share −0.014, tracked-wrong frames −23, wrong exits +1.

**Consequence.** All three questions pass, so under R65-2 the test **rejects** in R04 and on the hold-out.

**M0 on against off is path divergence, not the test's judgement.** `[workshop measurement: workshop/66-m0-offset-check.py]` In the log arm the test fails 2 tracking poses (660, 821) and passes both relocalizations and every evaluated exit. In the on arm it rejects 3 tracking poses (660, 679, 682) and nothing else; the trajectories differ from 660 on. All rejected poses are within 9 mm and 2.7° of the truth. The recorded on−off differences (APE 9.5 → 6.6 mm, wrong exits 1 → 3, wrong relocalizations 1 → 0, tracked-wrong 181 → 166, orientation-only over 10° 10 → 17, loss transitions 8 → 13) follow from that divergence. The "wrong" events are 2.06–2.53° at ≤ 16 mm (the alignment offset below), and the 7 extra orientation frames over 10° are 10.0–10.7° in the 2075–2318 stall (max 13.6° in both arms). None of them measures what the test decided.

**M0's "wrong" events vanish under an orientation-anchored alignment.** `[workshop measurement: workshop/66]` M0's full-pose truth positions have an RMS spread of 0.40 × 0.13 × 0.0005 m. Fixing the alignment rotation from the orientations (chordal mean), then fitting scale and translation on positions, rotates the alignment by 1.94° and costs 4.4 mm of APE (9.5 → 13.9 mm; on arm 6.6 → 11.7 mm). After it: rotation error median 0.39°, max 1.46°; position max 37 mm; **0** wrong full-pose frames, exits or relocalizations in either arm (181 / 166 and 1 / 3 under the position-only fit). This confirms workshop/61's reading of the 181 frames. M1a is scored in the truth frame and has no such offset. Reporting rule from R04 on: R66-1 (ratified 2026-10-09, `workshop/66`).

## Final reference R04: the final configuration (04-r04, verified 2026-10-10)

The project's reference system (corollary E, R64-1). It is M0 with `--residual_test on` (R65-2) and
`--map_rot_gap lost` (R65-4: the first frame after a recording gap is labelled lost in the rotation-only state).
There are three ordinary reruns on the full primary sequence. Verified at merge a28bf16 (build 5451421) on
attempt 1: 172 checks, verifier time 5.3 h. `[measured: runs/04-r04-final-reference/attempt-1, r04_run_1..3,
r04_repro, r04_compare]`

**Gate (R65-3), all met.**
- Reproducibility: the three runs' `estimated.txt`, `estimated_online.txt` and `map_points.txt` are
  byte-identical, and so are their metrics and logs (wall-clock fields stripped).
- Run 1 equals 04-m1's m0_on arm byte for byte. Its labels differ only at the five post-gap frames (1695, 1696,
  1721, 2486, 2491), each going from rotation to lost.
- Every standing and carried check passes.
- Non-vacuity holds: 1 hand-off, 483 mapper frames, 25 keyframes, 485 residual-test evaluations (3 rejections),
  5 post-gap frames.
- Each run took 573–582 s.

**Readings (run 1 = runs 2 and 3).** U = frames 382..2543. The coverage floor (R59-1) is 0.6117.

| Reading | R04 | M1a r52 (same build) | O3+ (oracle poses) |
|---|---:|---:|---:|
| Full-pose coverage over U / floor ratio | **0.2234 / 0.292, floor not met** | 0.765 | 0.9995 |
| Orientation-only / lost over U | 0.760 / 0.016 | – | – |
| Longest loss (frames) | 1,296 (1248 to the end) | 60 | 1 |
| Loss / recovery transitions | 13 / 13 | – | – |
| J over 10° on full-pose pairs (470) / max | **0** / 0.68° | 0 | 0 |
| J on orientation-only pairs, outside gaps / after a gap: count over 10°, max | 3, 13.7° / 0, – | – | – |
| APE RMSE (m), position-only Sim(3), scale 0.509 | 0.0066 | 0.083 | 0.034 |
| APE RMSE (m), orientation-anchored (R66-1); alignment offset | 0.0117; 1.95° | – | – |
| RPE translation median 1 frame / 1 s / 5 s (m) | 0.0020 / 0.0052 / 0.0081 | 0.0071 / 0.0158 / – | 0.00005 / 0.0016 / – |
| RPE rotation median 1 frame / 1 s / 5 s (deg) | 0.126 / 0.382 / 0.498 | 0.239 / – / – | ≈ 0 |
| Depth (a) abs / signed / p99 (n) | **0.067** / −0.045 / 0.87 (441) | – | 0.129 |
| Depth (c) local scale p90/p10 (keyframes) | **1.04** (25) | – | 1.48 |
| Map points at the end / promoted / culled | 570 / 1,662 / 1,134 | 17,564 | 1,949 |
| Writes, links, point changes, keyframes on frames the map did not pose | 0 / 0 / 0 / 0 | – | – |
| Promotions with true parallax under θ_min = 1.1° / promotions inside turn-in-place windows | 408 / 245 | – | – |
| Local BA attempted / accepted / skipped | 15 / 15 / 0 | – | – |

Honest labels:

| Reading | Position-only fit | Orientation-anchored fit (R66-1) |
|---|---:|---:|
| Post-gap frames supported and over 10° | 0 (R65-4 bar met) | 0 |
| Orientation-only frames over 10° | 17 | 3 |
| Tracked frames counted wrong | 166 | 0 |
| Wrong exits | 3 of 11 | 0 |
| Wrong relocalizations | 0 of 2 | 0 |

Under the anchored fit: rotation error median 0.34°, p90 0.72°, max 1.43°; position error max 33 mm.

Notes:
- **Below the floor**, R46-2 and R59-1 make the conditional error metrics (RPE, J, depth, APE) unavailable for
  the verdict. They are recorded above, and the product finding is loss of track (R60-2, unchanged).
- The errors cover 470 full-pose frames and 2.1 m of path. They do not make R04 more accurate than M1a or O3+,
  which are scored over much longer full-pose sets.
- Against M0 with the test off (04-m0), the differences are path divergence after frame 660 (`workshop/66`):
  APE −2.9 mm, coverage +0.0014, wrong relocalizations −1, wrong exits +2.
- Against 04-m1's m0_on arm, the only differences are rot_frames −5, lost_frames +5 and post-gap supported
  over 10° −3.
- The written per-fault verdict is R68-1 (ratified 2026-10-10, `workshop/68`): coverage not met (loss of track);
  faults 1 and 2 met where posed but unavailable as verdicts; fault 3 met as operationalized (low-parallax check
  open); honest labels met; fault 4 not claimed.

## The hold-out, scored once: fr2/pioneer_slam (04-ho, verified 2026-10-11)

The frozen final configuration (R04's product 5451421, FINAL_FLAGS) was run twice on the hold-out, beside the
comparators: a same-build M1a r52 run on a perfect map of the hold-out (the floor run), O3+, and three ORB-SLAM3 runs
(R68-2 to R68-8). Verified at attempt 1. Every validity item passed (R68-6): the product pin, the dataset, the
flags, 0 error lines, byte-identical runs, the perfect map's own verification, the reference runs and U, and the
standing checks. `[measured: runs/04-ho-holdout-scored-once/attempt-1, verifier.json and
artifacts/ho_compare/predictions.json]`. Grading, diagnosis and the proposed per-fault verdict:
`workshop/70-ho-result-and-verdict.md`.

U = frames 0..2920 (every ORB-SLAM3 run made its first keyframe at frame 0). The floor is 0.8 × 0.6546 = 0.5237.

| Reading | R04 (primary) | Hold-out final | Hold-out M1a r52 | Hold-out O3+ |
|---|---:|---:|---:|---:|
| Full-pose coverage over U / ratio to M1a r52 | 0.2234 / 0.292 | **0.1274 / 0.195, floor not met** | 0.6546 | 0.9997 |
| Orientation-only / lost over U | 0.760 / 0.016 | 0.796 / 0.076 | 0.147 / 0.198 | 0 / 0.0003 |
| Longest loss (frames) | 1,296 | 2,007 (837–2843) | 96 | 1 |
| Hand-offs / false starts / resets; surviving map from | 1 / 0 / 0; 588 | 2 / 1 / 1; 335 | | |
| Mapper frames / keyframes / points | 483 / 25 / 570 | 372 / 20 / 817 | – / – / 17,774 | – / – / 1,766 |
| Exits / relocalizations; wrong under the anchored fit | 11 / 2; 0 / 0 | 4 / 7; **4 / 7** | | |
| Tracked-wrong frames, position-only / anchored | 166 / 0 | **372 / 370** | | |
| APE RMSE (m), position-only / anchored; Sim(3) scale; offset | 0.0066 / 0.0117; 0.509; 1.95° | 0.537 / 0.558; 0.119; 18.4° | 0.103 | 0.096 |
| J over 10° on full-pose pairs / max | 0 of 470 / 0.68° | **1 of 358 / 16.3°** | 1 of 1,820 / 11.7° | 0 / 1.95° |
| J on orientation-only pairs outside gaps: over 10° / max | 3 / 13.7° | 35 / 80.2° | 5 / 11.7° | 0 |
| RPE translation median 1 frame / 1 s / 5 s (m) | 0.0020 / 0.0052 / 0.0081 | 0.033 / 0.131 / 0.545 | 0.013 / 0.041 / 0.052 | 0.0001 / 0.0034 / 0.015 |
| RPE rotation median 1 frame / 1 s / 5 s (deg) | 0.126 / 0.382 / 0.498 | 0.379 / 0.893 / 2.10 | 0.482 / 1.06 / 1.11 | ≈ 0 |
| Depth (a) abs / signed median (n) | 0.067 / −0.045 (441) | **0.291 / −0.195** (351) | – | 0.153 / −0.100 (1,205) |
| Depth (c) local scale p90/p10 (keyframes) | 1.04 (25) | **5.25** (15) | – | 1.56 (63) |
| Post-gap frames / supported over 10°, position-only / anchored | 5 / 0 / 0 | 2 / 2 / 1 | | |
| Writes, links, point changes, keyframes on frames not posed | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | | |
| Promotions under θ_min / in turn-in-place windows | 408 / 245 | 423 / 292 | | |

On the hold-out, the 1 s and 5 s RPE horizons are 30 and 150 frames, which is 1.6 s and 8.0 s.

ORB-SLAM3 on the hold-out, three runs:

| Run | Maps | Keyframes | Relocalizations | Span (s of 155.2) | Largest map: span (s) | Largest map: APE (m) | Largest map: scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 413 | 7 | 114.1 | 60.4 | 0.140 | 19.0 |
| 2 | 8 | 433 | 2 | 117.7 | 46.3 | 0.212 | 19.8 |
| 3 | 6 | 414 | 5 | 113.8 | 61.6 | 0.376 | 15.9 |

**Registered predictions (R68-8): 3 of 8 met.** Met: 1, 3 and 6. Not met: 2, 4, 5, 7 and 8 (`workshop/70`).

## Reference system on the primary sequence (adopted 2026-09-17; run 2026-09-17)

**Result in one sentence.** `[measured: runs/harness/reference-orbslam3/summary.txt, three runs]` ORB-SLAM3 monocular does not survive pioneer_slam3 as one map: it cannot initialize during the first 12.7 s, then fragments the sequence into 3 to 6 disconnected maps with no loop closure or map merge, so there is no single reference trajectory to compare against, only per-map fragments.

**How it was run.** `scripts/reference_orbslam3.sh` (build, run, score; reproducible). Upstream `UZ-SLAMLab/ORB_SLAM3` at commit `4452a3c4`, built under `.deps/ORB_SLAM3` with GCC 13, OpenCV 4.6, Eigen 3.4, Pangolin 0.9.4, `-j2`. Settings `Examples/Monocular/TUM2.yaml` unchanged (fx 520.9, fy 521.0, cx 325.1, cy 249.7, with the TUM2 distortion; 1000 features per image; 30 fps pacing). `mono_tum` on the full sequence, viewer off. Six patches, none algorithmic, all recorded in `runs/harness/reference-orbslam3/patches.diff`: C++17 flags, an allocator-constness fix for GCC 11+, viewer off, Sophus built without its tests, a `bool++` made `= true`, and an added output routine that writes every atlas map's keyframe trajectory plus an index, because upstream's save writes only the current map, which was empty in two of three runs (the last map had been reset before shutdown). Each run took about 140 s wall clock, dominated by frame-rate pacing.

**Per run** (keyframes only; each map aligned to ground truth on its own with the harness flags `-va --align --correct_scale`, which is generous: one Sim(3) per fragment).

| run | maps | keyframes (all maps) | keyframe span covered (s of 111.4) | largest map: keyframes, span, APE RMSE (m), Sim(3) scale | new-map events | relocalizations | loop closures or merges |
|---|---|---|---|---|---|---|---|
| 1 | 6 (last empty) | 203 | 63.5 | 87, 30.4 s, 0.039, 2.1 | 6 | 1 | 0 |
| 2 | 4 | 209 | 78.2 | 104, 38.5 s, 0.057, 22.0 | 4 | 3 | 0 |
| 3 | 3 (last empty) | 233 | 86.2 | 191, 72.9 s, 0.277, 28.1 | 3 | 3 | 0 |

Per-map rows for all three runs are in `summary.txt`; the trajectories (`run-N/KeyFrameTrajectory_map<id>.txt`), the map index (`KeyFrameTrajectory_maps.txt`) and the full tracking log (`run-N/tracking.log`) are kept so that per-second RPE at shared timestamps can be scored once H(RPE) exists.

**What the fragments say.**

- *Initialization.* `[measured]` In all three runs the first keyframe is at frame 382 (12.7 s in). ORB-SLAM3's model-selection initializer refuses the opening stretch entirely; the initialization time is identical across runs while everything after it differs.
- *Tracking loss, and what the robot was doing.* `[measured: tracking.log; groundtruth.txt 1 s windows; ORB keypoint counts at budget 1000]` "Fail to track local map" fires 339 / 198 / 110 times per run; each new map is preceded by it. The losses recur at the same moments across runs (map ends at +16.7, +51.2, +66.8, +85.6, +103.5 s), and at every one of them the robot was **driving and turning at once**: 0.17 to 0.31 m/s with 14 to 23 deg/s of yaw, with normal texture (870 to 1000 ORB keypoints per frame against a sequence median of 970), except the last, which is also the sequence's low-texture stretch (about 300 keypoints). The near-static stretches (+24 to +29 s) and the rotation-in-place stretches (+37 to +40 s, +54 to +60 s) were survived inside a map. The re-initializations all happen while translating (0.25 m/s, about 1 deg/s), as parallax requires. `[reasoning]` So the monocular failure is not "no motion" but "turn while driving": a forward-looking camera turning at 20 deg/s sweeps in new content faster than forward motion can triangulate it (small parallax on the points ahead), and the local map runs dry. The RGB-D diagnostic below confirms it is the geometry and not the scene: with depth, the same stretches are tracked.
- *No loop closure, and why.* `[measured]` The revisit (frames 640 to 900, +21.3 s to +30.0 s) falls inside one map in every run (map 1 in run 1, map 0 in runs 2 and 3), yet no loop or merge was detected. `[reasoning]` The place being revisited was first seen before frame 382, before ORB-SLAM3 had a map, so there was nothing to close against. The reference system therefore says nothing about loop closure on this sequence.
- *Scale.* `[measured]` The Sim(3) scale correction evo applies differs per map by an order of magnitude (2.1 to 28.1); each fragment carries its own arbitrary monocular scale, as expected, which is a second reason the fragments cannot be concatenated into one trajectory.
- *Nondeterminism.* `[measured]` Three runs gave 6, 4 and 3 maps and largest-map APE from 0.039 to 0.277 m. This is the free-running-threads behaviour `design.md#target-architecture` (Threads row) names as the reason our own evaluation mode must be synchronous.

**How to read it against the chain.** `[measured: #harness-measurements]` Our chain reports one continuous trajectory over all 2544 frames at 1.751 m APE; ORB-SLAM3 reports sub-metre APE only on fragments covering 57 to 77 percent of the sequence by keyframe span and nothing at all in the first 12.7 s. These are answers to different questions and must not be placed side by side as a ranking. `[reasoning]` For the decision point, the reference run supports the coverage precondition in `design.md#dependencies-and-decision-points`: comparisons with R are only meaningful on R's own timestamps, per fragment. The placeholder guidance above turned out half right: map-based monocular tracking as ORB-SLAM3 implements it does lose this sequence, but at driving-and-turning moments rather than at the static or rotation-only stretches, so the robustness the decision point should weigh is point creation and recovery under forward motion with yaw (initialization refusal at the start, map upkeep during turns, relocalization), at least as heavily as the tracking architecture itself.

**Not done.** Frame trajectory (upstream has no monocular frame save; optional per `design.md`); per-second RPE at shared timestamps (waits on H(RPE)); ORB-SLAM3's own vocabulary was used, not the toolbox one.

### RGB-D diagnostic (2026-09-17): not a reference, not a comparator

`[measured: runs/harness/reference-orbslam3/rgbd-diagnostic/REPORT.md, three runs]` The same ORB-SLAM3 build in RGB-D mode, run only to separate "monocular geometry fails here" from "the scene fails here". It initializes on frame 0 and holds one map for 64 / 103 / 88 s of 111 s at 3 to 9 cm APE RMSE (frame poses of the largest map, Sim(3)-aligned with the harness flags; fitted scale 1.03 to 1.05, so near metric), through every stretch where the monocular run broke. It breaks reliably only at +102.6 s, where texture collapses to about 250 ORB keypoints per frame while the robot yaws at 41 deg/s. `[reasoning]` So the monocular losses are geometric, not scenic; the scene fails only in its last ten seconds. **The RGB-D numbers are not targets and must not appear next to any monocular stage's numbers**: the product does not use depth (`design.md#goal`), and the acceptance bar is unchanged. The full account, with the motion and texture at every loss onset, is in the report.
