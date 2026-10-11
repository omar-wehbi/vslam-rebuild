# Deferred designs

*Wrap-up item 4, 2026-10-11. Companion to `theory/closing-report.md` (section 8).*

These are designs that were written but not built. They were deferred by the scope freeze (2026-09-20), by the hard
endpoint (R64-1, 2026-10-08), or because they were found after the test sequence was scored. None is planned.

Each entry gives:
- **what** it is;
- **why** (the evidence that motivates it);
- **what it would take**;
- **the prediction** it must meet, if one was registered before any result;
- **how to settle it.**

**About testing them.** The test sequence (`fr2/pioneer_slam`) has been scored once, so any later run on it is
labelled *post-hold-out*. It shows whether a fix addresses the mechanism it targets, but not whether the system
generalizes. A fresh judgement needs sequences the project has not seen.

Citation tags are as in the closing report: `[measured]`, `[workshop measurement]`, `[reasoning]`, `[paper]`.

## Contents

| # | Design | Targets | Status |
|---|---|---|---|
| 1 | No mapping while turning in place | the test sequence's scale change (faults 2 and 3) | new, post-hold-out |
| 2 | Map re-initialization (Atlas) | the coverage stall on both sequences | designed 2026-10-06 |
| 3 | Tracking across recording pauses | stale post-gap poses (fault 1, honest labels) | open question since 2026-09-24; widened post-hold-out |
| 4 | Mixed pose estimator, 1–29 map points | the loss-of-track onsets | designed 2026-10-07, deferred untested |
| 5 | Keyframe policy: interval against ORB-SLAM's rule | start-up with small maps | prediction registered 2026-10-07 |
| 6 | Generalized GRIC at initialization | the late first map | deferred 2026-10-01 |
| 7 | Kinematic translation prior | translation direction at 9 mm steps | rules fixed 2026-09-19 |
| 8 | Longer seed lifetime and deferred promotion | the mapper's depth shortfall | analysed 2026-09-21 |
| 9 | Sim(3) loop closure (stage 05) | revisit (fault 4) | deferred 2026-09-20 |
| 10 | Component comparisons (stage 06) | component choices | deferred 2026-09-20 |
| 11 | Threads (milestone 04-5) | display responsiveness | deferred 2026-09-20 |
| 12 | Newton-PnP in C++, speed against SQPnP | solver speed | deferred post-project 2026-09-25 |

---

## 1. No mapping while turning in place *(post-hold-out)*

**What.** While the camera only turns in place, the mapper stops making keyframes, seed updates and promotions,
even if the tracker still holds a full pose. Tracking continues against the existing map.

**Why.**
- **The test sequence.** During a 51° turn with 6 cm of true motion (frames 571–645), the system kept tracking. It
  made 3 keyframes and promoted 292 points. The map's scale then changed about 4 times between keyframes 640 and 660
  `[workshop measurement: theory/workshop/71]`.
- **The development sequence.** Points promoted while turning in place have 5 times the local depth error of the
  rest, though the scale held there.
- **Not low parallax as such.** On the development sequence, low-parallax promotions are as good as the rest.

**What it would take.**
- **An online detector of "turning in place".** It must use estimated quantities only, never ground truth. The
  natural one is the ratio of estimated rotation to estimated translation over a short window, or the rotation-only
  fit's residual that the opening already computes (`--tracking rotation`).
- **A switch** that gates the mapper on that detector.
- **A replay** of R04's configuration with the switch on, on both sequences.

The constants must come from an existing source (the rotation-only fit's own test) and not be fitted to frames
571–645.

**Prediction.** None registered. This design was written after the result it targets, so its first test is
post-hold-out by construction.

**How to settle it.**
- **Test sequence (post-hold-out).** The scale step at keyframes 640–660 disappears, and the local-scale spread
  (depth (c)) falls to the mapper's level on true poses (about 1.6).
- **Development sequence.** Full-pose coverage does not drop, and depth (a) does not get worse.
- **Generalization.** The real test is the cross-dataset study, on sequences with turns in place.

## 2. Map re-initialization (Atlas)

**What.** When relocalization has failed for a while, the active map is stored and a new map is started from
scratch with the opening's two-view initializer. This is ORB-SLAM3's Atlas
`[paper: orb-slam3-2021.pdf §III, via papers/orb-slam3.md]`. Merging maps later needs place recognition and a Sim(3)
welding BA, which is design 9's machinery.

**Why.** Coverage fails on both sequences by the same mechanism: the robot drives out of its only map, and no state
can add points once the map is out of view (closing report 4.0).
- **Development sequence.** From frame 1248 to the end, 1,296 frames.
- **Test sequence.** From frame 837 to 2843, 2,007 frames, while the robot drives 32.7 m.

ORB-SLAM3 covers 0.56–0.82 of the development sequence's frames 1248–2543 with new maps
`[workshop measurement: theory/workshop/60]`.

**What it would take.**
1. Relax R52-5 ("lost coverage alone is not a reason for a new state") for this one mechanism.
2. A ruling on per-map scoring: does a full pose inside a second map, with its own scale, count toward coverage, and
   how are J, depth and wrong frames read across maps?
3. A trigger: relocalization failing for N frames, with N taken from ORB-SLAM3's source, not tuned.
4. The opening's initializer re-entered with the current frame as the new first frame.

**Prediction** `[reasoning, workshop/60]`. From 1300 the robot drives straight at about 0.27 m/s on the development
sequence, which is easy parallax for the initializer, so the coverage gain could be large. That was written down,
but not registered as a number.

**How to settle it.**
- Full-pose coverage over U clears the floor on both sequences.
- The honest-label bars hold per map: anchored wrong frames, wrong exits and relocalizations, and post-gap frames.

## 3. Tracking across recording pauses

**What.** Handle the first frame after a recording pause (a gap over 150 ms in the timestamps) so that it is never
labelled supported with a stale pose.

**Why.**
- **On the perfect map** (2026-09-24). After a 1.3 s pause at frame 2491, the tracker posed the frame wrongly and
  corrected it on the next frame, a 23.7° step `[workshop measurement: theory/workshop/26]`.
- **On the test sequence.** The robot turned 16.2° during a 1.88 s pause. The tracker held the old orientation at
  frame 795, labelled it supported, and caught up at 796 `[workshop measurement: theory/workshop/70]`.
- **The rule we have does not apply there.** The post-gap rule R65-4 only acts in the rotation-only state, and both
  of the test sequence's pauses fell during tracking.

**Options.**
- **(B) Gap-aware motion prediction:** scale the constant-velocity step by the actual time gap. It is
  parameter-free. But workshop/26 already noted that it does not help when the robot *turned* during the pause,
  which is exactly the test sequence's case.
- **(C) Extend R65-4 to the tracked state** *(post-hold-out)*: the first frame after a pause is labelled lost
  whatever the state, and tracking resumes on the next frame. It costs one frame per pause (7 frames over both
  sequences) and fixes the label by construction. It does not fix the pose.
- **(D) Check the two-view rotation across the pause** *(post-hold-out)*: the rotation predictor already estimates
  the two-view rotation. If the tracked pose disagrees with it by more than the jump bar, the frame is not accepted.
  The cause of the stale pose at 795 has not been diagnosed, so whether (D) would catch it is unknown.

**Recommendation** `[reasoning]`: (C) first, because it is the honest-label fix. Then diagnose 795, and decide
between (B) and (D) on that diagnosis.

**How to settle it.** Replays of both sequences: 0 post-gap frames supported and over 10°, and 0 J over 10° on the
pair after a pause.

## 4. Mixed pose estimator, 1–29 map points *(04-m2, deferred untested)*

**What.** When the tracker sees between 5 and 29 map points, too few for a full pose by the 30-point rule, it
estimates the pose from those points *plus* 2D features: the depth filter's seeds, used as rays from their source
keyframe. This follows DT-SLAM's segment residual. It is accepted only through the residual-consistency test built
in 04-m1 (`theory/workshop/63`, R63-1 to R63-6).

**Why.** All 8 of M0's loss-of-track onsets on the development sequence had 7–29 map inliers. The loss at 992 cost
172 frames, and the one at 1248 cost the rest of the sequence `[measured: 04-m0b]`.

**Design** (ratified 2026-10-07):
- **The features.** The 2D features are the existing seeds (R63-2). The residual is the paper's segment from z_min
  to infinity, and inliers are classified at χ²(1) = 3.84 (R63-3).
- **The minimum.** At least 5 map points are needed (the paper's 4 + 1), because map points are the only source of
  scale (R63-4).
- **Two arms** (R63-5):
  - *pose arm*: mixed frames are read-only;
  - *map arm*: accepted mixed frames feed the mapper.

**Registered predictions** (R63-6, 2026-10-07):
1. Both arms move the 1248 onset later. Most of the 8 loss/recovery transitions disappear, including the 172-frame
   loss after 992.
2. The pose arm does not end the stall: the map is out of view at the true pose by about frame 1350.
3. The map arm: unresolved, no number registered.
4. Mixed-frame accuracy is worse than on frames with at least 30 points; the size of the gap is not predicted.

**Why deferred** (R65-1, `theory/workshop/65`). It buys coverage and answers none of the four faults. Even its
predicted gain would leave M0's verdict unchanged. "Deferred untested", never "shown not to help".

**How to settle it.** The pre-registered rule: full-pose coverage over U, guarded by J, RPE, depth and wrong
full-pose frames, with mixed frames reported as their own stratum. It works best combined with design 2, since
prediction 2 says it cannot end the stall alone.

## 5. Keyframe policy: interval against ORB-SLAM's visual-change rule

**What.** Replace keyframes every 20 frames with ORB-SLAM's rule `[paper: orb-slam-2015.pdf §V.E]`. Insert a
keyframe when:
- more than 20 frames have passed since the last relocalization;
- the mapper is idle or 20 frames have passed;
- the frame tracks at least 50 points;
- the frame tracks less than 90 % of the reference keyframe's points.

**Why.** The keyframe-pair point source failed at start-up on M0 (04-m0b). It handed over 55 points at frame 605 and
was lost at 620 with no new points. Interval keyframes were one suspect (`theory/workshop/62`, R62-2).

**Registered prediction** (2026-10-07). The paper's rule, ported literally, inserts **no** keyframe on a 55-point
opening map, because the 50-tracked-points condition never holds: the frame tracks 31–44. The pair arm fails at
start-up as it does now. A version without that condition would be an adaptation chosen after seeing the result.

**What it would take.** A 2×2 comparison: point source {depth filter, keyframe pairs} × {interval, §V.E rule}, on
the integrated system, with that prediction as written.

**How to settle it.** Start-up survival and full-pose coverage per cell. The prediction is met if the §V.E × pairs
cell inserts no keyframe before the loss.

## 6. Generalized GRIC at initialization

**What.** Replace the initializer's homography-or-fundamental selection threshold (`rh_select` = 0.45, ORB-SLAM's
value; ORB-SLAM3 uses 0.50) with generalized GRIC, which scores each model from the noise model
(`papers/gauglitz-hybrid-rotation-slam.md`).

**Why.** On the development sequence the first map comes late (frame 588). Every homography selection on the
opening had a ratio between 0.45 and 0.50, the range where the two sources disagree. The selection statistic does
not separate pure rotation from translation with 6–11° of parallax `[measured: 04u; theory/workshop/43]`. Changing
the threshold after seeing that was rejected as tuning (R43-3).

**What it would take.** An off-by-default switch on the selection step, with a pre-registered rule, run on the
opening with the synthetic pure-rotation regime re-verified through the initializer's checks.

**Prediction.** None registered.

**How to settle it.** The first-map frame moves earlier with no wrong first map, on the development sequence and on
new sequences. On the test sequence the opening produced a false start at 236 and a surviving map at 335; any
change there is post-hold-out.

## 7. Kinematic translation prior

**What.** Use the wheeled robot's motion model (forward motion along the camera's mounting direction) as a prior on
the translation direction.

**Why.** At 9 mm steps, two views barely constrain the direction: 31 % of the old chain's steps pointed backwards
`[workshop measurement: 03c]`. Map-based tracking reduced the problem, but the prediction step still uses a
constant-velocity model.

**Rules it must obey** (ruled 2026-09-19, `design.md#platform-priors-are-switches-never-load-bearing`):
1. It is a stated physical model with frozen, recorded constants.
2. It is one switch, off in the reference configuration.
3. The system must meet its bars with the switch off.
4. On a sequence where the model is violated (fr1/desk, handheld), the system must detect the violation and degrade
   to the general path.

The test sequence is the same robot, so it does not test rule 4.

**What it would take.** The constants: camera-to-vehicle rotation and axle offset, measured once. Then a prediction
term in the tracker, and the fr1/desk run.

**Prediction.** None registered. It was first in line among the dropped arms if time had remained (corollary D).

## 8. Longer seed lifetime and deferred promotion (the mapper's depth shortfall)

**What.**
- **Longer seed lifetime:** keep depth-filter seeds alive longer than five keyframes.
- **Deferred promotion:** promote a point only after more parallax has accumulated than at first convergence.

**Why.** On true poses the adopted mapper reaches 0.144 median depth error, against the 0.122 bar. The analysis
shows the error is admitted at promotion `[workshop measurement: theory/workshop/19]`:
- a relative convergence test passes seeds that are too close, with less true parallax than θ_min, so first
  convergence is biased toward too-close points by selection;
- deferring promotion dilutes the bias;
- the five-keyframe lifetime cuts off the long-lived seeds that approach the bar.

Stage 04i measured the lifetime rule. The adopted configuration keeps age-based lifetime.

**What it would take.** A lifetime and deferral sweep on O3+ (the mapper on true poses), with the bars 0.122 and 1.5
fixed in advance.

**The trade-off.** Deferral costs latency. The integrated system was lost for density at 1248 partly because a point
takes about 40 frames to appear `[measured: 04j; workshop/60]`. Any deferral must be judged on the integrated
system's coverage, not only on O3+ depth. This is the lesson of the closing report's section 9.

## 9. Sim(3) loop closure *(stage 05)*

**What.**
- Appearance-based loop candidates.
- A 3D–3D Sim(3) constraint: Horn's method in RANSAC, then reprojection refinement in both keyframes.
- A Sim(3) pose graph, with map points following their keyframe's correction (`papers/orb-slam.md`, loop-closing
  heading).

A tested alternative to Horn + RANSAC is the RBD lab's provable 3D–3D registration
(`papers/feldman-rbd-lab.md`).

**Why.** Fault 4 is not claimed. The original loop closer chose candidates by time, and was seeded from the drifted
pose; the seeding was fixed in 01c. On a dense map it accepts wrong loops: one detection, frame 1200 against frame
660 on 11 PnP inliers, moved a *perfect* trajectory by 0.29 m `[measured: runs/harness/ladder-u]`. It is off in the
final configuration.

**What it would take.** The done criteria in `design.md#adopted-ideas` (Sim(3) row):
- **The positive case:** a synthetic drifted loop of known scale and pose error, recovered within a stated
  tolerance, with an FD check on the Sim(3) Jacobian.
- **The negative cases:** an identity loop accepted with a near-zero correction; corrupted matches and degenerate
  geometry rejected; no double correction.
- **Map consistency after the correction:** observation reciprocity and shared-landmark consistency.

The test case from the record: the 1200↔660 detection must be refused.

`[reasoning]` Sim(3) corrects scale *drift*, which presupposes a scale that changes slowly. The test sequence's
step change at 640–660 is not slow drift, so design 1 comes before this one.

**How to settle it.** The development sequence's revisit (frames 640–900) is detected and verified, and its
correction matches the drift measured against ground truth at that point.

## 10. Component comparisons *(stage 06)*

**What.** One switch per stage, each compared against the reference configuration with an APE guard. The candidates
are in `design.md#candidate-ideas-from-the-rbd-lab`:

| Candidate | Pre-test before any stage |
|---|---|
| Coverage-based keyframe selection | – |
| Appearance-ranked loop candidates | goes with design 9 |
| Ray-clustering point fusion (k > 1) | – |
| Submodular map cleaning | – |
| 1-line-mean triangulation | – |
| Ground-plane scale | ORB feature counts on the floor region decide whether it is viable |
| Learned depth priors | only if a GPU is available; a monocular depth network scored by the depth harness first |

Each candidate carries its guarantee-transfer statement on file: what the paper proves, and whether it holds in this
setting.

## 11. Threads *(milestone 04-5)*

**What.** Tracking, local mapping and loop closing on separate threads, with a keyframe queue and a map lock. The
verifier uses a synchronous evaluation mode, which the threaded mode must reproduce at defined sync points.

**Why deferred.** The verified system is the synchronous one. Under Python's global interpreter lock, threading buys
display responsiveness, not throughput or a result (`design.md#target-architecture`, Threads row). Two display
costs are recorded: the per-frame snapshot copies the whole trajectory and every map point, and the display path is
never exercised by the verifier.

## 12. Newton-PnP in C++, and a speed comparison with SQPnP *(post-project)*

**What.** A C++ implementation (Eigen, bound to Python) of the globally optimal Newton-PnP studied in 04o–04q, timed
against SQPnP.

**Why deferred.** The study's bar was correctness only; our Newton-PnP is Python, so no timing claim could be made
(D1a). The study kept EPnP as the relocalization solver (`theory/expected-performance.md`, Newton-PnP parts A–C;
`theory/studies/newton-pnp-vs-epnp.md`).
