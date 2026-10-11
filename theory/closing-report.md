# Closing report: repairing a monocular visual SLAM system

*October 2026.*

**Who did the work.** I ran this project with AI agents (Claude models, through Claude Code) inside a verification
loop I designed: agents planned and built each stage, scripts measured it, and I made the decisions of substance.
"We" in this report means me and those agents. Section 9 describes the loop.

**How to read the citations.** Every number in this report comes from a recorded run or analysis file. Tags say
which kind of evidence each number is:
- `[measured: <path>]`: a run scored by the project's verifier, or a file it produced;
- `[workshop measurement: <path>]`: an analysis of saved run files, done outside the verifier;
- `[reasoning]`: our own inference;
- `[paper: <file> §<section>]`: a claim from a source paper.

The R-numbers are rulings: decisions taken during the project, each recorded with its discussion.

**What is public.** Citations give paths in the working repository. This repository holds the code, this report,
`theory/expected-performance.md` (every number and its definition) and `theory/deferred-designs.md` (the designs
that were not built). The run records (`runs/`), the rulings ledger (`theory/design.md`) and the workshop notes
(`theory/workshop/`) are kept in the private working repository.

---

## 1. Summary

**What this is.** A Python monocular visual SLAM system. It estimates a camera's path, and a sparse 3D map, from the
colour images of a single camera on a wheeled robot. The development sequence is TUM RGB-D `fr2/pioneer_slam3`
(2,544 frames). The test sequence, scored once at the end, is `fr2/pioneer_slam` (2,921 frames). Both are hard for
monocular vision:
- the walls have little texture;
- the robot often stands still or turns in place;
- its steps are small, about 9 mm per frame.

**The starting point.** A review of the original system named four faults:
1. the path jumps;
2. the map points are at the wrong depth;
3. map points are invented while the camera only rotates;
4. returning to a seen place has no visible effect.

Reading the code confirmed those faults and found their causes. The system was a frame-to-frame chain, every step
of unit length, with occasional corrections. Under it lay wrong derivatives in bundle adjustment and in the pose
graph, an inflated success counter, map tables that disagreed with each other, and a loop closer seeded from the
drifted pose it was meant to correct.

**What we did** (September to October 2026; 45 stage contracts, 33 verified):
1. **Repaired the foundation.** Correct derivatives, honest counters and consistent map tables, each now checked by
   a script on every stage.
2. **Measured what the chain costs.** We fed the old chain perfect inputs, one kind at a time (the "oracle
   ladder"). Even with perfect rotations and directions, unit-length steps leave 0.91 m of error. That decided the
   architecture: **locate the camera against a map every frame**, as ORB-SLAM does, instead of chaining motions.
3. **Built the new system component by component.** Each part was tested with the others replaced by ground truth:
   - a mapper on true poses;
   - a tracker on a perfect map;
   - a start-up that waits for enough geometry;
   - a rotation-only state for turning in place.
4. **Integrated it** (configuration M0), froze it as the final reference **R04**, and scored it once on the test
   sequence.

**Did it work?** Partly. The repairs worked, and so did the components, each on its own. The integrated system did
not reach its goal.

| Fault | Development sequence (R04) | Test sequence (scored once) |
|---|---|---|
| **Coverage** (the precondition for judging the rest) | **Not met.** A full pose on 22 % of the frames, against a floor of 61 %. The robot drives out of the small area it mapped | **Not met.** 13 %, against 52 %. Same mechanism |
| 1. Jumps | **Unavailable as a verdict** (below the floor). Where it posed: 0 of 470 frame pairs over 10° | **Unavailable as a verdict.** 1 of 358 over 10°: a stale pose across a 1.9 s recording gap |
| 2. Map depth | **Unavailable as a verdict.** Where it mapped: 6.7 % median depth error | **Unavailable, and not met where mapped.** 29 % median error: the map's scale changed about 4 times |
| 3. Points under rotation | **Met as the bar was written:** nothing is written to the map while only rotating. But points made while turning in place *and still tracking* are 5 times worse than the rest | **Met as written.** The same flaw came right before the scale change |
| Honest labels (a frame labelled "supported" must be right) | **Met** | **Not met:** 370 of 372 posed frames are wrong under the stricter alignment |
| 4. Revisit | **Not claimed.** Scoped out on 2026-09-20; the old loop closer is off | **Not claimed** |

**In one line:** on the sequence it was developed on, the system is right where it poses but cannot stay posed. On
a second sequence it also cannot keep its map's scale, which changed right after it kept mapping through a turn in
place.

**What we would keep.** The verification method, the oracle ladder and the foundation repairs, all reusable (section 9). The
clearest next step is not to map while turning in place (`theory/deferred-designs.md`).

---

## 2. Where we started

**The system as found** (code as of 2026-09-08; `DIAGNOSIS.md`, `PROJECT_EVALUATION.md`). It was a
frame-to-frame odometry chain:
1. match features to the previous frame;
2. estimate the essential matrix;
3. take its rotation and its translation *direction*, setting the step length to 1;
4. compose the result onto the previous pose.

Every 10 frames it tried to correct the pose by PnP against map points. Every 30 frames it ran bundle adjustment
(BA) over the last 12 frames. Map points were triangulated from consecutive frames. A loop closer compared the
current frame to five keyframes spaced evenly in time.

**Defects confirmed by reading the code and by checks:**

| Defect | Evidence | What it did |
|---|---|---|
| BA's analytical derivative wrong for nonzero rotations | Finite-difference check: 14–28 % error in the rotation block | Every optimizer comparison so far had been run on wrong derivatives |
| Pose-graph residual and derivative used different conventions | Disagreement over 100 % | Loop corrections could not converge correctly |
| BA success counter counted solves that did nothing | 69 "successes" where 32 solves completed | Accuracy claims inflated |
| A frame's keypoint→landmark table could disagree with the landmark's observation table | Two-view triangulation could overwrite a link; fusion could rewire two keypoints to one landmark | The map could not be trusted as a data structure |
| Loop-closure PnP seeded from the drifted pose, with candidates chosen by time | Iterative solver converges next to its seed | The one historical loop detection was almost certainly a no-op |
| Rotation-only model off, and its selection rule unable to fire under pure rotation | Code reading | A unit baseline was fabricated during pure rotation, producing a self-consistent but meaningless cloud |
| Near-180° orientation flips | `[measured: runs/02-reference-and-repeatability]` rotation error 54.6° RMS against a median of 0.36° | Found on 2026-09-18; caused most of the "jumps" |
| Lens distortion ignored | Found 2026-09-19 | A one-signed depth bias of a few percent |

**The baseline.** The original code scored **1.74 m** trajectory error (APE after similarity alignment) on the full
development sequence. With the foundation repaired, the frozen reference R02 scored **1.48 m**. After the flip fix,
R03b scored **1.40 m** `[measured: runs/02-…, runs/03b-…]`.

**A first repair attempt** (August 2026, before this work) added features on top: switching on dormant options,
smoothing corrections, local-map matching. It made accuracy worse or left it unchanged. An independent evaluation
concluded, and we agreed, that it was building on the defects above rather than fixing them (that plan: `plan/00-INDEX.md`; its stage reports: `runs/stage-1/` to `runs/stage-3/`).

**How we worked this time.** Every change went through a stage contract. Each contract fixed, *before* any code was
written, the executable checks and the thresholds that decide pass or fail. A verifier script then computed the
verdict, and every number was recorded with its source. Finite-difference checks of both derivatives and a
map-consistency check became standing checks, re-run on every stage. Section 9 describes the five-role loop that ran
this, and what it cost.

---

## 3. The oracle ladder: what the chain cost, and why we changed the architecture

**The question.** Was the chain's 1.4 m error a matter of tuning, or of structure? We answered it by feeding the
chain perfect information one kind at a time. Each step of this "oracle ladder" replaces one estimated quantity
with ground truth, and the error that remains belongs to what is still estimated (`theory/workshop/04`,
`expected-performance.md#oracle-ablation`).

`[measured: runs/02-reference-and-repeatability, runs/03c-oracle-pose-rungs, ladder.json]`

| Rung | Given from ground truth | APE (m) | Map points | What it shows |
|---|---|---:|---:|---|
| O0 (R03b) | nothing | 1.404 | 13,473 | the reference |
| O1 | step lengths | 1.641 | 12,648 | True scale buys nothing: the error is not about scale |
| O2z / S1 | rotation and direction, unit steps | **0.907** | 317 | **The floor of any unit-step chain, confirmed two ways** (in the pipeline and by synthesis) |
| O3 | full relative motion | 0.0117 | 320 | Downstream does no harm, but does little |
| O4 | true absolute pose at PnP frames | 1.417 | – | The PnP gate accepted 1 of 251 *true* poses |

**What the ladder found:**
- **The chain's error is translation direction.** Rotation is good: 0.41° median error on translating frames.
  Direction is not: 39° median error, and **31 % of steps point backwards** `[workshop measurement: 03c [ORACLE]
  lines]`. At this robot's 9 mm steps, two views do not constrain the direction well. That is a property of the
  geometry, not of a tunable.
- **No scale source exists to rescue the chain.** The only one in the code fired on 7 % of frames with a median
  relative error of 3.7, and it made the error worse (1.60 m against 1.48 m).
- **The chain's map is an artefact of wrong motion.** It has 13,473 points under estimated motion, and about 320
  under true motion, because consecutive-frame triangulation is not a viable point source at 9 mm steps. The points
  sat at about **a tenth of their true depth** (depth (a) −0.91), with a nine-fold spread of local scale across
  keyframes `[measured: runs/03e-…]`.
- **The PnP correction could not act.** Its step gate rejected 250 of 251 true absolute poses.
- **The finding survived undistortion.** Re-measured after the lens-distortion fix (`runs/harness/ladder-u`), the
  direction error and the 0.9 m floor stand.

**The decision (2026-09-18).** Replace the chain with **map-based tracking**. Every frame is located against map
points, and the two-view estimate only predicts. This is the architecture of ORB-SLAM
`[paper: orb-slam-2015.pdf §V]`; our notes are in `theory/papers/orb-slam.md`. We also froze the scope on
2026-09-20: a new loop closer (fault 4) and component comparisons were deferred. The new system would be built as a
**map ladder**, where each part is tested with the others replaced by ground truth:

| Rung | What is real | What is ground truth | Milestone |
|---|---|---|---|
| O3+ | the mapper | camera poses | 04-1 |
| M1a / M1b | the tracker | the map (a perfect map built from depth images and true poses) | 04-2 |
| opening | start-up and the rotation-only state | nothing (first frames only) | 04-3 |
| **M0** | everything | nothing | 04-4 → R04 |

**What each milestone delivered** `[measured: expected-performance.md, sections 04-1 to 04-m0]`:
- **04-1, mapper on true poses.**
  - Point source: SIFT features plus a per-feature depth filter (after SVO); this replaced consecutive-frame
    triangulation.
  - Depth error: 0.144 median, against a bar of 0.122. That is a recorded shortfall, but it improves on the chain's
    0.91.
  - Undistortion alone took the signed median from −0.42 to −0.28.
- **04-2, tracker on a perfect map.**
  - Full-pose coverage reached 0.81 with relocalization. EPnP was kept as the solver, after a three-part study of an
    alternative.
  - The configuration used as the coverage floor (M1a r52) reached 0.765.
- **04-3, the opening.**
  - The system tracks orientation only while it turns.
  - It builds the first map only when two-view geometry supports one: an H/F model check with refusal under low
    parallax, after ORB-SLAM's initializer.
  - On the development sequence, no map is built during the opening turn in place.
- **04-4, integration (M0).**
  - The live map, the tracker, the mapper and the rotation-only fallback run together.
  - The final configuration R04 adds a residual-consistency test and a post-gap labelling rule.
  - Three reruns are byte-identical.

---

## 4. The verdict, fault by fault

**How to read this section.** Each fault has a **bar** (fixed before the measurement), the **evidence** on both
sequences, and the **verdict** (R70-1, amended by R71-1; `theory/workshop/70`, `71`).

One rule governs everything below. The error metrics of faults 1 and 2 are only a verdict if the system holds a
full pose on enough of the sequence: **coverage at least 80 % of what the same tracker achieves on a perfect map**
(R46-2, R59-1, fixed before the integrated system first ran). Below that floor, a small error on a small part of the
sequence would stand in for the whole, so the metrics are reported but are **unavailable as verdicts**.

U is the frame range a reference system (ORB-SLAM3) can initialize on: frames 382–2543 on the development
sequence and all frames on the test sequence (R68-5).

### 4.0 Coverage: not met on either sequence

| | R04 (development) | Test sequence |
|---|---:|---:|
| Full pose over U | **0.223** | **0.127** |
| The same tracker on a perfect map (M1a r52) / floor (80 %) | 0.765 / 0.612 | 0.655 / 0.524 |
| Orientation only / lost | 0.760 / 0.016 | 0.796 / 0.076 |
| Longest stretch without a full pose (frames) | 1,296 | 2,007 |

**The mechanism is the same on both sequences: the map's extent.**
- **Development sequence.** The robot mapped 2.1 m of path and then drove out of it. From frame 1248 to the end it
  was in the rotation-only state.
- **Test sequence.** It mapped about 5 m and then drove 32.7 m, at a median 1.8 m from anywhere it had mapped. It
  relocalized only when it came back (frame 2844).

In neither case can a fallback state add map points, so once the map is out of view nothing can bring it back
(R52-1, R60-2) `[workshop measurement: theory/workshop/60, 70]`.

`[reasoning]` That is a deliberate choice, not an oversight. The fallback states never write to the map, which is
what fixes fault 3 under the bar as written. Its cost is coverage. The design that would address it (re-initializing
a new map, as ORB-SLAM3's Atlas does) is in `theory/deferred-designs.md`.

### 4.1 Jumps: unavailable as a verdict on both sequences

**Bar.** No frame-to-frame rotation error over 10° on pairs the system labels as fully posed (J, ruled 2026-09-18).

| | Original chain (R03b) | R04 | Test sequence |
|---|---:|---:|---:|
| Full-pose pairs over 10° | 11 of 2,507 (max 150°); 236 before the flip fix | **0 of 470** (max 0.68°) | **1 of 358** (16.3°) |
| Orientation-only pairs over 10°, outside recording gaps | – | 3 (max 13.7°) | 35 (max 80.2°) |

- **Where it posed, the new system did not jump on the development sequence.** The flips that caused most of the
  original jumps were a decomposition bug, fixed in stage 03b. Of R02's 236 jumps, 225 were exact half-turns.
- **The test sequence's one jump.** The robot turned 16.2° during a 1.88 s recording gap. The tracker kept the old
  orientation at the frame after the gap (labelled supported), then caught up in one step
  `[workshop measurement: theory/workshop/70]`. The post-gap rule (R65-4) only acts in the rotation-only state. On
  the development sequence all five gaps happened to fall there; here both fell during tracking.

### 4.2 Map depth: unavailable as a verdict; on the test sequence also not met where mapped

**Bar.** Depth (a) is the median relative error of map-point depth, checked against the depth images (which the
system never reads), at the alignment's scale. Its bar is the 04-1 mapper's result. Depth (c), the spread of local
scale across keyframes, is reported beside it.

| | Original chain (O0) | Mapper on true poses (04-1) | R04 | Test sequence | O3+ on the test sequence |
|---|---:|---:|---:|---:|---:|
| Depth (a): median abs / signed | 0.909 / −0.907 | 0.144 / −0.062 | **0.067** / −0.045 | **0.291** / −0.195 | 0.153 / −0.100 |
| Depth (c): local-scale spread p90/p10 | 9.2 | 1.53 | **1.04** | **5.25** | 1.56 |

- **Development sequence.** The map was accurate where it existed: 7 % median error, better than the mapper's own
  result on true poses. It covered 2.1 m of path, so this is the evidence that the components worked, not that the
  system did.
- **Test sequence.** **The map's scale was not held.** It was about 2–3.6 m per map unit at the start and about
  11–13 m per unit from frame 643 on: a change of about 4 times `[workshop measurement: theory/workshop/71]`. The
  step falls between keyframes 640 and 660, right after the system kept tracking and mapping through a 51° turn in
  place (frames 571–645). Neither the relocalization at frame 380 nor bundle adjustment caused it. Because no single
  scale fits the map, most posed frames are "wrong" under any one alignment (section 4.4).

### 4.3 Points under rotation: met as the bar was written; the fault occurs on the tracked path

**Bar** (04-4): no map writes from frames the map does not pose. These are the frames in the rotation-only state,
where the original system fabricated a baseline.

| | R04 | Test sequence |
|---|---:|---:|
| Writes, links, point changes, keyframes on frames not posed | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Points promoted while turning in place (all logged) | 245 | 292 |
| Their local depth error, against the rest | **0.129 against 0.025** (5×) | **0.188 against 0.087** (about 2×) |
| Points promoted with less than 1.1° of true parallax: local error against the rest | 0.022 against 0.023 (no difference) | 0.179 against 0.057 |

- **As the bar was written, it is met.** The rotation-only state creates nothing, by construction and as measured.
- **The bar missed the other path.** When the robot turns in place while the tracker still holds a pose, the
  mapper keeps working. Its keyframes then have baselines of a few centimetres, and the points they produce are
  clearly worse `[workshop measurement: theory/workshop/71]`.
  - On the development sequence the map's scale survived this.
  - On the test sequence the scale changed about 4 times right after.
- **Low parallax as such is not the cause.** On the development sequence, points promoted with little true
  parallax are as good as the rest. The cause is turning in place while still tracking.
- **The grade stays "met as operationalized".** The bar was fixed before both runs, and re-grading against a bar
  written afterwards would be fitting the verdict to the result (R71-1). We record the substance next to the grade.

### 4.4 Honest labels: met on the development sequence, not on the test sequence

**Bar.** A frame the system labels as fully posed should not be wrong. Two parts:
- **The post-gap bar:** the frame after a recording gap is not labelled supported while more than 10° off (R65-4).
- **Wrong frames:** a frame is wrong if it is more than 0.10 m or 2° off. Wrong frames are counted under an
  alignment anchored on orientation (R66-1), because a similarity fitted to positions alone can tilt by a few
  degrees on short monocular trajectories.

| | R04 | Test sequence |
|---|---:|---:|
| Post-gap frames supported and over 10° (anchored) | 0 of 5 | **1 of 2** |
| Posed frames wrong (anchored) | 0 of 483 | **370 of 372** |
| Exits from rotation-only / relocalizations that were wrong (anchored) | 0 / 0 | 4 of 4 / 7 of 7 |

On the test sequence both parts fail:
- **The post-gap frame** is the stale pose of section 4.1.
- **The wrong frames** follow from the scale change of section 4.2. No single similarity fits a map whose scale
  changed about 4 times. Within each tracked stretch, orientation stays good (median 0.2°–0.6°)
  `[workshop measurement: theory/workshop/70, 71]`.

### 4.5 Revisit: not claimed

**Bar** (`design.md#goal`): a revisit is detected, its correspondence geometrically verified, and its correction
consistent with the drift measured against ground truth.

- **The repair.** The original loop closer's seeding defect was fixed in stage 01c.
- **The deferral.** Its candidate choice by time was not fixed: the redesign (a Sim(3) loop closure with
  appearance-based candidates) was deferred when the scope was frozen on 2026-09-20.
- **Why it is off.** Re-tested after the mapper changed, the old loop closer acted on the denser map and issued a
  176° wrong correction `[measured: runs/harness/ladder-u]`, so it is off in the final configuration.

The design is in `theory/deferred-designs.md`.

---

## 5. Where the gap is: the final system against its comparators

Each comparator replaces one part of the final system with ground truth, or is an external system. Reading across a
row shows which part costs what. `[measured: runs/04-r04-final-reference/attempt-1, runs/04-ho-holdout-scored-once/
attempt-1, runs/harness/reference-orbslam3]`

| | Final system (R04) | Same tracker on a perfect map (M1a r52) | Same mapper on true poses (O3+) | ORB-SLAM3 monocular (3 runs) |
|---|---|---|---|---|
| **Development sequence** | | | | |
| Full-pose coverage over U | 0.223 | 0.765 | 0.9995 | keyframe span 57–77 % of the sequence, in 3–6 separate maps |
| APE (m) | 0.007 on 483 frames (2.1 m of path) | 0.083 | 0.034 | 0.039 / 0.057 / 0.277 (largest map per run) |
| Depth (a) median error | 0.067 | – (the map is given) | 0.129 | – |
| Map points at the end | 570 | 17,564 (given) | 1,949 | – |
| **Test sequence** | | | | |
| Full-pose coverage over U | 0.127 | 0.655 | 0.9997 | keyframe span 73–76 %, in 6–8 separate maps |
| APE (m) | 0.537 | 0.103 | 0.096 | 0.140 / 0.212 / 0.376 (largest map per run) |
| Depth (a) median error | 0.291 | – | 0.153 | – |
| Map points at the end | 817 | 17,774 (given) | 1,766 | – |

How to read it:
- **Most of the coverage gap is the map, not the tracker.** On a perfect map the same tracker holds a full pose on
  77 % and 66 % of the frames. With the map it builds itself, it reaches 22 % and 13 %. The difference is the map's
  extent and density: the perfect map has 17,000–18,000 points along the whole route, ours has 600–800 in one area.
- **The mapper is not the weak part when poses are right.** On true poses it covers everything and places points
  within 13–15 % of their true depth.
- **The comparison with ORB-SLAM3 is not a ranking.** It reports separate fragments, each aligned on its own, and
  never initializes on the development sequence's first 12.7 s. Our system reports one trajectory with explicit
  labels. Both systems lose track on these sequences. ORB-SLAM3 recovers by starting a new map (its "Atlas"); ours
  waits for the old map to come back into view. `[reasoning]` That difference, not accuracy, is the largest
  difference in coverage.
- **The test sequence is harder for every system.** The perfect-map tracker drops from 0.77 to 0.66, and ORB-SLAM3
  makes more maps with larger errors. Our system dropped further (floor ratio from 0.29 to 0.19), and section 4.2
  gives the reason.

---

## 6. What we learned about this hall and this robot

These are facts about the data, and they hold for any method run on it. They are why the original system failed
the way it did, and why the new one struggles.

| Fact | Value | Consequence | Source |
|---|---|---|---|
| Step size | median 9 mm per frame (p10 5 mm, p90 19 mm) on translating frames | Two consecutive views barely constrain the direction of motion: 31 % of the chain's steps pointed backwards | `[workshop measurement: 03c]` |
| Parallax per keyframe interval | about 0.5° on typical points under forward motion | Below the 1.1° triangulation gate; points need many frames to converge | `theory/workshop/14` |
| Opening | the robot turns in place for the first 243 frames (max true step 2.1 mm) | A map cannot be initialized there. ORB-SLAM3 refuses the first 12.7 s, and our opening waits until frame 588 | `[measured: 03c, reference-orbslam3]` |
| Matching error on repeated texture | about 10 px median along the epipolar line (6 px after undistortion) | Half the disparity a 1.1° gate admits, so the error is one-signed: points come out too close | `[measured: runs/04c, workshop/20]` |
| Radial distortion under forward motion | a one-signed bias of 2–4 % in depth | Removed by undistortion (stage 04d) | `[measured: 04d]` |
| Depth images | 34 % of keyframe keypoints have a valid depth pixel; the perfect map reprojects at 1.7 px median | Bounds what a perfect map can offer | `[measured: 04-H]` |
| True rotation per frame | median 0.57°, max 7.05° on normal pairs; up to 25.6° across recording pauses | The 10° jump bar is meaningful except across pauses | `[workshop measurement: theory/workshop/09]` |
| Dropped frames and pauses | development: 575 of 2,543 intervals over 50 ms, 5 pauses over 150 ms. Test: 1,540 of 2,920 over 50 ms, 2 pauses (18.8 fps against 22.8) | A constant-velocity prediction is wrong across a pause if the robot turned. Where a pause lands decides whether the post-gap rule applies | `[measured: rgb.txt]` |
| When ORB-SLAM3 loses track | every loss on the development sequence comes while driving **and** turning (0.17–0.31 m/s at 14–23°/s) with normal texture | The hard regime for monocular tracking here is combined motion, not low texture | `[measured: reference-orbslam3/tracking.log]` |
| ORB-SLAM3 with depth (RGB-D, as a diagnostic) | one map for 64–103 s of 111 s at 3–9 cm | The scene is trackable when depth is given: the difficulty is monocular geometry, not the scene | `[measured: reference-orbslam3/rgbd-diagnostic]` |

---

## 7. What the test sequence taught us

The test sequence was scored once, after the final configuration had been frozen and eight predictions written
down (R68-8). Nothing was tuned after it was seen. Any analysis done afterwards is labelled as such
(`theory/workshop/70`, `71`).

**Three of the eight predictions held:**
1. byte-identical reruns;
2. loss of track by the map-extent mechanism;
3. no map writes on unposed frames.

**Five failed:**
- the opening map is not wrong;
- no jumps on posed frames;
- depth at least as good as the mapper on true poses;
- no supported post-gap frame;
- no wrong exits or relocalizations.

The five failures share one assumption: *where the system poses, it is right*. That held on the development
sequence and did not transfer.

What this says about the work:
- **A clean result on one sequence can rest on where that sequence's events fall.** On the development sequence,
  every recording pause happened to land in the rotation-only state, where the post-gap rule acts. Its turns in place came either before
  the map existed (frames 16–254) or mostly in the rotation-only state (1095–1199). On the test sequence, both pauses and the main turn in place came while
  the system was tracking. Neither result reflects a different method; they reflect a different arrangement of the
  same events.
- **The fault the bars were built to catch can move to a path the bars do not watch.** Fault 3 was barred on the
  rotation-only state, which is where the original system failed. The new system moved the fault to the tracked
  path (section 4.3).
- **The map-extent finding is robust.** It was predicted, and it appeared on both sequences by the same mechanism.
- **Scoring once worked as intended.** The result is worse than we hoped, and it stands as measured. The
  explanations in section 4 came after the scoring and are labelled as post-hoc diagnoses, not new results.

---

## 8. Open items, and the evidence that would settle each

None of these is planned. Each is written as a design in `theory/deferred-designs.md`, with what it would take and
the prediction it would have to meet.

| Item | What is open | Evidence that would settle it |
|---|---|---|
| **Mapping while turning in place** | Does stopping the mapper during a tracked turn in place remove the test sequence's scale change, at no cost in coverage? | A replay of both sequences with the mapper held off whenever the *estimated* motion is a turn in place, decided online and not from ground truth. Settled if the test sequence's local-scale spread (depth (c)) falls to the mapper's own level (about 1.6), there is no scale step at keyframes 640–660, and development coverage does not drop |
| **Map extent (the stall)** | Can the system recover when it drives out of its map? | Re-initialization of a new map when the old one is out of view, as ORB-SLAM3's Atlas does. Settled if full-pose coverage clears the floor on both sequences without breaking the honest-label bars |
| **Tracking across recording pauses** | Should the constant-velocity prediction be scaled by the actual time gap? | A replay over the 7 pauses of both sequences. Settled if no post-gap frame is supported and over 10°, with J over 10° at zero after pauses (`design.md#open-questions`) |
| **Keyframe policy** | Interval keyframes against ORB-SLAM's visual-change rule | A comparison under both point sources, with its registered prediction: no keyframe on a 55-point opening map (`theory/workshop/62`) |
| **Mixed pose estimator** (1–29 map points) | Does it move the loss of track later? | Its registered predictions: it moves the onset later, but it cannot end the stall. It was deferred untested because it buys coverage without answering any of the four faults (`theory/workshop/63`, `65`) |
| **Mapper depth shortfall** | 0.144 against the 0.122 bar on true poses | Longer seed lifetimes and deferred promotion. The 04-1 analysis shows the error is admitted at promotion by selection (`theory/workshop/19`) |
| **Revisit (fault 4)** | Sim(3) loop closure with appearance-based candidates | Its synthetic test case first: a drifted loop of known error recovered, plus the negative cases. Then the development sequence's revisit around frames 640–900 |
| **Threads** | Tracking, mapping and loop closing on separate threads | A synchronous evaluation mode that reproduces the threaded trajectory at sync points. Deferred: it buys responsiveness, not a result |

---

## 9. How the work was run, and what we would change

**The loop.** Every stage went through the same five roles. Three were AI agents, each started as a fresh session
per call, with a restricted set of tools and a fixed answer format. Two were plain scripts, so no agent could skip or
soften a check: an agent never called the verifier, the controller did.

| Role | Agent or script | What it did | What it could change |
|---|---|---|---|
| Orchestrator | AI agent | Wrote each stage contract: the executable checks and the pass/fail thresholds, fixed before any code | Stage contracts only |
| Theorist | AI agent | Judged each contract before the build, and each finished diff after it, against the method notes in `theory/` | Nothing (read-only) |
| Executor | AI agent | Built the stage | Its own isolated git worktree |
| Verifier | script | Ran the checks and the evaluation, and computed pass or fail | Its own result file |
| Controller | script | Ran the sequence the same way every time. Stopped for me when two roles disagreed, when a check file was missing, or when a stage would change work already accepted | Git and the run records |

**The theory workshop.** Outside the loop, I worked with a separate, interactive AI session to read the papers,
settle the method and write the notes in `theory/` that the theorist judged against. Each ruling (the R-numbers)
was proposed there and took effect only once I ratified it. The theorist never graded against notes it had written
in the same session.

**My part.** The original system was my course project. In this rebuild the agents wrote nearly all of the new code;
I set the direction and made the decisions. After my first repair attempt (August 2026) left accuracy no better, I
designed this loop so that no agent's claim would count until a script had measured it. Within it:
- **Method.** I proposed the experimental method the project ran on: switch one idea at a time against a frozen
  reference. The oracle ladder grew out of it. When the architecture changed, I ruled that the new system must not
  be a copy of ORB-SLAM: each open design choice had to be tested the way the ladder had tested the chain.
- **Rulings.** Every decision of substance was put to me with its alternatives, and took effect only
  once I ratified it: more than 80 rulings, plus a ruling at every gate the controller raised. Some went against
  the agents' recommendation (the initializer's refit test, `workshop/41`), and some of my own proposals were
  dropped when the analysis showed them redundant.
- **Ideas in the final system.** I pushed to test SIFT at once rather than defer it, which led to the front end the
  final system uses. I drafted the depth filter's lifetime rules: dormant seeds instead of dead ones, no failure
  counted on uninformative frames, and no measurement below the matcher's error scale. I also supplied the paper
  that replaced the agents' ad-hoc retirement test. I brought in a Newton-based PnP solver from recent literature
  and asked for its full three-part test against EPnP; EPnP was kept.
- **Course corrections.** When the fallback began gaining one tuned constant per problem, I stopped and asked for
  the problem to be restated structurally. The planned design was withdrawn before it was built, and the fallback
  restructured (`workshop/52`). I froze the scope at stage 04. Later I asked for an impartial review of the project
  and, on its findings, set a hard endpoint.
The finite-difference and map-consistency checks re-ran on every stage, and predictions were registered before
results. `WORKFLOW.md` describes the loop in more detail.

**The record.** 45 stage contracts between 2026-09-17 and 2026-10-11: 33 verified, 12 failed and replanned.
`[workshop reading 2026-10-08, runs/*/attempt-*/orchestrator.json]`:
- **None of the failures was a coding error.** In all of them the executor built what the contract asked.
- **The causes were:**
  - the harness;
  - a wrong bar, premise or test scene in the contract;
  - a real measurement that no bar covered;
  - a design withdrawn before the build.
- **Two verified stages carried a bar that failed by construction.** It was caught only in the harness dry run.

`[reasoning]` The weak link was writing the specification, not building.

**What worked, and we would keep:**
- the verifier is code;
- contracts are locked before the build;
- standing derivative and invariant checks;
- predictions registered before results;
- adverse outcomes committed in advance;
- one sequence held out and scored once.

The oracle ladder worked as **diagnosis**. It found the near-180° flips, the lens-distortion bias and the front end
as the cause of the depth error. An integrated, failing system would have hidden all three.

**What we would change: the development order.**
- **Components were selected on oracles and integrated last.** Each component was chosen by its score under ideal
  conditions (true poses, a perfect map), and the parts were joined only at the end.
- **Under ideal conditions, selection cannot see latency, density or feedback.** The adopted depth filter needs
  about 40 frames to create a point `[measured: 04j]`. That latency is what lost the integrated system at frame
  1248 on the development sequence `[workshop measurement: theory/workshop/60]`.
- **The risk was named early and did not change the plan.** It was named as the integration's first risk on
  2026-09-23 and again, with a registered forecast, on 2026-10-04 (R56-1). Integration still came last (2026-10-06),
  and the risk materialized as forecast.
- **Stage 03 had the better order.** It swapped one oracle at a time *into* a running whole.

`[reasoning]` A crude end-to-end system first, improved at its weakest link with oracle swaps used to find that
link, would likely have surfaced the extent and density problem weeks earlier.

**Planned follow-ups** (not part of this project's program):
1. a retrospective on the workflow, measuring cause, time and cost per failure;
2. a retrospective on this development order;
3. a cross-dataset study of the frozen system beside ORB-SLAM3, one regime at a time.

---

*End of report. Every number and its definition: `theory/expected-performance.md`. Designs not built:
`theory/deferred-designs.md`.*
