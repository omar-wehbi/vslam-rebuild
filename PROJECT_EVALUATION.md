# Project and Remediation Plan Evaluation

Date: 2026-09-08

My assessment is that **you have a substantial experimental prototype, but its core geometry and map consistency are not yet reliable enough to support trustworthy method comparisons.** Your parameterization approach is useful; the difficulty is that several options interact with correctness defects, so changing a parameter can expose or mask a bug rather than reveal an algorithm’s strengths.

**The plan is a useful foundation, but I would revise it before continuing.** It identifies many real problems and records failures honestly. However, it postpones some foundational fixes, misses important defects, and includes several misleading acceptance criteria.

I reviewed the implementation, plan, retained logs and reports, and ran the seven existing tests plus focused numerical checks. The dataset and referenced estimated trajectories/evo outputs are absent from this checkout, so the APE numbers below are **reported results, not independently reproduced measurements**. No implementation changes were made during the review.

You did reach Stage 3:

| Stage | Assessment |
|---|---|
| 0: Baseline | Recorded as complete; some referenced artifacts are missing here. |
| 1: Configuration | Useful experiments, but no demonstrated better default configuration. |
| 2: Tracking | Integrity fixes implemented; scale accuracy remains unresolved. The plan grants a conditional pass, while the report still says FAIL. |
| 3: Mapping | Implemented first attempt; explicitly failed map-continuity criteria. |
| 4–6 | Planned work remains; existing rotation, loop and BA implementations still contain the identified problems. |

The recorded results show the distinction between smoother behavior and greater accuracy:

| Metric | Original baseline | Stage 3 |
|---|---:|---:|
| Sim(3)-aligned APE RMSE | 1.346 m | 1.526 m |
| Maximum trajectory step, internal map units | 28.31 | 2.76 |
| PnP acceptance | 16.5% | 10.2% |
| Loop detections | 1 | 0 |

Stage 3 also reduced nearby point pairs from **408 to 314** against its control, but median observations per point fell from **12 to 8**, and old keyframes still had a median of **one surviving link**. That is meaningful progress on jumps and duplicate reduction, but insufficient evidence of a better SLAM system. The Stage 3 report (kept in the private working repository) largely acknowledges this correctly.

Your strongest work is the experimental infrastructure: interchangeable components, detailed logs, trajectory export, playback, explicit pose conventions, and recorded ablations. These make the system inspectable. The main weaknesses are in the contracts between components.

1. **The optimization mathematics needs fixing before further solver comparisons.**

   BA stores rotations as additive Rodrigues-vector parameters, but its analytical rotation derivative uses a local rotation-perturbation formula without the required conversion. My finite-difference checks found approximately **13.5% and 27.7% relative error** in the rotation block for two nonzero rotations; the other blocks agreed closely. See [optimization.py](optimization.py), line 110.

   The pose-graph Jacobian also failed a numerical check: approximately **59% relative error** on a small example away from zero residuals, consistent across three finite-difference step sizes. See [loop_closure.py](loop_closure.py), line 702.

   These are correctness defects. Tuning iteration limits or comparing optimizers while supplying incorrect derivatives makes the conclusions difficult to trust.

2. **Map association can become internally inconsistent.**

   The frame update first determines triangulation candidates, then performs local-map matching. A candidate keypoint can acquire an existing landmark during that matching, yet default triangulation can subsequently overwrite its link with a new landmark. The safeguard against already-linked keypoints is enabled only in the multi-frame path. See [frame.py](frame.py), line 429, and [mapping.py](mapping.py), line 587.

   Fusion has another concrete problem. I reproduced a merge where two keypoints in the same frame both end up pointing to one landmark, while that landmark’s observation table records only one of them. BA and observation statistics then see different representations of the same association. See [mapping.py](mapping.py), line 279.

   **Association consistency should precede threshold tuning and duplicate reduction.** Fewer nearby points does not establish that the surviving landmarks are correct.

3. **The system still relies heavily on frame-to-frame VO, with intermittent map corrections.**

   PnP is periodic, and most attempts fail. When relative scale is disabled, successful essential-matrix steps generally use unit translation. This limits how faithfully the system can represent changing camera speed between corrections.

   Stage 2’s distributed PnP correction improves smoothness, but it moves historical camera poses without jointly reconciling their landmark observations. That can worsen historical reprojection consistency. See [frame.py](frame.py), line 525.

   I would move toward map-based tracking once initialized, using relative motion as a prediction/fallback, and make historical corrections through an optimization that respects observations.

4. **Some measurements and tests provide weaker evidence than their labels suggest.**

   Stage 3 reports **69 BA successes**, but its log contains only **32 completed BA solves**. `bundle_adjust()` can return without optimizing, and the pipeline still increments success when the unchanged state passes its checks. The reported 82% success rate therefore overstates actual optimization activity. See [pipeline.py](pipeline.py), line 819.

   Likewise, the lost-frame test manually reproduces the fallback calculation rather than exercising the pipeline’s failure path. The “soft rejects outside relative band” test exits on empty matches and never tests that rejection branch. See [test_stage2_tracking.py](tests/test_stage2_tracking.py).

   Five existing tests passed here. Two failed because `np.asarray(..., copy=True)` is incompatible with installed NumPy 1.26, although `numpy>=1.20` permits that environment. The same pattern appears in production point creation. This is a reproducibility defect in the code/dependency contract.

5. **The parameter comparisons need tighter experimental controls.**

   For example, BA’s `gradient_descent` option actually uses conjugate gradient, and the non-LM paths cap points at 80 while LM can use 250. Those runs compare different optimization workloads as well as different solvers. See [optimization.py](optimization.py), line 250.

   Keep the options, but distinguish algorithm comparisons from resource-budget comparisons. A shared configuration definition, saved resolved settings, exact code revision and dependency versions would make this much more defensible.

On the plan, I would preserve its baseline/control/report structure and revise the following points:

- **Move essential BA correctness earlier.** Correct derivatives, scale-gauge handling and preservation of outside observations affect the map that Stages 3 and 5 depend on. They should not wait until the final hardening stage. Add explicit numerical Jacobian acceptance tests to Stage 6; its current “Jacobian honesty” package misses the confirmed rotation-parameter defect.

- **Separate homography selection from pure-rotation detection.** A translating camera viewing a plane can also produce a homography. Stage 4 risks suppressing legitimate translation if selecting H automatically means `t=0`. Add translating-planar and low-parallax cases, plus explicit observability checks. ORB-SLAM’s model-selection procedure distinguishes these cases and delays ambiguous initialization. [ORB-SLAM paper](https://arxiv.org/pdf/1502.00956)

- **Replace “metric scale recovery” with the correct objective.** A monocular map can establish consistent internal scale, but cannot establish metres by itself without additional scale information. Fixing BA’s gauge stabilizes that internal coordinate system. Separately, Stage 5 should address scale drift explicitly: an SE(3) graph cannot directly represent scale corrections; Sim(3) loop correction is the relevant extension if that remains in scope. [ORB-SLAM’s scale-aware loop formulation](https://arxiv.org/pdf/1502.00956)

- **Do not require every valid loop to produce a large correction.** A correct revisit can need almost no correction if drift is small. Validate geometric consistency and false positives; use deliberately drifted synthetic examples to prove that correction works. Also, using the current pose as a PnP seed is not, by itself, proof that PnP must return that pose—the plan overstates this causal explanation.

- **Strengthen acceptance metrics.** “Step lengths are not constant,” “median observations increased,” and “duplicate density decreased” can each improve without improving reconstruction. Retain them as diagnostics, but add relative pose error, rotation error, valid-tracking coverage, recovery behavior, and geometrically verified association quality. Compare step magnitudes with scale accounted for. Select a hold-out sequence now and use repeated runs to distinguish small gains from variation.

- **Make the evaluation record reproducible and internally consistent.** Reconcile Stage 2’s plan/report status. Restore the referenced report template and retain trajectories/evo outputs alongside reports. A frozen command using defaults is insufficient when those defaults later change. The historical baseline needs an exact revision and resolved configuration.

One detail in the underlying diagnosis also deserves correction: a constant behind-camera residual has a zero derivative within that region. The problem is its discontinuity and lack of a useful recovery gradient, not an inherent disagreement between a constant residual and zero Jacobian.

**My recommended next work is a foundation repair pass before proceeding to Stage 4 or 5:** fix compatibility and metric accounting; verify both optimizers numerically; enforce bidirectional landmark-observation consistency; then repeat Stage 3 with controlled ablations. After that, tackle initialization/degeneracy and loop closure.

I would retain the codebase and its configurable methods. The key change is to establish a small, trustworthy reference pipeline first, then make each experimental option demonstrate its effect against that reference.
