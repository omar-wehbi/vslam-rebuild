"""Stage 04j (ruled 2026-09-22, S1-S6, workshop/20): the SIFT front end, existing
switches only.

S-a sigma_px for SIFT. local_mapping._keypoint_sigmas gives sigma = 2 ** max(octave, 0):
    octave -1 (the upsampled scale-space level) and octave 0 -> 1 px, octave 1 -> 2 px,
    octave 2 -> 4 px. KEPT, and stated as the regime quantity: the workshop's matcher
    decomposition measured SIFT's correct-class along-line scatter |along| p50 = 1.30 to
    1.79 px on this hall at octaves -1 and 0, which hold 91 percent of its matches
    (workshop/20 addenda). A sigma below 1 px at octave -1 would state a precision the
    matcher does not have, and the pair mapper's chi-square band and the depth filter's
    measurement variance would then refuse correct matches. local_mapping.py is not edited.

S-b the depth (a) bar for SIFT. m_fo(theta_min, sigma) = 0.6745 * sqrt(2) * sigma /
    (f * theta_min), f = 521 px, theta_min = 1.1 deg; with sigma = 1 px (the median created
    octave in the pre-test is -1, and 1,529 of 2,627 creations sat there):
    .venv/bin/python -c "import math; print(0.6745*math.sqrt(2)*1.0/(521*math.radians(1.1)))"
    = 0.0954. The ruled bar stays 0.122 (R4; S3: the cap is not loosened). The derived SIFT
    figure 0.0954 is tighter than the cap and is RECORDED next to it per arm; which figure an
    adoption uses is the human's ruling after 04k (S6).

S-c the descriptor protocol. An ORB descriptor threshold maps to the SIFT threshold that
    accepts the same fraction of nearest-neighbour candidates on the frozen calibration
    pairs, computed once and frozen (see CALIBRATION_PAIRS, calibrate_front_end.py). The
    SIFT row of PROTOCOL below is the output of calibrate_front_end.py on the primary
    dataset, run 2026-09-22, reproducing the workshop's bound of 248 (within 247-249):
      orb_rate_desc = 0.548011 (ORB Hamming<=50 acceptance)
      sift_desc_bound = 248 (SIFT L2 bound at the same acceptance rate)
      orb_rate_track  = 0.326420 (ORB accepted at the frame-to-frame ratio 0.75)
      sift_track_ratio = 0.766 (SIFT ratio at the same acceptance rate)
      orb_rate_reloc  = 0.449716 (ORB accepted at the relocalization ratio 0.85)
      sift_reloc_ratio = 0.876 (SIFT ratio at the same acceptance rate)
    Stated and NOT changed (the design names the three items above; the pre-test ran with
    these; changing them would break the reproduction arm (a) is meant to be): the local-map
    search bound (mapping.py, the SIFT default 150 L2 since Stage 3), the fusion bound
    (--fusion_max_desc_dist 40, applied as an L2 distance under SIFT, so fusion merges only
    near-identical SIFT descriptors), relocalization's count thresholds (25 matches, 10 2D-3D
    pairs, 10 PnP inliers, 5.0 from the keyframe) and estimate_motion's RANSAC threshold and
    count floors (5 matches, 5 inliers).

S-d why the ratio tests are the tracker's thresholds. The frame-to-frame tracker has no
    absolute descriptor threshold: match_frames applies Lowe's ratio test only, and
    relocalization applies a ratio test and counts. A ratio is dimensionless but not
    detector-independent (the two detectors' distance distributions differ), so the protocol
    rate-matches it like the bound. The pre-test's APE miss is attributed to relocalization on
    VO-failure frames; this contract applies the protocol and the harness MEASURES the
    result (APE is barred per arm). No value here is tuned against APE or against any run
    output; no frame numbers appear anywhere except the four calibration pairs.
"""

import numpy as np
import cv2


PROTOCOL = {
    "ORB": {"desc_bound": 50.0, "track_ratio": 0.75, "reloc_ratio": 0.85},
    # calibrate_front_end.py on the primary dataset, 2026-09-22, at ORB acceptance rates
    # q_desc=0.548011, q_track=0.326420, q_reloc=0.449716 (S-c above).
    "SIFT": {"desc_bound": 248.0, "track_ratio": 0.766, "reloc_ratio": 0.876},
}

CALIBRATION_PAIRS = ((560, 600), (700, 740), (1300, 1340), (2320, 2360))
ORB_CALIBRATION_NFEATURES = 1000


def track_ratio(method):
    if method == "SIFT":
        return PROTOCOL["SIFT"]["track_ratio"]
    return PROTOCOL["ORB"]["track_ratio"]


def preset_argv(name):
    if name == "orb":
        return []
    if name == "sift":
        S = PROTOCOL["SIFT"]
        B = f"{S['desc_bound']:g}"
        R = f"{S['reloc_ratio']:g}"
        return [
            "--method", "SIFT",
            "--lm_max_desc_dist", B,
            "--df_max_desc_dist", B,
            "--relocalization_ratio_threshold", R,
        ]
    raise ValueError(f"unknown front-end preset: {name!r}")


def expand_argv(argv):
    argv = list(argv)
    idx = None
    val = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--front_end" and i + 1 < len(argv):
            idx = i
            val = argv[i + 1]
            i += 2
            continue
        if tok.startswith("--front_end="):
            idx = i
            val = tok.split("=", 1)[1]
            i += 1
            continue
        i += 1
    if idx is None or val not in ("sift", "orb"):
        return list(argv)
    return preset_argv(val) + list(argv)


def acceptance_at_bound(d1, bound):
    d1 = np.asarray(d1, dtype=np.float64)
    return float(np.mean(d1 <= bound))


def bound_at_rate(d1, q):
    d1 = np.asarray(d1, dtype=np.float64)
    return float(np.quantile(d1, q))


def acceptance_at_ratio(d1, d2, r):
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    return float(np.mean((d2 > 0) & (d1 < r * d2)))


def ratio_at_rate(d1, d2, q):
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    mask = d2 > 0
    return float(np.quantile(d1[mask] / d2[mask], q))


def nn_distances(desc_a, desc_b, norm):
    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    d1 = []
    d2 = []
    for pair in knn:
        if len(pair) < 2:
            continue
        d1.append(pair[0].distance)
        d2.append(pair[1].distance)
    return np.asarray(d1, dtype=np.float64), np.asarray(d2, dtype=np.float64)


def calibrate(pairs):
    orb = cv2.ORB_create(nfeatures=ORB_CALIBRATION_NFEATURES)
    sift = cv2.SIFT_create()

    orb_d1_all = []
    orb_d2_all = []
    sift_d1_all = []
    sift_d2_all = []

    for img_a, img_b in pairs:
        ka, da = orb.detectAndCompute(img_a, None)
        kb, db = orb.detectAndCompute(img_b, None)
        d1, d2 = nn_distances(da, db, cv2.NORM_HAMMING)
        orb_d1_all.append(d1)
        orb_d2_all.append(d2)

        ka, da = sift.detectAndCompute(img_a, None)
        kb, db = sift.detectAndCompute(img_b, None)
        d1, d2 = nn_distances(da, db, cv2.NORM_L2)
        sift_d1_all.append(d1)
        sift_d2_all.append(d2)

    orb_d1 = np.concatenate(orb_d1_all)
    orb_d2 = np.concatenate(orb_d2_all)
    sift_d1 = np.concatenate(sift_d1_all)
    sift_d2 = np.concatenate(sift_d2_all)

    orb_rate_desc = acceptance_at_bound(orb_d1, PROTOCOL["ORB"]["desc_bound"])
    sift_desc_bound = int(round(bound_at_rate(sift_d1, orb_rate_desc)))

    orb_rate_track = acceptance_at_ratio(orb_d1, orb_d2, PROTOCOL["ORB"]["track_ratio"])
    sift_track_ratio = round(ratio_at_rate(sift_d1, sift_d2, orb_rate_track), 3)

    orb_rate_reloc = acceptance_at_ratio(orb_d1, orb_d2, PROTOCOL["ORB"]["reloc_ratio"])
    sift_reloc_ratio = round(ratio_at_rate(sift_d1, sift_d2, orb_rate_reloc), 3)

    return {
        "orb_rate_desc": orb_rate_desc,
        "sift_desc_bound": sift_desc_bound,
        "orb_rate_track": orb_rate_track,
        "sift_track_ratio": sift_track_ratio,
        "orb_rate_reloc": orb_rate_reloc,
        "sift_reloc_ratio": sift_reloc_ratio,
        "n_orb": int(orb_d1.size),
        "n_sift": int(sift_d1.size),
        "n_sift_d2_zero": int(np.sum(sift_d2 == 0)),
    }
