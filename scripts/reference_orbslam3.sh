#!/usr/bin/env bash
# Stage R (outside the loop): ORB-SLAM3 monocular on the primary sequence, scored with the
# harness's evo command. See theory/design.md "Reference-system run" and workshop/08.
#
#   bash scripts/reference_orbslam3.sh [build|run|score|all|rgbd]   (default: all)
#   rgbd = DIAGNOSTIC ONLY (not the reference; the project is monocular by goal): RGB-D mode on the
#          same sequence, to separate 'monocular geometry fails here' from 'the scene fails here'.
#
# Builds under .deps/ORB_SLAM3 (gitignored) with -j2 (7 GB RAM box), runs mono_tum N times
# (ORB-SLAM3 is nondeterministic run to run: free-running threads), keeps every run's
# KeyFrameTrajectory.txt and stdout, and scores each with evo exactly as the verifier does.
# Outputs: runs/harness/reference-orbslam3/{build.log,run-N/,summary.txt}.
#
# Patches applied to upstream (recorded here so the R record can say so):
#   1. C++17 everywhere (Pangolin 0.9 headers need it; upstream says c++11/14).
#   2. LoopClosing.h: Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3>> (GCC >= 11 fix).
#   3. mono_tum.cc: viewer off (headless), so the run can go in the background.
#   4. Sophus built without its tests/examples (they fail under GCC 13 -Werror; library is header-only).
#   5. LoopClosing.cc: mnFullBAIdx++ on a bool -> mnFullBAIdx = true (same effect; bool++ is a C++17 error).
#   6. Output only: SaveAllMapsKeyFrameTrajectoryTUM writes every atlas map's keyframe trajectory
#      (KeyFrameTrajectory_map<id>.txt) and an index (KeyFrameTrajectory_maps.txt); upstream saves
#      only the current map, which is empty if the last map was reset before shutdown.
#   7. rgbd_tum.cc (diagnostic only): viewer off + the all-maps save, same as mono_tum.
# No algorithmic change. Keyframe trajectory only (mono SaveTrajectoryTUM is not available upstream).
#
# Environment switches (04-ho task H, 2026-10-10; with none set, every mode behaves as before):
#   REF_DATASET=primary|holdout  eval.dataset_root (default) or eval.holdout_root of system/config.yaml
#   REF_OUT=<dir>                output directory (default runs/harness/reference-orbslam3)
#   REF_SRC=<dir>                the built ORB_SLAM3 tree (default .deps/ORB_SLAM3)
#   REF_DATASET_DIR=<dir>        a TUM sequence directory; overrides REF_DATASET and needs no config file
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="$ROOT/.deps"
SRC="${REF_SRC:-$DEPS/ORB_SLAM3}"
OUT="${REF_OUT:-$ROOT/runs/harness/reference-orbslam3}"
REF_DATASET="${REF_DATASET:-primary}"
CFG="$ROOT/system/config.yaml"
PY="$ROOT/.venv/bin/python"
N_RUNS="${N_RUNS:-3}"
JOBS="${JOBS:-2}"
MODE="${1:-all}"
mkdir -p "$OUT" "$DEPS"

cfg() { "$PY" - "$CFG" "$1" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
node = cfg
for k in sys.argv[2].split("."):
    node = (node or {}).get(k)
print(" ".join(map(str, node)) if isinstance(node, list) else ("" if node is None else node))
EOF
}
if [[ -n "${REF_DATASET_DIR:-}" ]]; then
  # No harness config needed: the TUM sequence directory and the verifier's evo flags.
  DATASET="$REF_DATASET_DIR"
  GT="$DATASET/groundtruth.txt"
  EVO_FLAGS="-va --align --correct_scale"
else
  case "$REF_DATASET" in
    primary) DATASET="$(cfg eval.dataset_root)";;
    holdout) DATASET="$(cfg eval.holdout_root)";;
    *) echo "REF_DATASET must be primary or holdout, not '$REF_DATASET'"; exit 1;;
  esac
  GT="$DATASET/$(cfg eval.groundtruth_file)"
  EVO_FLAGS="$(cfg eval.evo_flags)"
fi
EVO_APE="$ROOT/.venv/bin/evo_ape"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

build() {
  if [[ ! -d "$SRC/.git" ]]; then
    log "cloning ORB_SLAM3"
    git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$SRC"
  fi
  cd "$SRC"
  git rev-parse HEAD > "$OUT/upstream_commit.txt"
  log "patching (c++17, LoopClosing allocator, headless mono_tum)"
  # 1. C++ standard: every CMakeLists in the tree.
  find . -name CMakeLists.txt -exec sed -i -E 's/-std=c\+\+(11|14)/-std=c++17/g; s/CMAKE_CXX_STANDARD (11|14)/CMAKE_CXX_STANDARD 17/g' {} +
  # 2. GCC >= 11 allocator key constness.
  sed -i 's/std::pair<const KeyFrame\*, g2o::Sim3>/std::pair<KeyFrame* const, g2o::Sim3>/g' include/LoopClosing.h
  # 3. Headless mono_tum: System(..., MONOCULAR, true) -> false.
  sed -i -E 's/(ORB_SLAM3::System SLAM\(argv\[1\],argv\[2\],ORB_SLAM3::System::MONOCULAR,)\s*true\)/\1false)/' Examples/Monocular/mono_tum.cc
  grep -q "MONOCULAR,false)" Examples/Monocular/mono_tum.cc || { echo "mono_tum viewer patch did not apply"; exit 2; }
  # 5. Upstream increments a bool member (mnFullBAIdx++), forbidden in C++17; bool++ set it to true.
  sed -i 's/mnFullBAIdx++;/mnFullBAIdx = true;/g' src/LoopClosing.cc
  # 6. Output only: save EVERY map's keyframe trajectory plus a coverage index. Upstream's
  #    SaveKeyFrameTrajectoryTUM writes the CURRENT map only, which is empty when the last map was
  #    reset before shutdown (observed on pioneer_slam3: seven maps, last one reset).
  if ! grep -q SaveAllMapsKeyFrameTrajectoryTUM include/System.h; then
    sed -i 's|    void SaveKeyFrameTrajectoryTUM(const string &filename);|    void SaveKeyFrameTrajectoryTUM(const string \&filename);\n    void SaveAllMapsKeyFrameTrajectoryTUM(const string \&prefix);  // patch 6: all atlas maps + index|' include/System.h
    grep -q SaveAllMapsKeyFrameTrajectoryTUM include/System.h || { echo "System.h patch did not apply"; exit 2; }
    cat >> src/System.cc <<'CPP'

// patch 6 (V-SLAM stage R): keyframe trajectory of every map in the atlas, one TUM file per map,
// plus <prefix>_maps.txt: map_id n_keyframes n_good first_ts last_ts is_current
namespace ORB_SLAM3 {
void System::SaveAllMapsKeyFrameTrajectoryTUM(const string &prefix)
{
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pCur = mpAtlas->GetCurrentMap();
    ofstream idx((prefix + "_maps.txt").c_str());
    idx << "# map_id n_keyframes n_good first_ts last_ts is_current" << endl;
    for(Map* pMap : vpMaps)
    {
        vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
        ofstream f((prefix + "_map" + to_string(pMap->GetId()) + ".txt").c_str());
        f << fixed;
        int good = 0; double t0 = -1.0, t1 = -1.0;
        for(KeyFrame* pKF : vpKFs)
        {
            if(pKF->isBad()) continue;
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2)
              << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            if(t0 < 0) t0 = pKF->mTimeStamp;
            t1 = pKF->mTimeStamp; good++;
        }
        f.close();
        idx << pMap->GetId() << " " << vpKFs.size() << " " << good << " " << fixed << setprecision(6) << t0 << " " << t1 << " " << (pMap == pCur ? 1 : 0) << endl;
    }
    idx.close();
}
} // namespace ORB_SLAM3
CPP
    sed -i 's|    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");|    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");\n    SLAM.SaveAllMapsKeyFrameTrajectoryTUM("KeyFrameTrajectory");  // patch 6|' Examples/Monocular/mono_tum.cc
    grep -q SaveAllMapsKeyFrameTrajectoryTUM Examples/Monocular/mono_tum.cc || { echo "mono_tum patch 6 did not apply"; exit 2; }
  fi
  # 7. rgbd_tum: headless + all-maps save (diagnostic mode only).
  sed -i -E 's/(ORB_SLAM3::System SLAM\(argv\[1\],argv\[2\],ORB_SLAM3::System::RGBD,)\s*true\)/\1false)/' Examples/RGB-D/rgbd_tum.cc
  grep -q SaveAllMapsKeyFrameTrajectoryTUM Examples/RGB-D/rgbd_tum.cc || sed -i 's|    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");|    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");\n    SLAM.SaveAllMapsKeyFrameTrajectoryTUM("KeyFrameTrajectory");  // patch 7|' Examples/RGB-D/rgbd_tum.cc
  grep -q "RGBD,false)" Examples/RGB-D/rgbd_tum.cc || { echo "rgbd_tum viewer patch did not apply"; exit 2; }
  git diff --stat > "$OUT/patches.diffstat.txt"; git diff > "$OUT/patches.diff"

  for lib in DBoW2 g2o Sophus; do
    log "building Thirdparty/$lib"
    extra=""
    # 4. Sophus: its own tests and examples fail under GCC 13 with -Werror (array-bounds); the
    #    header-only library is all ORB-SLAM3 needs.
    [[ "$lib" == Sophus ]] && extra="-DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF"
    mkdir -p "Thirdparty/$lib/build"; ( cd "Thirdparty/$lib/build" && cmake .. -DCMAKE_BUILD_TYPE=Release $extra > /dev/null && make -j"$JOBS" )
  done
  if [[ ! -f Vocabulary/ORBvoc.txt ]]; then
    log "uncompressing vocabulary"; ( cd Vocabulary && tar -xf ORBvoc.txt.tar.gz )
  fi
  log "building ORB_SLAM3 (-j$JOBS)"
  mkdir -p build; ( cd build && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j"$JOBS" )
  test -x Examples/Monocular/mono_tum || { echo "mono_tum not built"; exit 2; }
  log "build ok"
}

run() {
  cd "$SRC"
  for i in $(seq 1 "$N_RUNS"); do
    R="$OUT/run-$i"; mkdir -p "$R"
    log "run $i/$N_RUNS: mono_tum on $DATASET"
    T0=$(date +%s)
    ( cd "$R" && "$SRC/Examples/Monocular/mono_tum" "$SRC/Vocabulary/ORBvoc.txt" "$SRC/Examples/Monocular/TUM2.yaml" "$DATASET" > tracking.log 2>&1 ) || log "run $i exited non-zero (see $R/tracking.log)"
    T1=$(date +%s)
    echo "run_seconds=$((T1-T0))" > "$R/timing.txt"
    test -s "$R/KeyFrameTrajectory_maps.txt" || { log "run $i produced no map index"; continue; }
    log "run $i done in $((T1-T0)) s: $(grep -vc '^#' "$R/KeyFrameTrajectory_maps.txt") maps, current-map keyframes $(grep -vc '^#' "$R/KeyFrameTrajectory.txt" || true)"
  done
}

score() {
  set +e; set +o pipefail  # scoring is all greps and pipes; a closed pipe (grep | head) must not abort it
  SEQ_T0=$(grep -v '^#' "$DATASET/rgb.txt" | head -1 | awk '{print $1}'); SEQ_T1=$(grep -v '^#' "$DATASET/rgb.txt" | tail -1 | awk '{print $1}')
  {
    echo "# ORB-SLAM3 reference run (stage R), $(date '+%Y-%m-%d %H:%M')"
    echo "upstream: $(cat "$OUT/upstream_commit.txt" 2>/dev/null); settings: Examples/Monocular/TUM2.yaml; dataset: $DATASET"
    echo "evo per map: evo_ape tum GT KeyFrameTrajectory_map<id>.txt $EVO_FLAGS   (same flags as the verifier; each map is aligned on its own, which is generous)"
    echo "frames in sequence: $(grep -vc '^#' "$DATASET/rgb.txt"); sequence time span: $SEQ_T0 .. $SEQ_T1 ($(awk -v a="$SEQ_T0" -v b="$SEQ_T1" 'BEGIN{printf "%.1f", b-a}') s)"
    echo
    echo "## Per run"
    echo
    echo "| run | seconds | maps | keyframes (all maps, good) | covered span s | largest map kfs | largest map span s | largest map ape_rmse_m | largest map sim3_scale | new-map events | relocalizations | loop/merge lines |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
    for i in $(seq 1 "$N_RUNS"); do
      R="$OUT/run-$i"; [[ -s "$R/KeyFrameTrajectory_maps.txt" ]] || { echo "| $i | (no map index) |"; continue; }
      best=""; bestn=0; total=0; span=0
      while read -r mid nkf ngood t0 t1 cur; do
        [[ "$mid" == \#* ]] && continue
        total=$((total + ngood))
        (( ngood >= 2 )) && span=$(awk -v s="$span" -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", s + (b-a)}')
        if (( ngood > bestn )); then bestn=$ngood; best=$mid; fi
      done < "$R/KeyFrameTrajectory_maps.txt"
      ape="n/a"; sc="n/a"; bspan="n/a"
      if [[ -n "$best" ]]; then
        "$EVO_APE" tum "$GT" "$R/KeyFrameTrajectory_map$best.txt" $EVO_FLAGS > "$R/evo_ape_map$best.txt" 2>&1 || true
        ape=$(grep -E "^\s*rmse\s+[0-9.eE+-]+" "$R/evo_ape_map$best.txt" | awk '{print $2}' | head -1); ape=${ape:-n/a}
        sc=$(grep -oE "Scale correction:\s*[0-9.eE+-]+" "$R/evo_ape_map$best.txt" | awk '{print $3}'); sc=${sc:-n/a}
        bspan=$(awk -v m="$best" '$1==m {printf "%.1f", $5-$4}' "$R/KeyFrameTrajectory_maps.txt")
      fi
      nmaps=$(grep -vc '^#' "$R/KeyFrameTrajectory_maps.txt")
      newmaps=$(grep -c "Creation of new map with id" "$R/tracking.log" || true)
      reloc=$(grep -c "Relocalized" "$R/tracking.log" || true)
      loops=$(grep -ciE "loop detected|merge detected|loop closed|Loop closure" "$R/tracking.log" || true)
      echo "| $i | $(cut -d= -f2 "$R/timing.txt") | $nmaps | $total | $span | $bestn (map $best) | $bspan | $ape | $sc | $newmaps | $reloc | $loops |"
    done
    echo
    echo "## Per map (every run)"
    echo
    echo "| run | map | keyframes | good | first ts | last ts | span s | ape_rmse_m | sim3_scale | current at exit |"
    echo "|---|---|---|---|---|---|---|---|---|---|"
    for i in $(seq 1 "$N_RUNS"); do
      R="$OUT/run-$i"; [[ -s "$R/KeyFrameTrajectory_maps.txt" ]] || continue
      while read -r mid nkf ngood t0 t1 cur; do
        [[ "$mid" == \#* ]] && continue
        ape="n/a"; sc="n/a"
        if (( ngood >= 3 )); then
          "$EVO_APE" tum "$GT" "$R/KeyFrameTrajectory_map$mid.txt" $EVO_FLAGS > "$R/evo_ape_map$mid.txt" 2>&1 || true
          ape=$(grep -E "^\s*rmse\s+[0-9.eE+-]+" "$R/evo_ape_map$mid.txt" | awk '{print $2}' | head -1); ape=${ape:-n/a}
          sc=$(grep -oE "Scale correction:\s*[0-9.eE+-]+" "$R/evo_ape_map$mid.txt" | awk '{print $3}'); sc=${sc:-n/a}
        fi
        echo "| $i | $mid | $nkf | $ngood | $t0 | $t1 | $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (a<0)?0:b-a}') | $ape | $sc | $cur |"
      done < "$R/KeyFrameTrajectory_maps.txt"
    done
    echo
    echo "Patches to upstream: see patches.diff (C++17, allocator constness, viewer off, Sophus tests off, bool increment, all-maps save). No algorithmic change. Keyframe trajectories only."
  } > "$OUT/summary.txt"
  log "summary written: $OUT/summary.txt"
  cat "$OUT/summary.txt"
}


# ---- DIAGNOSTIC, NOT REFERENCE: RGB-D mode on the same sequence ------------------------------- #
rgbd() {
  D="$OUT/rgbd-diagnostic"; mkdir -p "$D"
  ASSOC="$D/associations.txt"
  # associate.py writes to -o and prints a header line; rgbd_tum wants bare 4-column lines.
  [[ -s "$ASSOC" ]] || { ( cd "$D" && "$PY" "$ROOT/associate.py" "$DATASET/rgb.txt" "$DATASET/depth.txt" -o assoc.raw.txt > /dev/null ) && grep -v '^#' "$D/assoc.raw.txt" > "$ASSOC" && rm -f "$D/assoc.raw.txt"; }
  cd "$SRC"
  log "rebuilding rgbd_tum (patch 7)"
  ( cd build && make -j"$JOBS" rgbd_tum > /dev/null )
  for i in $(seq 1 "$N_RUNS"); do
    R="$D/run-$i"; mkdir -p "$R"
    log "rgbd run $i/$N_RUNS"
    T0=$(date +%s)
    ( cd "$R" && "$SRC/Examples/RGB-D/rgbd_tum" "$SRC/Vocabulary/ORBvoc.txt" "$SRC/Examples/RGB-D/TUM2.yaml" "$DATASET" "$ASSOC" > tracking.log 2>&1 ) || log "rgbd run $i exited non-zero"
    echo "run_seconds=$(( $(date +%s) - T0 ))" > "$R/timing.txt"
    log "rgbd run $i done: $(grep -vc '^#' "$R/KeyFrameTrajectory_maps.txt" 2>/dev/null || echo 0) maps, $(grep -vc '^#' "$R/CameraTrajectory.txt" 2>/dev/null || echo 0) frame poses"
  done
  set +e; set +o pipefail
  NASSOC=$(grep -vc '^#' "$ASSOC")
  {
    echo "# ORB-SLAM3 RGB-D run on pioneer_slam3: DIAGNOSTIC ONLY, NOT A REFERENCE ($(date '+%Y-%m-%d %H:%M'))"
    echo
    echo "Purpose: separate 'monocular geometry fails on this sequence' from 'the scene fails'. The project is"
    echo "monocular by goal (design.md#goal); nothing here is a comparator for any stage. Same binary tree and"
    echo "patches as the monocular reference run; settings Examples/RGB-D/TUM2.yaml unchanged (DepthMapFactor 5208)."
    echo "Association: associate.py rgb.txt depth.txt (default 0.02 s tolerance): $NASSOC pairs of $(grep -vc '^#' "$DATASET/rgb.txt") rgb frames."
    echo
    echo "| run | seconds | maps | frame poses saved / associated pairs | keyframes | largest map kfs, span s | frame APE RMSE m, Sim(3) (harness flags) | frame APE RMSE m, SE(3) (--align only) | sim3 scale | new-map events | relocalizations | loop/merge lines |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
    for i in $(seq 1 "$N_RUNS"); do
      R="$D/run-$i"; [[ -s "$R/CameraTrajectory.txt" ]] || { echo "| $i | (no trajectory) |"; continue; }
      "$EVO_APE" tum "$GT" "$R/CameraTrajectory.txt" $EVO_FLAGS > "$R/evo_ape_sim3.txt" 2>&1
      "$EVO_APE" tum "$GT" "$R/CameraTrajectory.txt" -va --align > "$R/evo_ape_se3.txt" 2>&1
      a1=$(grep -E "^\s*rmse\s+[0-9.eE+-]+" "$R/evo_ape_sim3.txt" | awk '{print $2}' | head -1)
      a2=$(grep -E "^\s*rmse\s+[0-9.eE+-]+" "$R/evo_ape_se3.txt" | awk '{print $2}' | head -1)
      sc=$(grep -oE "Scale correction:\s*[0-9.eE+-]+" "$R/evo_ape_sim3.txt" | awk '{print $3}')
      nposes=$(grep -vc '^#' "$R/CameraTrajectory.txt")
      nmaps=$(grep -vc '^#' "$R/KeyFrameTrajectory_maps.txt"); nkf=$(awk 'NR>1{s+=$3} END{print s+0}' "$R/KeyFrameTrajectory_maps.txt")
      best=$(awk 'NR>1 && $3>m {m=$3; b=$1; sp=$5-$4} END{printf "%d (map %s), %.1f", m, b, sp}' "$R/KeyFrameTrajectory_maps.txt")
      newmaps=$(grep -c "Creation of new map with id" "$R/tracking.log"); reloc=$(grep -c "Relocalized" "$R/tracking.log")
      loops=$(grep -ciE "loop detected|merge detected|loop closed|Loop closure" "$R/tracking.log")
      echo "| $i | $(cut -d= -f2 "$R/timing.txt") | $nmaps | $nposes / $NASSOC | $nkf | $best | ${a1:-n/a} | ${a2:-n/a} | ${sc:-n/a} | $newmaps | $reloc | $loops |"
    done
    echo
    echo "Per map:"
    echo
    echo "| run | map | keyframes | first ts | last ts | span s | current at exit |"
    echo "|---|---|---|---|---|---|---|"
    for i in $(seq 1 "$N_RUNS"); do
      R="$D/run-$i"; [[ -s "$R/KeyFrameTrajectory_maps.txt" ]] || continue
      awk -v r="$i" 'NR>1 {printf "| %s | %s | %s | %s | %s | %.1f | %s |\n", r, $1, $3, $4, $5, ($4<0)?0:$5-$4, $6}' "$R/KeyFrameTrajectory_maps.txt"
    done
    echo
    echo "Note: CameraTrajectory.txt (frame poses) is what upstream's SaveTrajectoryTUM writes for RGB-D: poses of the"
    echo "CURRENT map's frames relative to their reference keyframes; frames of earlier maps are not included when the atlas"
    echo "fragmented, so 'frame poses saved / associated pairs' is the coverage of the final map only."
  } > "$D/summary.txt"
  log "rgbd diagnostic summary: $D/summary.txt"
  cat "$D/summary.txt"
}

case "$MODE" in
  build) build;;
  run) run;;
  score) score;;
  all) build; run; score;;
  rgbd) build; rgbd;;
  *) echo "usage: $0 [build|run|score|all]"; exit 1;;
esac
log "DONE ($MODE)"
