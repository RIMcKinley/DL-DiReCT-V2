#!/bin/bash
#
# prep-samseg.sh -- build a --space cropped prep from SAMSEG instead of DeepSCAN.
#
# SAMSEG is run with --save-posteriors, so the prep carries real per-structure
# probabilities and nothing has to invent logits. See dldirect/prep_from_seg.py
# for why the logit route is avoided and for the aseg -> DeepSCAN id mapping.
#
# The segmenter runs on the ALREADY CROPPED T1, so its output shares that grid
# and no resampling happens between the segmentation and the surfaces. That is
# the same single-grid discipline as --space cropped elsewhere.
#
# Usage:
#   prep-samseg.sh [options] T1_STRIPPED MASK OUT_DIR
#     T1_STRIPPED  skull-stripped, conformed (LIA 1mm) T1
#     MASK         brain mask defining the crop (T1_STRIPPED works as its own)
#     OUT_DIR      prep directory to create
#
# Options:
#   --threads N       SAMSEG threads (default 8). CPU-only, but not slow on a
#                     cropped volume: measured 2 min 59 s end to end at
#                     --threads 8 on OAS30001_ses-d0129_run-01 (127x124x159),
#                     against ~25 s for the DeepSCAN path and ~1 min for
#                     SynthSeg on the GPU.
#   --lesion          use the lesion atlas (adds --lesion to run_samseg)
#   --binary          ignore the posteriors; use the hard mask as the
#                     probability. Cheaper, but the GM/WM isosurfaces then land
#                     on a voxel staircase instead of at partial volume.
#   --keep-samseg     keep SAMSEG's own output directory (default: kept anyway
#                     under OUT_DIR/samseg, this is a no-op kept for symmetry)
#   --skip-existing   do nothing if the prep already has label_def.csv

set -u
FREESURFER_HOME=${FREESURFER_HOME:-/data/disk2/freesurfer}
PKG=$(dirname $(dirname $(readlink -f "$0")))   # .../dldirect
REPO=$(dirname "$PKG")                          # repo root, for -m dldirect.*
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PY=${PY:-/home/student/miniconda3/envs/DL_DiReCT/bin/python}
THREADS=8
LESION=""
BINARY=""
SKIP=0

die() { echo "ERROR: $1" >&2; exit 1; }

while [[ $# -gt 3 ]]; do
	case "$1" in
		--threads)       THREADS=$2; shift ;;
		--lesion)        LESION="--lesion" ;;
		--binary)        BINARY="--binary" ;;
		--keep-samseg)   ;;
		--skip-existing) SKIP=1 ;;
		*)               die "unknown option $1" ;;
	esac
	shift
done
[[ $# -eq 3 ]] || { sed -n '3,30p' "$0"; exit 1; }

T1=$1
MASK=$2
OUT=$3

[[ -f "$T1" ]]   || die "no such T1: $T1"
[[ -f "$MASK" ]] || die "no such mask: $MASK"
[[ -x "$FREESURFER_HOME/bin/run_samseg" ]] || die "run_samseg not found under $FREESURFER_HOME"

if [ $SKIP -eq 1 ] && [ -f "$OUT/label_def.csv" ] && [ -f "$OUT/mri/aparc.atlas+aseg.nii.gz" ]; then
	echo "$OUT: already prepared, skipping"; exit 0
fi

mkdir -p "$OUT" || die "cannot create $OUT"

# 1. crop to the brain-mask bounding box, exactly as dl+direct.sh does, so the
#    prep sits on the same grid family as a DeepSCAN prep of the same scan.
CROPPED=$OUT/T1w_norm_noskull_cropped.nii.gz
$PY $PKG/crop.py "$MASK" "$T1" "$CROPPED" || die "crop failed"

# 2. SAMSEG on the cropped volume. --save-posteriors with no argument saves
#    every structure; we only read the cortex and white matter ones.
env FREESURFER_HOME=$FREESURFER_HOME \
    PATH=$FREESURFER_HOME/bin:$PATH \
    SUBJECTS_DIR=${SUBJECTS_DIR:-$OUT} \
	$FREESURFER_HOME/bin/run_samseg \
		-i "$CROPPED" -o "$OUT/samseg" --threads "$THREADS" $LESION \
		--save-posteriors \
	|| die "run_samseg failed (see $OUT/samseg)"

[[ -f "$OUT/samseg/seg.mgz" ]] || die "SAMSEG produced no seg.mgz in $OUT/samseg"

# 3. translate into the DeepSCAN id scheme and write the probabilities, then
#    build mri/ exactly as a DeepSCAN prep does.
$PY -m dldirect.prep_from_seg \
	--seg "$OUT/samseg/seg.mgz" --t1 "$CROPPED" --out-dir "$OUT" \
	--posterior-dir "$OUT/samseg/posteriors" $BINARY \
	--run-preparedata \
	|| die "prep_from_seg failed"

echo "SAMSEG prep ready: $OUT"
echo "  next: python -m dldirect.pial_pipeline --prep-dir $OUT --out-dir $OUT/field_pial"
