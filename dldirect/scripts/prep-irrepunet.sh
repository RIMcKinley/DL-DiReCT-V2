#!/bin/bash
#
# prep-irrepunet.sh -- build a --space cropped prep from irrepunet-seg.
#
# WHY THIS ONE IS THE SIMPLEST OF THE THREE. irrepunet-seg was trained on
# DL+DiReCT's own label set: its seg_label_map.FS_NAMES is the same 99-label
# table, custom ids included (101/112 Ventricle-all, 102/113 Cerebellum, 125
# Corpus-Callosum), with the cortex as lh-*/rh-* Desikan-Killiany parcels. So
# there is NOTHING to translate -- the volume is passed through unchanged and an
# existing DeepSCAN prep's label_def.csv is copied as the name table
# (--label-def). Forcing it through prep_from_seg's aseg LABEL_MAP would be
# lossy: it would collapse 68 parcels onto two cortex labels.
#
# It also means the cortex is already named lh-*/rh-*, so the white matter fill
# excludes it for the same reason DeepSCAN's does -- see the prep_from_seg
# docstring for what goes wrong when it is not.
#
# NO POSTERIORS. irrepunet-seg emits hard labels only (--keep-scheme-idx just
# swaps the id convention), so prob_gm/prob_wm are the hard masks: 1.0 inside,
# 0.0 outside. Measured elsewhere in this repo, that costs very little -- both
# isosurfaces are meshed from binary masks anyway, and PV vs binary moved the
# pial by a 0.0016 mm median under --segmentation logits.
#
# GM is every lh-*/rh-* parcel (plus the aggregate 3/42 if present); WM is
# Left/Right-Cerebral-White-Matter plus Corpus-Callosum, since preparedata.py
# relabels CC into left/right white matter a step later.
#
# Usage:
#   prep-irrepunet.sh [options] T1_STRIPPED MASK OUT_DIR
#
# Options:
#   --label-def PATH  label_def.csv to copy (default: the DeepSCAN prep named by
#                     $REF_PREP, else fail -- it is not guessed)
#   --gpu N           GPU index passed to irrepunet-seg (default 0)
#   --largest-cc      drop label islands (irrepunet-seg --largest-cc)
#   --skip-existing   do nothing if the prep already has label_def.csv

set -u
PKG=$(dirname $(dirname $(readlink -f "$0")))   # .../dldirect
REPO=$(dirname "$PKG")                          # repo root, for -m dldirect.*
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PY=${PY:-/home/student/miniconda3/envs/DL_DiReCT/bin/python}
IRREP=${IRREP:-/home/student/miniconda3/envs/irrepunet/bin/irrepunet-seg}
REF_PREP=${REF_PREP:-/data/disk2/oasis60/OAS30001_ses-d0129_run-01}
LABEL_DEF=""
GPU=0
LARGEST=""
SKIP=0

die() { echo "ERROR: $1" >&2; exit 1; }

while [[ $# -gt 3 ]]; do
	case "$1" in
		--label-def)     LABEL_DEF=$2; shift ;;
		--gpu)           GPU=$2; shift ;;
		--largest-cc)    LARGEST="--largest-cc" ;;
		--skip-existing) SKIP=1 ;;
		*)               die "unknown option $1" ;;
	esac
	shift
done
[[ $# -eq 3 ]] || { sed -n '3,40p' "$0"; exit 1; }

T1=$1
MASK=$2
OUT=$3
[[ -n "$LABEL_DEF" ]] || LABEL_DEF=$REF_PREP/label_def.csv

[[ -f "$T1" ]]        || die "no such T1: $T1"
[[ -f "$MASK" ]]      || die "no such mask: $MASK"
[[ -x "$IRREP" ]]     || die "irrepunet-seg not found at $IRREP (set IRREP)"
[[ -f "$LABEL_DEF" ]] || die "no label_def.csv at $LABEL_DEF (set --label-def or REF_PREP)"

if [ $SKIP -eq 1 ] && [ -f "$OUT/label_def.csv" ] && [ -f "$OUT/mri/aparc.atlas+aseg.nii.gz" ]; then
	echo "$OUT: already prepared, skipping"; exit 0
fi

mkdir -p "$OUT/irrepunet" || die "cannot create $OUT"

# 1. crop to the brain-mask bounding box, same grid discipline as the other preps
CROPPED=$OUT/T1w_norm_noskull_cropped.nii.gz
$PY $PKG/crop.py "$MASK" "$T1" "$CROPPED" || die "crop failed"

# 2. segment the CROPPED volume, so the output shares that grid and nothing is
#    resampled between the segmentation and the surfaces. --skull-stripped skips
#    the brain-localization pass, which the input has already had applied.
SEG=$OUT/irrepunet/seg.nii.gz
$IRREP --input "$CROPPED" --output "$SEG" --gpu "$GPU" --skull-stripped $LARGEST \
	|| die "irrepunet-seg failed"
[[ -f "$SEG" ]] || die "irrepunet-seg produced no segmentation at $SEG"

# 3. pass the ids through, write the masks, build mri/
$PY -m dldirect.prep_from_seg \
	--seg "$SEG" --t1 "$CROPPED" --out-dir "$OUT" \
	--label-def "$LABEL_DEF" --run-preparedata \
	|| die "prep_from_seg failed"

echo "irrepunet prep ready: $OUT"
echo "  next: python -m dldirect.pial_pipeline --prep-dir $OUT --out-dir $OUT/field_pial"
