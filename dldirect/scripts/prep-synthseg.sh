#!/bin/bash
#
# prep-synthseg.sh -- build a --space cropped prep from SynthSeg instead of DeepSCAN.
#
# SynthSeg's --post writes a 4D posterior volume, so like the SAMSEG path this
# prep carries real probabilities and nothing has to invent logits. See
# dldirect/prep_from_seg.py for the aseg -> DeepSCAN id mapping.
#
# ONE THING TO WATCH: SynthSeg's stock label set has NO corpus-callosum labels.
# preparedata.py splits CC voxels into left/right white matter, and with no CC
# present there is nothing to split -- SynthSeg already lateralises the white
# matter as 2/41. That needs preparedata.py to tolerate a missing
# Corpus-Callosum row rather than raising on an empty argwhere.
#
# WHICH SYNTHSEG. FreeSurfer 7.4.1 ships every 2.0 weight
# ($FREESURFER_HOME/models/synthseg_{2.0,robust_2.0,parc_2.0,qc_2.0}.h5) and
# $FREESURFER_HOME/python/scripts/mri_synthseg is the same code with the same
# CLI, resolving its model directory from FREESURFER_HOME. That is the default
# here. The standalone checkout at $SS_DIR has only synthseg_1.0.h5, so --repo
# implies --v1; nothing needs downloading either way.
#
# The segmenter runs on the ALREADY CROPPED T1 so everything shares one grid.
# SynthSeg internally resamples to 1mm and resamples its output back, so the
# returned labels are on the grid it was given; prep_from_seg checks that.
#
# Usage:
#   prep-synthseg.sh [options] T1_STRIPPED MASK OUT_DIR
#
# Options:
#   --v1              use SynthSeg 1.0 instead of the default 2.0.
#   --repo            run the standalone SynthSeg checkout at $SS_DIR instead of
#                     FreeSurfer's copy. That checkout only has
#                     models/synthseg_1.0.h5, so it implies --v1.
#   --parc            run cortical parcellation and keep the 1000-series ids
#   --robust          SynthSeg's robust (slower) predictions
#   --fast            skip some postprocessing
#   --cpu             force CPU
#   --threads N       cores for the CPU path (default 8)
#   --binary          ignore the posteriors; use the hard mask as the probability
#   --skip-existing   do nothing if the prep already has label_def.csv

set -u
PKG=$(dirname $(dirname $(readlink -f "$0")))   # .../dldirect
REPO=$(dirname "$PKG")                          # repo root, for -m dldirect.*
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PY=${PY:-/home/student/miniconda3/envs/DL_DiReCT/bin/python}
FREESURFER_HOME=${FREESURFER_HOME:-/data/disk2/freesurfer}
SS_PY=${SS_PY:-/home/student/miniconda3/envs/synthseg/bin/python}
SS_DIR=${SS_DIR:-/data/disk2/projects/Synthseg_res/SynthSeg}
THREADS=8
PARC=""
V1=""
REPO_SS=0
EXTRA=""
BINARY=""
SKIP=0

die() { echo "ERROR: $1" >&2; exit 1; }

while [[ $# -gt 3 ]]; do
	case "$1" in
		--v1)            V1="--v1" ;;
		--repo)          REPO_SS=1; V1="--v1" ;;
		--parc)          PARC="--parc" ;;
		--robust)        EXTRA="$EXTRA --robust" ;;
		--fast)          EXTRA="$EXTRA --fast" ;;
		--cpu)           EXTRA="$EXTRA --cpu" ;;
		--threads)       THREADS=$2; shift ;;
		--binary)        BINARY="--binary" ;;
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
if [ $REPO_SS -eq 1 ]; then
	[[ -x "$SS_PY" ]] || die "no SynthSeg python at $SS_PY (set SS_PY)"
	[[ -f "$SS_DIR/scripts/commands/SynthSeg_predict.py" ]] \
		|| die "no SynthSeg_predict.py under $SS_DIR (set SS_DIR)"
	MODEL_DIR="$SS_DIR/data/labels_classes_priors"
else
	[[ -x "$FREESURFER_HOME/bin/mri_synthseg" ]] \
		|| die "no mri_synthseg under $FREESURFER_HOME (use --repo, or set FREESURFER_HOME)"
	MODEL_DIR="$FREESURFER_HOME/models"
fi

if [ $SKIP -eq 1 ] && [ -f "$OUT/label_def.csv" ] && [ -f "$OUT/mri/aparc.atlas+aseg.nii.gz" ]; then
	echo "$OUT: already prepared, skipping"; exit 0
fi

mkdir -p "$OUT/synthseg" || die "cannot create $OUT"

# 1. crop to the brain-mask bounding box (same grid discipline as everywhere else)
CROPPED=$OUT/T1w_norm_noskull_cropped.nii.gz
$PY $PKG/crop.py "$MASK" "$T1" "$CROPPED" || die "crop failed"

# 2. SynthSeg on the cropped volume, with posteriors
SEG=$OUT/synthseg/seg.nii.gz
POST=$OUT/synthseg/post.nii.gz
if [ $REPO_SS -eq 1 ]; then
	PYTHONPATH="$SS_DIR:${PYTHONPATH:-}" \
		$SS_PY "$SS_DIR/scripts/commands/SynthSeg_predict.py" \
			--i "$CROPPED" --o "$SEG" --post "$POST" \
			--threads "$THREADS" $V1 $PARC $EXTRA \
		|| die "SynthSeg_predict failed"
else
	env FREESURFER_HOME=$FREESURFER_HOME PATH=$FREESURFER_HOME/bin:$PATH \
		$FREESURFER_HOME/bin/mri_synthseg \
			--i "$CROPPED" --o "$SEG" --post "$POST" \
			--threads "$THREADS" $V1 $PARC $EXTRA \
		|| die "mri_synthseg failed"
fi

[[ -f "$SEG" ]] || die "SynthSeg produced no segmentation at $SEG"

# 3. the channel order of --post is SynthSeg's own label list, which
#    prep_from_seg needs in order to pick the cortex and white matter channels.
#    It is derived here from the package's label file rather than hardcoded, so
#    a --parc run (which has a longer list) works too.
LABELS=$OUT/synthseg/post_labels.txt
$PY - "$MODEL_DIR" "$V1" "$POST" "$LABELS" <<'PY' || die "could not resolve the posterior label order"
import os, sys, numpy as np, nibabel as nib
model_dir, v1, post, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# 1.0 and 2.0 have DIFFERENT label lists, and the channel order of --post is
# np.unique() of whichever one the model was built from, BACKGROUND INCLUDED.
name = 'synthseg_segmentation_labels.npy' if v1 else 'synthseg_segmentation_labels_2.0.npy'
p = os.path.join(model_dir, name)
if not os.path.exists(p):
    print('label file not found: %s' % p, file=sys.stderr); sys.exit(1)
lab = np.unique(np.load(p))
n_chan = nib.load(post).shape[-1]
if len(lab) != n_chan:
    print('%s gives %d unique labels but %s has %d channels'
          % (name, len(lab), os.path.basename(post), n_chan), file=sys.stderr)
    sys.exit(1)
np.savetxt(out, lab, fmt='%d')
print('posterior labels: %s -> %d channels, matches the volume' % (name, len(lab)))
PY

# 4. translate, write probabilities, build mri/
$PY -m dldirect.prep_from_seg \
	--seg "$SEG" --t1 "$CROPPED" --out-dir "$OUT" \
	--posteriors "$POST" --posterior-labels "$LABELS" \
	${PARC:+--parc-labels} $BINARY \
	--run-preparedata \
	|| die "prep_from_seg failed"

echo "SynthSeg prep ready: $OUT"
echo "  next: python -m dldirect.pial_pipeline --prep-dir $OUT --out-dir $OUT/field_pial"
