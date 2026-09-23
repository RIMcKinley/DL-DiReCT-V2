#!/bin/bash
#
# fsr-and-field-pial -- run BOTH pial reconstructions from ONE white surface,
# with consistent geometry, so the results can be compared and displayed
# together without any resampling.
#
# The two pipelines need different grids:
#
#   fast_surface_reconstruction  mri_normalize / mri_edit_wm_with_aseg /
#                                mri_pretess / mris_make_surfaces all require
#                                the LIA 256^3 conform.
#   field_pial_prototype         works on the cropped grid the segmentation,
#                                DiReCT and the velocity field already live on,
#                                and refuses a surface built on another grid.
#
# So this builds the white ONCE, in conformed space, lets FreeSurfer deform it
# into its pial, then maps that same mesh into the cropped tkrRAS frame with an
# exact affine (retarget_surface: no interpolation, vertex order preserved) and
# propagates it along the velocity field. Both pials therefore share vertices
# one-for-one, and both are written with volume_info describing the cropped
# grid, so freeview places every surface on the cropped MRI correctly.
#
# Usage:
#   fsr-and-field-pial.sh --prep CROPPED_PREP --out OUT_DIR [pial options...]
#
#   --prep   a --space cropped prep (seg_<Label>.nii.gz, softmax_seg.nii.gz,
#            label_def.csv, T1w_norm_noskull_cropped.nii.gz)
#   --out    output root; creates conformed/, surf/, field_pial/ and view/
#   anything after those is passed through to field_pial_prototype, e.g.
#            --gate-mode sigmoid --gate-sigmoid-center-deg 90 ...
#
# Requires FREESURFER_HOME (7.4.1 tested) for the FSR half.
#
set -e
usage() { sed -n '2,/^set -e/p' "$0" | sed 's/^# \?//'; exit 1; }

PREP=""; OUT=""; PIAL_OPTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep) PREP="$2"; shift 2 ;;
    --out)  OUT="$2";  shift 2 ;;
    -h|--help) usage ;;
    *) PIAL_OPTS+=("$1"); shift ;;
  esac
done
[[ -n "$PREP" && -n "$OUT" ]] || usage
[[ -n "$FREESURFER_HOME" ]] || { echo "ERROR: FREESURFER_HOME is not set" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${DLDIRECT_PYTHON:-python}"
SUBJ="$(basename "$PREP")"
CONF="$OUT/conformed"

mkdir -p "$OUT" "$CONF/mri" "$CONF/surf" "$OUT/surf" "$OUT/field_pial" "$OUT/view"

# ---------------------------------------------------------------- conformed
# A second, conformed prep of the same segmentation. Separate directory: the
# cropped prep must keep its own mri/ and surf/, and preparedata.py overwrites
# both.
echo "== conformed prep (for FreeSurfer) =="
cp "$PREP"/*.nii.gz "$PREP"/label_def.csv "$CONF/" 2>/dev/null || true
[[ -d "$PREP/label" ]] && cp -r "$PREP/label" "$CONF/"
"$PY" "$REPO/dldirect/preparedata.py" -inputpath "$CONF"

echo "== FreeSurfer volume steps =="
mri_normalize -aseg "$CONF/mri/aseg.presurf.mgz" "$CONF/mri/norm.mgz" "$CONF/mri/brain.mgz"
cp "$CONF/mri/brain.mgz" "$CONF/mri/norm.mgz"
cp "$CONF/mri/brain.mgz" "$CONF/mri/nu.mgz"
cp "$CONF/mri/brain.mgz" "$CONF/mri/brain.finalsurfs.mgz"
mri_edit_wm_with_aseg "$CONF/mri/wm.seg.mgz" "$CONF/mri/brain.mgz" \
    "$CONF/mri/aseg.presurf.mgz" "$CONF/mri/wm.asegedit.mgz"
mri_pretess "$CONF/mri/wm.asegedit.mgz" wm "$CONF/mri/brain.mgz" "$CONF/mri/wm.mgz"

echo "== white surface (shared by both pipelines) =="
"$PY" "$REPO/dldirect/dl_wm_surface_parallel_dev.py" -inputpath "$CONF" -outputpath "$CONF" -ns 50

echo "== FreeSurfer pial (mris_make_surfaces) =="
# Sequential rather than GNU parallel: two at once is memory-hungry, and
# parallel is not always installed.
export SUBJECTS_DIR="$OUT"
for h in lh rh; do
  mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc \
      -aseg aseg.presurf -nowhite -mgz -T1 brain.finalsurfs conformed "$h"
done
"$PY" "$REPO/dldirect/smooth_pial.py" -filepath "$CONF"
for h in lh rh; do
  mv "$CONF/surf/$h.pial" "$CONF/surf/$h.pial.raw"
  mv "$CONF/surf/$h.thickness" "$CONF/surf/$h.thickness.raw"
done
"$PY" "$REPO/dldirect/extract_morphometrics.py" -filepath "$CONF" -subj "$SUBJ"

# ------------------------------------------------------------------ cropped
echo "== map the shared white into the cropped frame =="
"$PY" -m dldirect.retarget_surface --src-prep "$CONF" --dst-prep "$PREP" \
    --out-dir "$OUT/surf" --surf white

echo "== our pial, propagated from that same white =="
"$PY" -m dldirect.field_pial_prototype --prep-dir "$PREP" --surf-dir "$OUT/surf" \
    --out-dir "$OUT/field_pial" --skip-naive "${PIAL_OPTS[@]}"

# --------------------------------------------------------------------- view
# Everything in one directory, one frame, one volume_info, ready for freeview.
echo "== assembling view/ =="
for s in pial.raw pial white.smoothed; do
  "$PY" -m dldirect.retarget_surface --src-prep "$CONF" --dst-prep "$PREP" \
      --out-dir "$OUT/view" --surf "$s" --rename "fsr.$s" 2>/dev/null || \
      echo "  (skipped $s: not produced)"
done
for h in lh rh; do
  cp "$OUT/surf/$h.white" "$OUT/view/$h.white"
  cp "$OUT/field_pial/$h.pial.field_best" "$OUT/view/$h.pial.ours"
done

cat > "$OUT/view/freeview.sh" <<VIEW
#!/bin/bash
# Open both reconstructions on the cropped MRI. Every surface here is in the
# cropped tkrRAS frame and carries matching volume_info, so no -trans is needed.
V="\$(cd "\$(dirname "\$0")" && pwd)"
exec freeview -v "$PREP/T1w_norm_noskull_cropped.nii.gz" \\
  -f "\$V/lh.white:edgecolor=yellow" "\$V/rh.white:edgecolor=yellow" \\
     "\$V/lh.pial.ours:edgecolor=red" "\$V/rh.pial.ours:edgecolor=red" \\
     "\$V/lh.fsr.pial.raw:edgecolor=blue" "\$V/rh.fsr.pial.raw:edgecolor=blue" \\
     -viewport coronal
VIEW
chmod +x "$OUT/view/freeview.sh"

echo
echo "done."
echo "  shared white : $OUT/surf/{lh,rh}.white          (cropped frame)"
echo "  our pial     : $OUT/field_pial/{lh,rh}.pial.field_best"
echo "  FSR pial     : $CONF/surf/{lh,rh}.pial.raw      (conformed frame)"
echo "  display set  : $OUT/view/   -- run $OUT/view/freeview.sh"
