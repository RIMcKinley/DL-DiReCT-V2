#!/usr/bin/env python
"""Build a --space cropped prep from a FOREIGN segmentation (SAMSEG, SynthSeg).

WHAT A PREP HAS TO CONTAIN

`pial_pipeline.reconstruct` and `preparedata.py` between them read:

  T1w_norm_noskull_cropped.nii.gz   the cropped, skull-stripped T1 (LIA)
  softmax_seg.nii.gz                integer labels in DeepSCAN's id scheme
  label_def.csv                     ID,LABEL for every id in that volume
  a grey and a white matter PROBABILITY                    (see next section)

The DeepSCAN id scheme is NOT stock aseg. It merges the lateral and inferior
lateral ventricles into one `*-Ventricle-all` id (101/112), merges cerebellar
white and grey into `*-Cerebellum` (102/113), and carries a single
`Corpus-Callosum` (125) where aseg has the five CC_* labels. LABEL_MAP below is
that translation, written out so the two segmenters' native label sets land in
the ids `wm_labels.ribbon_labels` and `preparedata.py` expect.

Cortex is written under the names `lh-cortex` / `rh-cortex` (ids 3 and 42), NOT
`Left-Cerebral-Cortex`. The name matters and is not cosmetic:
`wm_labels.hemisphere_labels` builds the white matter fill from EVERY label
whose name starts `Left-`/`Right-`, so a cortex label named that way is filled
into the white surface's interior. DeepSCAN avoids this by naming cortex as
`lh-*`/`rh-*` DKT parcels, which only `ribbon_labels`' second branch picks up.
Naming cortex `Left-Cerebral-Cortex` here was measured to swallow the entire
ribbon: the surface-derived segmentation came out WM 783395 / GM 13590 voxels
against DeepSCAN's 412296 / 408574, 67%% of vertices had no cortex to move into,
and the pial travelled 0.07 mm. `surface_frames.cortex_labels` still returns
(3, 42) because it keys on the ids, which are unchanged.
Pass --parc-labels if a parcellated source should keep its 1000-series ids.

PROBABILITIES, NOT LOGITS

DeepSCAN preps carry per-label LOGITS in seg_<Label>.nii.gz, which
`load_gm_wm_probability` turns into probabilities with expit. A hard
segmentation has no logits to write, and faking them round-trips badly: the
loader reads `np.where(logit == 0, 0, expit(logit))`, so a structure with
probability exactly 0.5 has logit 0 and would be read as probability 0.

Both segmenters can emit posteriors directly (`run_samseg --save-posteriors`,
`SynthSeg_predict --post`), so this writes those as `prob_gm.nii.gz` and
`prob_wm.nii.gz` and the logit step is bypassed entirely. Without posteriors
(--binary) the probabilities are the hard mask itself, 1.0 inside and 0.0
outside -- usable, but the surface then lands on a voxel staircase rather than
at partial volume, since under the default `segmentation='surface-pv'` these
probabilities are what the GM and WM isosurfaces are extracted from.

GM is the max over the cortex structures and WM the max over the white matter
structures, matching what `load_gm_wm_probability` does to the logits.
"""

import argparse
import csv
import os
import sys

import numpy as np
import nibabel as nib


# --- the DeepSCAN id scheme, as read off a v0_f1 prep's label_def.csv --------
DEEPSCAN_IDS = {
    'Left-Cerebral-White-Matter': 2, 'lh-cortex': 3,
    'Left-Ventricle-all': 101, 'Left-Cerebellum': 102,
    'Left-Thalamus-Proper': 10, 'Left-Caudate': 11, 'Left-Putamen': 12,
    'Left-Pallidum': 13, 'Left-Hippocampus': 17, 'Left-Amygdala': 18,
    'Left-Accumbens-area': 26, 'Left-VentralDC': 28,
    'Right-Cerebral-White-Matter': 41, 'rh-cortex': 42,
    'Right-Ventricle-all': 112, 'Right-Cerebellum': 113,
    'Right-Thalamus-Proper': 49, 'Right-Caudate': 50, 'Right-Putamen': 51,
    'Right-Pallidum': 52, 'Right-Hippocampus': 53, 'Right-Amygdala': 54,
    'Right-Accumbens-area': 58, 'Right-VentralDC': 60,
    'Brain-Stem': 16, '3rd-Ventricle': 14, '4th-Ventricle': 15,
    'Corpus-Callosum': 125,
}

# --- aseg id -> DeepSCAN label name ----------------------------------------
# SAMSEG and SynthSeg both label in stock aseg ids, so one table serves both.
# Anything absent here is mapped to background and counted; --strict turns that
# into an error rather than a silent drop.
LABEL_MAP = {
    2: 'Left-Cerebral-White-Matter', 3: 'lh-cortex',
    4: 'Left-Ventricle-all', 5: 'Left-Ventricle-all',
    31: 'Left-Ventricle-all',                      # choroid plexus, intraventricular
    7: 'Left-Cerebellum', 8: 'Left-Cerebellum',
    10: 'Left-Thalamus-Proper', 11: 'Left-Caudate', 12: 'Left-Putamen',
    13: 'Left-Pallidum', 17: 'Left-Hippocampus', 18: 'Left-Amygdala',
    26: 'Left-Accumbens-area', 28: 'Left-VentralDC',
    41: 'Right-Cerebral-White-Matter', 42: 'rh-cortex',
    43: 'Right-Ventricle-all', 44: 'Right-Ventricle-all',
    63: 'Right-Ventricle-all',
    46: 'Right-Cerebellum', 47: 'Right-Cerebellum',
    49: 'Right-Thalamus-Proper', 50: 'Right-Caudate', 51: 'Right-Putamen',
    52: 'Right-Pallidum', 53: 'Right-Hippocampus', 54: 'Right-Amygdala',
    58: 'Right-Accumbens-area', 60: 'Right-VentralDC',
    16: 'Brain-Stem', 14: '3rd-Ventricle', 15: '4th-Ventricle',
    # aseg's five corpus-callosum labels collapse to DeepSCAN's single id.
    251: 'Corpus-Callosum', 252: 'Corpus-Callosum', 253: 'Corpus-Callosum',
    254: 'Corpus-Callosum', 255: 'Corpus-Callosum',
    # Deliberately dropped to background: 24 CSF (DeepSCAN has no CSF class and
    # the pial stops against seg==0), 30/62 vessel, 72 5th ventricle,
    # 77/80 (non-)WM-hypointensities -- v0_f1 has no hypointensity class either,
    # and preparedata only folds it into the midline mask when it is present.
}

# aseg 77, WM-hypointensities. Handled outside LABEL_MAP because it is not
# lateralised at source and DeepSCAN does not keep it as a class of its own:
# `load_gm_wm_probability` folds its logit into the white matter, and
# preparedata.py relabels its voxels to left/right cerebral white matter along
# with the corpus callosum. Dropping it to background instead would leave a
# hole in the white matter where a hypointense lesion is -- 470 voxels on
# OAS30001_ses-d0129_run-01 -- so it is lateralised here (see map_labels) and
# its posterior joins the white matter probability below.
HYPO_ID = 77

# (label_def name, source aseg ids, required, posterior file basename). The
# `required` False entry means a segmenter without that class is fine and it is
# simply left out of the max -- SynthSeg's stock label set has no 77, SAMSEG's
# does. The
# last field is the SEGMENTER's own structure name, which SAMSEG uses for
# posteriors/<name>.mgz -- it is not the label_def name, because cortex has to
# be renamed for the white matter fill (see the module docstring).
GM_TISSUE = (('lh-cortex', (3,), True, 'Left-Cerebral-Cortex'),
             ('rh-cortex', (42,), True, 'Right-Cerebral-Cortex'))
WM_TISSUE = (('Left-Cerebral-White-Matter', (2,), True, 'Left-Cerebral-White-Matter'),
             ('Right-Cerebral-White-Matter', (41,), True, 'Right-Cerebral-White-Matter'),
             ('WM-hypointensities', (HYPO_ID,), False, 'WM-hypointensities'))

GM_STRUCTURES = tuple(n for n, _, _, _ in GM_TISSUE)
WM_STRUCTURES = tuple(n for n, _, r, _ in WM_TISSUE if r)


def map_labels_passthrough(src, label_def_path, verbose=True):
    """Keep the source ids as they are, for a segmenter that ALREADY emits the
    DeepSCAN scheme.

    irrepunet-seg does: its `seg_label_map.FS_NAMES` is DL+DiReCT's own 99-label
    table, custom ids included (101/112 Ventricle-all, 102/113 Cerebellum, 125
    Corpus-Callosum) with the cortex as lh-*/rh-* Desikan-Killiany parcels.
    Pushing that through LABEL_MAP would be a lossy round trip -- it would
    collapse 68 parcels onto two cortex labels -- so the volume is passed through
    untouched and `label_def_path` (an existing DeepSCAN prep's label_def.csv) is
    copied verbatim as the name table.

    Returns (labels, rows, unknown) where `rows` is the (id, name) table to write
    and `unknown` is {id: n_voxels} for ids the table does not name.
    """
    import csv as _csv
    with open(label_def_path) as fh:
        rows = [(int(r['ID']), r['LABEL']) for r in _csv.DictReader(fh)]
    known = {i for i, _ in rows}
    out = np.asarray(src).astype(np.int32)
    unknown = {}
    for val in np.unique(out):
        val = int(val)
        if val and val not in known:
            unknown[val] = int((out == val).sum())
    if unknown and verbose:
        print('ids not named by %s: %s' % (os.path.basename(label_def_path),
              ', '.join('%d (%d vox)' % kv for kv in sorted(unknown.items()))))
    if verbose:
        present = sorted({int(v) for v in np.unique(out)} & known)
        print('pass-through scheme: %d labels present of %d in the table'
              % (len(present), len(rows)))
    return out, rows, unknown


def tissue_masks_from_names(labels, rows):
    """(gm_mask, wm_mask) from a DeepSCAN-scheme label volume and its name table.

    Cortex is whatever is named lh-*/rh-* (the DK parcels) plus the aggregate
    Left/Right-Cerebral-Cortex ids, so this works whether the source
    parcellates or not. White matter takes the corpus callosum and any
    hypointensities with it: preparedata.py relabels CC to left/right cerebral
    white matter, and load_gm_wm_probability folds hypointensities into WM, so
    counting them here keeps the two routes consistent.
    """
    gm_names = lambda n: n.startswith(('lh-', 'rh-')) or n in (
        'Left-Cerebral-Cortex', 'Right-Cerebral-Cortex')
    wm_names = ('Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter',
                'Corpus-Callosum', 'WM-hypointensities')
    gm_ids = [i for i, n in rows if gm_names(n)]
    wm_ids = [i for i, n in rows if n in wm_names]
    return (np.isin(labels, gm_ids).astype(np.float32),
            np.isin(labels, wm_ids).astype(np.float32))


def map_labels(src, strict=False, parc_labels=False, verbose=True):
    """Translate an aseg-convention volume into DeepSCAN ids.

    Returns (labels, used_names, dropped) where `dropped` is {src_id: n_voxels}
    for ids with no entry in LABEL_MAP.
    """
    out = np.zeros_like(src, dtype=np.int32)
    used, dropped = set(), {}
    hypo = src == HYPO_ID
    for val in np.unique(src):
        val = int(val)
        if val == 0 or val == HYPO_ID:
            continue                      # 77 is lateralised after the loop
        m = src == val
        if parc_labels and 1000 <= val < 3000:
            # a parcellated source (SynthSeg --parc): keep the 1000-series id,
            # which ribbon_labels reads through its lh-/rh- branch
            out[m] = val
            used.add(('lh-' if val < 2000 else 'rh-') + 'parc%d' % val)
            continue
        name = LABEL_MAP.get(val)
        if name is None:
            dropped[val] = int(m.sum())
            continue
        out[m] = DEEPSCAN_IDS[name]
        used.add(name)
    # WM-hypointensities carries no side, so give each voxel the label of the
    # nearest cerebral white matter. preparedata.py does the same job for a
    # DeepSCAN prep with a mid-sagittal split off the corpus callosum; a nearest
    # -label assignment needs no CC (neither SAMSEG nor SynthSeg emits one) and
    # no assumption about which array axis is left-right.
    n_hypo = int(hypo.sum())
    if n_hypo:
        from scipy import ndimage
        wm_ids = (DEEPSCAN_IDS['Left-Cerebral-White-Matter'],
                  DEEPSCAN_IDS['Right-Cerebral-White-Matter'])
        wm = np.isin(out, wm_ids)
        if wm.any():
            _, idx = ndimage.distance_transform_edt(~wm, return_indices=True)
            out[hypo] = out[tuple(i[hypo] for i in idx)]
            used.update(('Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter'))
            if verbose:
                n_l = int((out[hypo] == wm_ids[0]).sum())
                print('WM-hypointensities (aseg 77): %d voxels -> nearest cerebral '
                      'white matter (%d left, %d right)' % (n_hypo, n_l, n_hypo - n_l))
        else:
            dropped[HYPO_ID] = n_hypo
            if verbose:
                print('WM-hypointensities present but no cerebral white matter to '
                      'attach them to; dropped %d voxels' % n_hypo)
    if dropped:
        msg = ', '.join('%d (%d vox)' % (k, v) for k, v in sorted(dropped.items()))
        if strict:
            raise ValueError('unmapped source labels: %s' % msg)
        if verbose:
            print('dropped to background (no DeepSCAN equivalent): %s' % msg)
    return out, used, dropped


def write_label_def(path, used_names, parc_ids=()):
    """label_def.csv covering exactly what softmax_seg.nii.gz contains.

    `preparedata.py` indexes this by name for Corpus-Callosum and the two
    cerebral white matter labels, and `wm_labels` walks every row, so a row for
    a label that is not in the volume is harmless but a missing row is fatal.
    """
    rows = [(DEEPSCAN_IDS[n], n) for n in sorted(used_names) if n in DEEPSCAN_IDS]
    rows += [(int(i), n) for i, n in parc_ids]
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['ID', 'LABEL'])
        for i, n in rows:
            w.writerow([i, n])
    return rows


def _load(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img


def tissue_probabilities(labels, posterior_dir=None, posteriors_4d=None,
                         posterior_labels=None, verbose=True):
    """(gm_prob, wm_prob) in [0, 1], from posteriors when available.

    posterior_dir     SAMSEG's `posteriors/` -- one <Structure>.mgz per label
    posteriors_4d     SynthSeg's --post volume, plus `posterior_labels`, the
                      aseg id of each channel in it
    neither           fall back to the hard mask (1.0 inside, 0.0 outside)

    The max over each tissue's structures mirrors what load_gm_wm_probability
    does with the logits, so the two prep families feed the solver the same
    shape of input.
    """
    def _from_dir(tissue):
        acc = None
        for n, _ids, required, fname in tissue:
            for ext in ('.mgz', '.nii.gz', '.nii'):
                p = os.path.join(posterior_dir, fname + ext)
                if os.path.exists(p):
                    d = _load(p)[0]
                    acc = d if acc is None else np.maximum(acc, d)
                    break
            else:
                if not required:
                    continue          # this segmenter has no such class; fine
                if verbose:
                    print('no posterior for %s; falling back to its hard mask' % n)
                m = (labels == DEEPSCAN_IDS[n]).astype(np.float32)
                acc = m if acc is None else np.maximum(acc, m)
        return acc

    def _from_4d(tissue):
        acc = None
        n_chan = posteriors_4d.shape[-1]
        if len(posterior_labels) != n_chan:
            # predict_synthseg writes one channel per np.unique(labels), BACKGROUND
            # INCLUDED. Getting this list wrong shifts every index and either
            # overruns the array or silently reads the wrong structure, which is
            # worse. Fail here with both counts rather than downstream.
            raise ValueError(
                'posterior label list has %d entries but the posterior volume has '
                '%d channels. The list must be np.unique() of the segmenter\'s '
                'label file, background included -- check that the label file '
                'matches the model version that produced these posteriors '
                '(SynthSeg 2.0 uses synthseg_segmentation_labels_2.0.npy).'
                % (len(posterior_labels), n_chan))
        for n, ids, required, _fname in tissue:
            chans = [i for i, sid in enumerate(posterior_labels) if int(sid) in ids]
            if not chans:
                if not required:
                    continue
                if verbose:
                    print('no posterior channel for %s; using its hard mask' % n)
                d = (labels == DEEPSCAN_IDS[n]).astype(np.float32)
            else:
                d = posteriors_4d[..., chans].max(axis=-1)
            acc = d if acc is None else np.maximum(acc, d)
        return acc

    if posterior_dir:
        gm, wm = _from_dir(GM_TISSUE), _from_dir(WM_TISSUE)
    elif posteriors_4d is not None:
        gm, wm = _from_4d(GM_TISSUE), _from_4d(WM_TISSUE)
    else:
        gm = np.isin(labels, [DEEPSCAN_IDS[n] for n in GM_STRUCTURES]).astype(np.float32)
        wm = np.isin(labels, [DEEPSCAN_IDS[n] for n in WM_STRUCTURES]).astype(np.float32)
    return np.clip(gm, 0, 1).astype(np.float32), np.clip(wm, 0, 1).astype(np.float32)


def build_prep(seg_path, t1_path, out_dir, posterior_dir=None, posteriors_path=None,
               posterior_labels=None, strict=False, parc_labels=False,
               label_def=None, verbose=True):
    """Write softmax_seg / label_def / prob_gm / prob_wm and the cropped T1.

    The segmentation is assumed to be ON THE SAME GRID as `t1_path`; both
    segmenters write their output on the grid they were given, so passing the
    already-cropped T1 to the segmenter keeps everything on one grid and no
    resampling happens anywhere in here.
    """
    seg_img = nib.load(seg_path)
    src = np.asarray(seg_img.dataobj)
    t1_img = nib.load(t1_path)
    if seg_img.shape[:3] != t1_img.shape[:3]:
        raise ValueError('segmentation %s and T1 %s are on different grids; run the '
                         'segmenter on the cropped T1 so nothing needs resampling'
                         % (seg_img.shape[:3], t1_img.shape[:3]))

    os.makedirs(out_dir, exist_ok=True)
    for sub in ('mri', 'label', 'surf'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    if label_def:
        # the source already speaks DeepSCAN's scheme: pass it through
        labels, rows, _unknown = map_labels_passthrough(src, label_def, verbose=verbose)
        used = None
    else:
        labels, used, _ = map_labels(src, strict=strict, parc_labels=parc_labels,
                                     verbose=verbose)
    if not np.any(labels == DEEPSCAN_IDS['Corpus-Callosum']) and verbose:
        print('NOTE: the source has no corpus-callosum label. preparedata.py splits '
              'CC voxels into left/right white matter; with none present there is '
              'nothing to split, and the source already lateralises the white '
              'matter. Requires the missing-CC tolerance in preparedata.py.')

    aff = seg_img.affine
    nib.save(nib.Nifti1Image(labels, aff),
             os.path.join(out_dir, 'softmax_seg.nii.gz'))

    if label_def:
        with open(os.path.join(out_dir, 'label_def.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['ID', 'LABEL'])
            for i, n in rows:
                w.writerow([i, n])
    else:
        parc_ids = ()
        if parc_labels:
            u = [int(v) for v in np.unique(labels) if 1000 <= int(v) < 3000]
            parc_ids = [(v, ('lh-' if v < 2000 else 'rh-') + 'parc%d' % v) for v in u]
        write_label_def(os.path.join(out_dir, 'label_def.csv'), used, parc_ids)

    posteriors_4d = None
    if posteriors_path:
        posteriors_4d = np.asarray(nib.load(posteriors_path).dataobj, dtype=np.float32)
    if label_def:
        # no posteriors from this source; the masks come from the names
        gm, wm = tissue_masks_from_names(labels, rows)
    else:
        gm, wm = tissue_probabilities(labels, posterior_dir, posteriors_4d,
                                      posterior_labels, verbose=verbose)
    nib.save(nib.Nifti1Image(gm, aff), os.path.join(out_dir, 'prob_gm.nii.gz'))
    nib.save(nib.Nifti1Image(wm, aff), os.path.join(out_dir, 'prob_wm.nii.gz'))

    dst_t1 = os.path.join(out_dir, 'T1w_norm_noskull_cropped.nii.gz')
    if os.path.realpath(t1_path) != os.path.realpath(dst_t1):
        nib.save(nib.Nifti1Image(np.asarray(t1_img.dataobj), t1_img.affine), dst_t1)

    if verbose:
        src_kind = ('posteriors' if (posterior_dir or posteriors_path) and not label_def
                    else 'hard mask')
        n_lab = len(rows) if label_def else len(used)
        print('%s: %d labels, GM %.0f vox (p>0.5), WM %.0f vox (p>0.5), from %s'
              % (out_dir, n_lab, (gm > 0.5).sum(), (wm > 0.5).sum(), src_kind))
    return dict(labels=labels, gm=gm, wm=wm, affine=aff, used=used)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seg', required=True, help='label volume in aseg convention')
    p.add_argument('--t1', required=True,
                   help='the cropped skull-stripped T1 the segmenter was run on')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--posterior-dir',
                   help="SAMSEG's posteriors/ directory (one volume per structure)")
    p.add_argument('--posteriors',
                   help="SynthSeg's --post 4D volume")
    p.add_argument('--posterior-labels',
                   help='npy/txt of the aseg id of each --posteriors channel')
    p.add_argument('--binary', action='store_true',
                   help='ignore posteriors; use the hard mask as the probability')
    p.add_argument('--parc-labels', action='store_true',
                   help='keep a parcellated source\'s 1000-series cortex ids')
    p.add_argument('--label-def',
                   help='pass the source ids through UNCHANGED and copy this '
                        "label_def.csv as the name table. For a segmenter that "
                        "already emits DeepSCAN's scheme (irrepunet-seg does); "
                        'point it at an existing DeepSCAN prep\'s label_def.csv')
    p.add_argument('--strict', action='store_true',
                   help='fail on a source label with no DeepSCAN equivalent')
    p.add_argument('--run-preparedata', action='store_true',
                   help='also run preparedata.py --space cropped on the result')
    args = p.parse_args()

    plabels = None
    if args.posterior_labels:
        plabels = (np.load(args.posterior_labels) if args.posterior_labels.endswith('.npy')
                   else np.loadtxt(args.posterior_labels))
    build_prep(args.seg, args.t1, args.out_dir,
               posterior_dir=None if args.binary else args.posterior_dir,
               posteriors_path=None if args.binary else args.posteriors,
               posterior_labels=plabels, strict=args.strict,
               parc_labels=args.parc_labels, label_def=args.label_def)

    if args.run_preparedata:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run([sys.executable, os.path.join(here, 'preparedata.py'),
                            '-inputpath', args.out_dir, '--space', 'cropped'])
        return r.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
