"""Score the mesh CRF over a cohort, paired, one row per hemisphere per setting.

WHY THIS EXISTS. Every setting in mesh_crf -- the metaclass removal, the null
label, the sampling depths, the vote gate, theta and beta -- was chosen on a
single subject, where the differences between settings are smaller than the
spread between subjects (Dice 0.829 to 0.921 over 30 subjects, against at most
0.0005 between CRF settings). A single-case comparison cannot see past that,
so nothing here should be judged on a mean: judge on SIGN CONSISTENCY across
hemispheres, per the project's ablation convention.

WHAT IS MEASURED. Dice against FreeSurfer's aparc is the weak signal -- moving
a border a few vertices along a bank barely changes overlap. The stronger one
is whether the labelling's own borders sit in sulcal fundi, and it is measured
TWICE per row:

    gap_z       in hull_depth's z-score, which the CRF's edge weights are
                built from -- so a CRF driven hard enough raises this by
                construction, and at high beta it is not evidence.
    gap_sulc    in FreeSurfer's own ?h.sulc, sampled through the same
                nearest-vertex map as the annotation. The energy cannot pay
                for this one, so it is the honest check.

FreeSurfer's own aparc is carried in both units as the anchor.

The logits are not kept: the model writes 94 volumes (389 MB) per case, which
are read and deleted. Results append per case and completed cases are skipped,
so the run survives being killed.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import nibabel as nib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dldirect.mesh_crf import (column_unary, soft_unary, border_sulcality_prior,
                               icm_pairwise, dice_per_label, n_stray,
                               freesurfer_labels_on_our_mesh, boundary_sulcality,
                               fs_vertex_map, fs_morph_on_our_mesh)
from dldirect.hull_depth import (rasterize, hull_depth_field, sample,
                                 geodesic_zscore, crf_edge_weights)
from dldirect.field_pial_prototype import make_transforms, _mesh_adjacency
from dldirect import regional_stats as rs

fsio = nib.freesurfer.io

# (tag, theta, beta); theta None = the unary alone, no CRF
SETTINGS = [('unary', None, None), ('t0.5b1', 0.5, 1.0),
            ('t1b4', 1.0, 4.0), ('t2b8', 2.0, 8.0)]
FIELDS = ['case', 'hemi', 'setting', 'dice', 'null_frac', 'fs_unknown_frac',
          'null_dice', 'strays', 'gap_z', 'gap_sulc', 'fs_gap_z', 'fs_gap_sulc',
          'moved', 'nvert', 'nkeep']


def mean_dice(a, b, labels):
    return float(np.mean(list(dice_per_label(a, b, labels).values())))


def score_case(case, case_dir, fs_dir, logit_dir, run, radius, rows):
    lut, valid, _c = rs.get_labels()
    ref = nib.load(os.path.join(case_dir, 'mri', 'aparc.atlas+aseg.nii.gz'))
    tovox, _ = make_transforms(ref)
    parc = np.asarray(ref.dataobj).astype(np.int32)
    spacing = tuple(float(z) for z in ref.header.get_zooms()[:3])
    shape = tuple(ref.shape[:3])

    for h in ('lh', 'rh'):
        names = [k for k in valid if k.startswith('%s-' % h) and lut[k] > 1000]
        cortex = np.array([lut[k] for k in names])
        labels = np.concatenate([cortex, [0]])          # + null
        w, wf = fsio.read_geometry(os.path.join(case_dir, run, '%s.white' % h))
        p, pf = fsio.read_geometry(os.path.join(case_dir, run, '%s.pial' % h))
        wv = tovox(np.asarray(w, float))

        unary, _prob, _wt = column_unary(w, p, logit_dir, names, tovox)
        _m, Wm, deg = _mesh_adjacency(np.asarray(w, float), np.asarray(wf))
        mask = rasterize(tovox(np.asarray(p, float)), pf, shape)
        dep, _hull = hull_depth_field(mask, radius, spacing)
        z, _mu, _sd = geodesic_zscore(sample(dep, wv).astype(np.float64), w, wf,
                                      50, adjacency=(Wm, deg))
        zf = np.asarray(z, float)

        _v, votes = soft_unary(wv, parc, cortex)
        cort = votes.sum(1) > 0
        # one vertex correspondence for everything of FreeSurfer's
        j = fs_vertex_map(case_dir, fs_dir, h, np.asarray(w, float))
        ref_lab = freesurfer_labels_on_our_mesh(case_dir, fs_dir, h, w, labels,
                                                names + ['%s-NULL' % h], j=j)
        sulc = fs_morph_on_our_mesh(case_dir, fs_dir, h, w, 'sulc', j=j)
        keep = cort & (ref_lab > 0)
        base = labels[unary.argmin(1)]
        fs_unknown = ref_lab <= 0

        for tag, theta, beta in SETTINGS:
            if theta is None:
                out = base
                # an edge set is still needed to MEASURE the gap; it does not
                # touch the labels
                edges = crf_edge_weights(zf, w, wf, theta=1.0, floor=0.05)[0]
            else:
                edges, ew = crf_edge_weights(zf, w, wf, theta=theta, floor=0.05)
                G = border_sulcality_prior(base, edges, zf, labels, scale=0.6)
                out = labels[icm_pairwise(unary, edges, ew, G, beta=beta)[0]]
            null = out == 0
            rows.append(dict(
                case=case, hemi=h, setting=tag,
                dice=round(mean_dice(out[keep], ref_lab[keep], cortex), 5),
                null_frac=round(float(null.mean()), 5),
                fs_unknown_frac=round(float(fs_unknown.mean()), 5),
                null_dice=round(float(2 * (null & fs_unknown).sum()
                                      / max(null.sum() + fs_unknown.sum(), 1)), 4),
                strays=n_stray(out, Wm, labels),
                gap_z=round(boundary_sulcality(out, edges, zf, keep)[2], 4),
                gap_sulc=round(boundary_sulcality(out, edges, sulc, keep)[2], 4),
                fs_gap_z=round(boundary_sulcality(ref_lab, edges, zf, keep)[2], 4),
                fs_gap_sulc=round(boundary_sulcality(ref_lab, edges, sulc, keep)[2], 4),
                moved=round(float((out[keep] != base[keep]).mean()), 5),
                nvert=int(len(w)), nkeep=int(keep.sum())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY THIS')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subjects', help='file of case ids, one per line')
    ap.add_argument('--cases-dir', required=True)
    ap.add_argument('--fs-root', required=True, help='recon-all dirs, for scoring')
    ap.add_argument('--out', required=True, help='CSV; appended to, completed cases skipped')
    ap.add_argument('--run', default='field_pial_sigma0.65')
    ap.add_argument('--apply-script', required=True,
                    help='DeepSCAN apply with SAVE_LOGITS_FILTER = None')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--model', default='v0_f1')
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--radius', type=float, default=5.0, help='hull ball radius')
    ap.add_argument('--scratch', default=None, help='where logits go; deleted per case')
    args = ap.parse_args(argv)

    scratch = args.scratch or os.path.join(os.path.dirname(args.out), 'logit_tmp')
    cases = [l.strip() for l in open(args.subjects) if l.strip()]
    done = set()
    if os.path.exists(args.out):
        done = {r['case'] for r in csv.DictReader(open(args.out))}

    for i, case in enumerate(cases):
        if case in done:
            print('skip %s' % case, flush=True)
            continue
        logit_dir = os.path.join(scratch, case)
        t0, rows = time.time(), []
        try:
            os.makedirs(logit_dir, exist_ok=True)
            subprocess.run(
                [args.python, args.apply_script, '--model', args.model,
                 os.path.join(args.cases_dir, case, 'T1w_norm_noskull_cropped.nii.gz'),
                 logit_dir, case],
                check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            score_case(case, os.path.join(args.cases_dir, case),
                       os.path.join(args.fs_root, case), logit_dir, args.run,
                       args.radius, rows)
        except Exception as e:                       # one bad case must not end the run
            print('FAIL %-32s %s: %s' % (case, type(e).__name__, e), flush=True)
            shutil.rmtree(logit_dir, ignore_errors=True)
            continue
        shutil.rmtree(logit_dir, ignore_errors=True)
        new = not os.path.exists(args.out)
        with open(args.out, 'a', newline='') as fh:
            wri = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                wri.writeheader()
            wri.writerows(rows)
        print('[%2d/%d] %-32s %5.0fs  lh/rh unary Dice %.4f/%.4f'
              % (i + 1, len(cases), case, time.time() - t0,
                 rows[0]['dice'], rows[len(SETTINGS)]['dice']), flush=True)


if __name__ == '__main__':
    main()
