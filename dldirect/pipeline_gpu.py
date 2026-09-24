"""Segmentation, surfaces, field, pial and parcellation in ONE process.

The shipped chain is six processes driven by a shell script, each handing off
through disk. That is defensible for restartability and it is what the
published pipeline does, but it costs real time in the one place it does not
have to: the model's per-class output is written as 94 volumes (389 MB) and
read back to build the tissue probabilities and the parcel posteriors.
Measured on this hardware, ~21 s to write and ~33 s to read, against ~16 s of
actual computation -- and ~13 hours over an 840-case batch.

Here the model runs once and its output is used directly:

    segment()                       model -> logits in memory
    gm_wm_probability_from_logits   -> the tissue probabilities (verified
                                       identical to the on-disk collapse)
    CorticalPosterior.from_logits   -> 84 MB of parcel posteriors, the rest
                                       of the 941 MB dropped immediately
    reconstruct(...)                -> white, field, pial, and with
                                       parcellate the stamped ribbon

WHAT THIS GIVES UP. One process means one failure domain: a fault in the
surface stage discards the segmentation too, which the shell chain survives.
--save-segmentation writes the segmentation outputs anyway (and
--save-posteriors the 67 MB posterior archive), so a rerun can skip the model.
Use them when iterating; skip them in a batch.
"""

import argparse
import os
import sys
import time

import numpy as np
import nibabel as nib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dldirect.parcel_posterior import CorticalPosterior
from dldirect.field_pial_prototype import gm_wm_probability_from_logits


def run(t1_file, out_dir, subject_id=None, model='v0_f1', hemis=('lh', 'rh'),
        parcellate=True, repair_intersections=False, repair_max_move=1.0,
        save_segmentation=False, save_posteriors=False, propagate_on='cuda',
        verbose=True, **reconstruct_kw):
    """The whole chain for one case. Returns reconstruct's result dict."""
    from dldirect.DeepSCAN_Anatomy_Newnet_apply import segment, write_segmentation
    from dldirect import pial_pipeline

    os.makedirs(out_dir, exist_ok=True)
    subject_id = subject_id or os.path.basename(os.path.normpath(out_dir))

    t0 = time.time()
    seg = segment(t1_file, model=model, verbose=verbose)
    t_seg = time.time() - t0
    if verbose:
        print('segmentation: %.1f s, %d classes' % (t_seg, len(seg['names'])))

    post = None
    if parcellate:
        t0 = time.time()
        post = CorticalPosterior.from_logits(seg['logit'], seg['names'])
        if verbose:
            print('posteriors: %.1f s, %d voxels, %.0f MB held (%.0f MB if the '
                  'full volume were kept)'
                  % (time.time() - t0, len(post), post.nbytes / 1e6,
                     len(seg['names']) * np.prod(seg['logit'].shape[1:]) * 4 / 1e6))
        if save_posteriors:
            post.save(os.path.join(out_dir, 'parcel_posterior.npz'))

    # The TISSUE volumes still go to disk -- the white-surface builder reads
    # the prep directly, and they are what the stock chain writes anyway (9
    # classes, ~43 MB). What this pipeline saves is the other 85: the parcel
    # posteriors stay in memory as `post`, so the 346 MB of parcel volumes are
    # never written and never read back.
    write_segmentation(seg, out_dir, subject_id,
                       save_logits=list(seg['names']) if save_segmentation else None)

    # preparedata.py builds mri/aparc.atlas+aseg.nii.gz, which the
    # white-surface builder reads. It is still a subprocess: it is a
    # module-level script like the segmentation was, and refactoring it is a
    # separate job. It costs no model time -- it reads softmax_seg and the T1,
    # both of which are already here -- so the 346 MB of parcel volumes this
    # pipeline avoids are avoided either way.
    for sub in ('mri', 'label', 'surf'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)   # preparedata writes into these
    t1_local = os.path.join(out_dir, 'T1w_norm_noskull_cropped.nii.gz')
    if not os.path.exists(t1_local):
        os.symlink(os.path.abspath(t1_file), t1_local)
    import subprocess
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(_HERE, 'preparedata.py'),
                    '-inputpath', out_dir, '--space', 'cropped'],
                   check=True,
                   stdout=None if verbose else subprocess.DEVNULL)
    if verbose:
        print('preparedata: %.1f s' % (time.time() - t0))

    gm_prob, wm_prob, ref_img = gm_wm_probability_from_logits(
        seg['logit'], seg['names'], seg['affine'])
    seg['logit'] = None                      # 941 MB released; `post` is the keeper

    t0 = time.time()
    res = pial_pipeline.reconstruct(
        out_dir, out_dir=out_dir, hemis=tuple(hemis), propagate_on=propagate_on,
        parcellate=parcellate, repair_intersections=repair_intersections,
        repair_max_move=repair_max_move, verbose=verbose,
        subject_id=subject_id, tissue=(gm_prob, wm_prob, ref_img),
        parcel_posterior=post, **reconstruct_kw)
    if verbose:
        print('surfaces + field + pial: %.1f s' % (time.time() - t0))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHAT THIS')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('t1', help='skull-stripped, cropped T1 (as crop.py writes)')
    ap.add_argument('out_dir')
    ap.add_argument('--subject', default=None)
    ap.add_argument('--model', default='v0_f1')
    ap.add_argument('--hemi', nargs='+', default=['lh', 'rh'])
    ap.add_argument('--no-parcellate', action='store_true')
    ap.add_argument('--repair-intersections', action='store_true')
    ap.add_argument('--repair-max-move', type=float, default=1.0)
    ap.add_argument('--save-segmentation', action='store_true',
                    help='also write the usual seg_*.nii.gz etc, so a rerun can '
                         'skip the model')
    ap.add_argument('--save-posteriors', action='store_true',
                    help='write parcel_posterior.npz (67 MB) for the same reason')
    ap.add_argument('--propagate-on', default='cuda', choices=['cuda', 'cpu'])
    args = ap.parse_args(argv)
    run(args.t1, args.out_dir, subject_id=args.subject, model=args.model,
        hemis=tuple(args.hemi), parcellate=not args.no_parcellate,
        repair_intersections=args.repair_intersections,
        repair_max_move=args.repair_max_move,
        save_segmentation=args.save_segmentation,
        save_posteriors=args.save_posteriors, propagate_on=args.propagate_on)


if __name__ == '__main__':
    main()
