#!/usr/bin/env python
"""Per-parcel cortical thickness from the FIELD and the SURFACES, in DL+DiReCT's
CSV format, without reading a thickness map.

FOUR DEFINITIONS, ONE FORMAT

  field     voxel-based. Every WM voxel of the solve's ACTIVE ZONE is carried
            along the velocity field exactly as the propagation carries a
            vertex, and the thickness is how far it got.
  travel    per-vertex |pial_i - white_i|, the straight-line chord.
  nn        per-vertex nearest-neighbour distance from the white vertex to the
            pial mesh.
  sym_nn    per-vertex 0.5 * (w->p + p->w), the symmetric nearest-neighbour
            distance. White and pial share a tessellation here, so vertex i of
            one corresponds to vertex i of the other and the two directions can
            be averaged per vertex rather than only per region.

Each is written as result-thick-<name>.csv and result-thickstd-<name>.csv with
DL+DiReCT's exact header: SUBJECT, the 68 Desikan-Killiany parcels, then
lh-MeanThickness and rh-MeanThickness. Means and standard deviations are taken
over NON-ZERO values inside each parcel, as extract_stats.py does, and an empty
parcel reports NaN rather than 0.

WHY THE FIELD IS INTEGRATED, NOT JUST SAMPLED

The saved field is a VELOCITY, not a displacement. Reading |v| at a WM voxel
gives 0.134 mm median on the development subject -- times INTEGRATION_POINTS
that is 1.34 mm, against a surface travel of 2.79 mm on the same scan. The gap
is real and not a scaling constant: a point starting on the WM boundary moves
into grey matter, where the field is larger (|v| median 0.469 there), so the
integral along its trajectory exceeds the field at its start. The propagation
therefore steps `pos -= sample(field, pos) * STEP_SCALE`, ROUNDS times, and this
module does exactly the same from voxel centres. What it does NOT do is the mesh
relaxation and the medial-wall pin: those are operations on a surface and have
no meaning for an unconnected voxel.

`voxel_field_raw_mm` is reported alongside for reference -- |v| at the voxel
times INTEGRATION_POINTS -- because it is the quantity "the field value at this
voxel" most directly names, and seeing both makes the difference explicit.

THE ACTIVE ZONE, AND WHICH VOXELS ARE SAMPLED

`direct_cuda` defines active_mask = GM voxels + the WM contour, the WM voxels
adjacent to GM (`extract_wm_contours`), and the velocity is zero outside it.
This samples the WM half of that zone, which is the inner boundary of the
ribbon. Note that is NOT the same set DL+DiReCT averages over: extract_stats.py
takes the INNER GM BOUNDARY, `find_boundaries(seg == 2, mode='inner')`, which is
the GM voxels adjacent to non-GM, one voxel further out. The two rims are
adjacent but disjoint, and the repo's own note records the GM rim reading
0.24-0.41 mm thinner than the WM side. Numbers from here are therefore not
interchangeable with a DL+DiReCT result-thick.csv even though the format is.

The GM rim is deliberately NOT offered here, and cannot be: DL+DiReCT averages
the THICKNESS MAP over it, and that map's values at GM voxels come from the
solve's hit counting. Integrating the field from GM-rim voxels instead would
not reproduce it -- a trajectory starting inside grey matter covers only the
part of the ribbon beyond its starting point, not the whole thickness. Adding
that column means reading a thickness map, which is what this module exists to
avoid.

A KNOWN DIFFERENCE WE DO NOT CORRECT: THE LIMEN INSULAE

About 8.6% of insula vertices (510 of 5949 on OAS30001; 212 lh, 298 rh) are
counted here and called unknown by FreeSurfer. They are not noise and not a
defect on our side -- they form a compact bilateral band immediately ANTERIOR TO
THE AMYGDALA, at the limen insulae, where insular cortex becomes continuous with
the temporal pole. Our white surface runs forward through that junction;
FreeSurfer's turns and follows the amygdala boundary, leaving the strip outside
its surface. Everywhere else in those slices the two contours are superimposed.

Every structural explanation was tested and ruled out:

  pinning              they survive the no_push/pin exclusion above
  off-tissue labelling they survive the cortex-or-WM rule above
  the sqrt(3) rule     sweeping max_dist 1.0 -> 5.0mm moves the labelled
                       fraction by 0.3 points and the travel mean by 0.010mm
  topology correction  gpu 510 flagged vs nighres 513, and lh identical at 212
                       -- swapping the whole correction changes nothing
  hippocampus          well posterior and separate; it is the amygdala that abuts

TWO FIXES WERE MEASURED AND REJECTED:

  --exclude-amygdala, which drops it from the WM fill, degrades the surface:
  flipped faces 0.1047% -> 0.1317% (lh) and 0.1232% -> 0.1426% (rh) for no gain
  elsewhere. The repo's note records 99.1% of the amygdala lying inside the
  white surface by default, so the fill does swallow it -- but removing it costs
  more than it buys.

  An amygdala-adjacency exclusion separates well on distance (flagged median
  3.00mm from the amygdala against 47.78mm for agreeing vertices) but trades
  badly, because the amygdala sits deep to genuine insular and temporal-pole
  cortex and a ball around it sweeps up real cortex at the same rate:

      within 2mm   248 flagged (13.9%)  for   109 legitimate  ratio 2.3
      within 3mm   349        (19.6%)   for   358             ratio 1.0
      within 5mm   500        (28.1%)   for   755             ratio 0.7

  against 14-25:1 for the two rules that ARE applied above. Adding hippocampus
  makes it worse still. So this is left in, as a documented difference in where
  the two pipelines put the cortical boundary, not as a filter.

PARCEL ASSIGNMENT follows extract_stats.py exactly: nearest parcellation voxel
by cKDTree, and anything further than sqrt(3) from one is dropped (that is what
stops a hippocampal boundary voxel from being credited to a cortical parcel).
Surface vertices are labelled by the same rule, at the WHITE vertex's voxel
position, and the pial vertex inherits its white vertex's parcel so that the two
nearest-neighbour directions are aggregated over the same region.
"""

import argparse
import csv as _csv
import os
import sys

import numpy as np
import nibabel as nib
from scipy import spatial
from scipy.ndimage import binary_dilation, map_coordinates

from . import pial_clean as pc

_HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ('field', 'travel', 'nn', 'sym_nn')


# ---------------------------------------------------------------------------
# labels, in DL+DiReCT's order and naming
# ---------------------------------------------------------------------------
def get_labels(offset=0, lut_path=None):
    """(lut, parcel names, all column names). extract_stats.get_labels, but with
    the lookup table located relative to this file rather than sys.argv[0] --
    that version only works when run as a script."""
    import pandas as pd
    lut = pd.read_csv(lut_path or os.path.join(_HERE, 'fs_lut.csv'))
    lut = {l.Key: l.Label for _, l in lut.iterrows()
           if l.Label >= offset and l.Label <= 3000 + offset}
    names = [k for k in lut
             if (k.startswith('lh') or k.startswith('rh'))
             and 'nknown' not in k and 'corpuscallosum' not in k
             and 'Medial_wall' not in k]
    return lut, names, names + ['lh-MeanThickness', 'rh-MeanThickness']


def aggregate(values, parcel_ids, offset=0, lut_path=None):
    """(mean, std, column names) per parcel over NON-ZERO values.

    `values` and `parcel_ids` are flat and aligned: one entry per sampled voxel
    or vertex. Empty parcels give NaN, matching extract_stats.py.
    """
    lut, names, all_names = get_labels(offset, lut_path)
    values = np.asarray(values, float)
    parcel_ids = np.asarray(parcel_ids)

    groups = [values[parcel_ids == lut[n]] for n in names]
    groups += [values[(parcel_ids >= 1000 + offset) & (parcel_ids < 2000 + offset)],
               values[(parcel_ids >= 2000 + offset) & (parcel_ids < 3000 + offset)]]

    def _stat(g, fn):
        g = g[g.nonzero()]
        return fn(g) if g.size else np.nan

    return (np.array([_stat(g, np.mean) for g in groups]),
            np.array([_stat(g, np.std) for g in groups]), all_names)


def write_stats(row, subject_id, path, names):
    with open(path, 'w', newline='') as fh:
        w = _csv.writer(fh, delimiter=',')
        w.writerow(['SUBJECT'] + names)
        w.writerow([subject_id] + list(row))


# ---------------------------------------------------------------------------
# parcel assignment (extract_stats.py's rule)
# ---------------------------------------------------------------------------
def nearest_parcel(coords, parcellation, max_dist=np.sqrt(3.0)):
    """Parcel id for each voxel coordinate, 0 where none is within `max_dist`."""
    parc_coords = np.array(np.where(parcellation > 1000)).T
    if not len(parc_coords):
        raise ValueError('no cortical parcels (>1000) in the parcellation; a '
                         'source without a parcellation cannot be aggregated '
                         'per region')
    dist, idx = spatial.cKDTree(parc_coords).query(np.asarray(coords), k=1)
    near = parc_coords[idx]
    out = parcellation[near[:, 0], near[:, 1], near[:, 2]].astype(np.int32)
    out[dist > max_dist] = 0
    return out


# ---------------------------------------------------------------------------
# the field metric
# ---------------------------------------------------------------------------
def active_wm_voxels(seg):
    """WM voxels of the active zone: WM adjacent to GM (extract_wm_contours)."""
    gm = np.asarray(seg) == 2
    wm = np.asarray(seg) == 3
    return wm & binary_dilation(gm, np.ones((3, 3, 3), bool))


def _sample(field, pos):
    """Trilinear sample of a [D,H,W,3] field at voxel positions [N,3].

    map_coordinates(order=1, mode='nearest') is trilinear with the edge value
    held outside -- the same thing the GPU path's grid_sample(bilinear, border)
    does, as pial_pipeline's module docstring records.
    """
    c = pos.T
    return np.stack([map_coordinates(field[..., k], c, order=1, mode='nearest')
                     for k in range(3)], axis=1)


def field_thickness(seg, velocity, zooms, rounds=None, step_scale=None):
    """(thickness_mm, coords, raw_mm) for the active zone's WM voxels.

    The integration mirrors propagate_pial_torch: `pos -= sample(field, pos) *
    step_scale`, `rounds` times, in voxel coordinates. No relaxation and no pin
    -- both are mesh operations. Distances are converted to mm with `zooms`, so
    an anisotropic grid is handled correctly.
    """
    rounds = pc.ROUNDS if rounds is None else rounds
    step_scale = pc.STEP_SCALE if step_scale is None else step_scale
    mask = active_wm_voxels(seg)
    start = np.array(np.where(mask)).T.astype(np.float64)
    pos = start.copy()
    for _ in range(rounds):
        pos = pos - _sample(velocity, pos) * step_scale
    z = np.asarray(zooms[:3], float)
    thick = np.linalg.norm((pos - start) * z, axis=1)
    raw = np.linalg.norm(_sample(velocity, start) * z, axis=1) * pc.INTEGRATION_POINTS
    return thick, start.astype(int), raw


# ---------------------------------------------------------------------------
# the surface metrics
# ---------------------------------------------------------------------------
def surface_thickness(white, pial):
    """(travel, nn, sym_nn) per vertex, all in mm.

    Requires the vertex correspondence the propagation preserves: pial vertex i
    came from white vertex i.
    """
    white = np.asarray(white, float)
    pial = np.asarray(pial, float)
    if white.shape != pial.shape:
        raise ValueError('white %s and pial %s are not vertex-corresponded'
                         % (white.shape, pial.shape))
    travel = np.linalg.norm(pial - white, axis=1)
    w2p = spatial.cKDTree(pial).query(white)[0]
    p2w = spatial.cKDTree(white).query(pial)[0]
    return travel, w2p, 0.5 * (w2p + p2w)


# ---------------------------------------------------------------------------
def compute(prep_dir, surf_dir, subject_id, out_dir=None, hemis=('lh', 'rh'),
            velocity_name='pial_Velocity.nii.gz', write_white=False, verbose=True):
    """All four metrics, aggregated per parcel. Returns {metric: (mean, std, names)}."""
    from . import surface_seg
    from .field_pial_prototype import (build_no_push_mask, build_pin_mask,
                                       load_gm_wm_probability)

    out_dir = out_dir or surf_dir
    os.makedirs(out_dir, exist_ok=True)

    # the segmentation the solve used, rebuilt deterministically from the prep
    if verbose:
        print('rebuilding the solve segmentation...')
    sd = surface_seg.build_surface_segmentation(prep_dir, hemis=tuple(hemis),
                                                verbose=False)
    seg, tovox, ref_img = sd['seg'], sd['tovox'], sd['ref_img']
    zooms = ref_img.header.get_zooms()[:3]

    parc = np.asarray(nib.load(os.path.join(prep_dir, 'mri',
                                            'aparc.atlas+aseg.nii.gz')).dataobj)
    offset = 0 if parc.max() <= 3000 else 10000

    results = {}

    # --- field ---
    vel_path = os.path.join(surf_dir, velocity_name)
    if os.path.exists(vel_path):
        vel = np.asarray(nib.load(vel_path).dataobj, dtype=np.float32)
        thick, coords, raw = field_thickness(seg, vel, zooms)
        ids = nearest_parcel(coords, parc)
        keep = ids > 0
        if verbose:
            print('field: %d active-zone WM voxels, %d labelled (%.1f%%); '
                  'integrated median %.3f mm, raw |v|*n median %.3f mm'
                  % (len(thick), keep.sum(), 100 * keep.mean(),
                     np.median(thick[keep]), np.median(raw[keep])))
        results['field'] = aggregate(thick[keep], ids[keep], offset)
        results['field_raw'] = aggregate(raw[keep], ids[keep], offset)
    elif verbose:
        print('no %s in %s; skipping the field metric' % (velocity_name, surf_dir))

    # --- surfaces ---
    # the pin mask needs the same two files pial_pipeline's pin does
    import pandas as pd
    soft_p = os.path.join(prep_dir, 'softmax_seg.nii.gz')
    ldef_p = os.path.join(prep_dir, 'label_def.csv')
    soft_seg = np.asarray(nib.load(soft_p).dataobj) if os.path.exists(soft_p) else None
    id_map = ({r.LABEL: int(r.ID) for _, r in pd.read_csv(ldef_p).iterrows()}
              if os.path.exists(ldef_p) else None)
    if soft_seg is None or id_map is None:
        raise FileNotFoundError('softmax_seg.nii.gz and label_def.csv are needed to '
                                'exclude the pinned vertices; %s has neither' % prep_dir)

    vals = {m: [] for m in ('travel', 'nn', 'sym_nn')}
    ids_all = []
    n_drop_named = 0
    n_off_tissue = 0
    for h in hemis:
        wp = os.path.join(surf_dir, '%s.white' % h)
        pp = os.path.join(surf_dir, '%s.pial' % h)
        if not os.path.exists(pp):
            if verbose:
                print('missing %s.pial in %s; skipping the surface metrics'
                      % (h, surf_dir))
            vals = None
            break
        wfaces = None
        if os.path.exists(wp):
            w, wfaces = nib.freesurfer.io.read_geometry(wp)
        else:
            # `surface-pv` keeps the white surface in memory and writes only the
            # pial, so a batch run has no ?h.white on disk. sd['surfaces'] IS the
            # mesh the propagation started from -- same build, same parameters --
            # so use it rather than forcing a second pass over every subject.
            w = np.asarray(sd['surfaces'][h][0], float)
            if write_white:
                from .surface_frames import volume_info_from_image
                nib.freesurfer.io.write_geometry(
                    wp, w, sd['surfaces'][h][1], create_stamp=None,
                    volume_info=volume_info_from_image(ref_img, prep_dir))
        p = nib.freesurfer.io.read_geometry(pp)[0]
        tr, nn, sym = surface_thickness(w, p)
        # label at the WHITE vertex, by the same nearest-parcel rule
        vox = np.rint(tovox(np.asarray(w, float))).astype(int)
        for k in range(3):
            vox[:, k] = np.clip(vox[:, k], 0, parc.shape[k] - 1)
        ids = nearest_parcel(vox, parc)

        # A PINNED VERTEX HAS NO CORTICAL THICKNESS. The medial wall, the
        # subcortical structures the hemisphere fill swallows, anything with no
        # cortex to move into -- the propagation holds those in place, so their
        # travel is the feather blend's residue rather than a ribbon. They are
        # near enough to a parcel for nearest_parcel to name them, which is the
        # trap: they are not zero, so `aggregate`'s nonzero filter does not drop
        # them, and they drag the parcel mean down. `evaluate_surface` excludes
        # them via no_push=pin_mask; this now does the same.
        f = np.asarray(sd['surfaces'][h][1]) if wfaces is None else wfaces
        no_push = build_no_push_mask(w, f, seg, soft_seg, id_map, tovox, rings=0)
        pin = build_pin_mask(no_push, w, f, seg, soft_seg, id_map, tovox,
                             scope='medial-wall', rings=0)
        drop = no_push | pin

        # A VERTEX SITTING IN NOTHING IS NOT CORTEX EITHER. nearest_parcel names
        # a vertex from the nearest parcel voxel within sqrt(3); it never asks
        # what the vertex itself sits in. A vertex over background, ventricle,
        # hippocampus or amygdala is exactly as close to a cortical voxel as a
        # legitimate one, so distance cannot separate them -- sweeping max_dist
        # from 1.0 to 5.0mm moves the labelled fraction by 0.3 points and the
        # travel mean by 0.010mm. The label it sits in does separate them.
        #
        # Cortex OR white matter, not cortex alone: this is the WHITE surface,
        # so its vertices legitimately sit on the WM side of the boundary.
        # Requiring a cortical parcel would discard 53% of all counted vertices.
        # Measured on OAS30001, against the vertices FreeSurfer calls unknown:
        #
        #   rule                        flagged dropped   legitimate dropped
        #   own voxel > 1000 (cortex)     1364 (60.2%)     114315 (53.28%)  <- no
        #   own voxel != 0                 380 (16.8%)         15 ( 0.01%)
        #   own voxel in cortex or WM      483 (21.3%)         33 ( 0.02%)  <- this
        #   travel > 1.0mm                 886 (39.1%)        267 ( 0.12%)
        #
        # Travel thresholds score well but are circular -- they exclude vertices
        # for being thin, which is the quantity being measured, and would bias
        # every parcel mean upward by construction. Not used.
        own = parc[vox[:, 0], vox[:, 1], vox[:, 2]]
        wm_ids = [i for n, i in ((r.LABEL, int(r.ID)) for _, r in
                                 pd.read_csv(ldef_p).iterrows())
                  if n in ('Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter',
                           'Corpus-Callosum', 'WM-hypointensities')]
        off_tissue = ~((own > 1000) | np.isin(own, wm_ids))
        n_off_tissue += int((off_tissue & ~drop & (ids > 0)).sum())
        drop = drop | off_tissue

        ids = np.where(drop, 0, ids)
        n_drop_named += int((drop & (nearest_parcel(vox, parc) > 0)).sum())
        ids_all.append(ids)
        vals['travel'].append(tr); vals['nn'].append(nn); vals['sym_nn'].append(sym)

    if vals is not None and ids_all:
        ids = np.concatenate(ids_all)
        keep = ids > 0
        if verbose:
            print('surfaces: %d vertices, %d labelled (%.1f%%); %d pinned vertices '
                  'dropped that nearest_parcel would have named (%d of them for '
                  'sitting outside cortex and white matter)'
                  % (len(ids), keep.sum(), 100 * keep.mean(), n_drop_named,
                     n_off_tissue))
        for m in ('travel', 'nn', 'sym_nn'):
            v = np.concatenate(vals[m])
            results[m] = aggregate(v[keep], ids[keep], offset)

    for m, (mean, std, names) in results.items():
        write_stats(mean, subject_id, os.path.join(out_dir, 'result-thick-%s.csv' % m), names)
        write_stats(std, subject_id, os.path.join(out_dir, 'result-thickstd-%s.csv' % m), names)
        if verbose:
            print('%-10s lh %.3f  rh %.3f mm  -> result-thick-%s.csv'
                  % (m, mean[-2], mean[-1], m))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir')
    p.add_argument('--surf-dir', required=True,
                   help='directory holding ?h.white, ?h.pial and pial_Velocity.nii.gz')
    p.add_argument('--subject')
    p.add_argument('--out-dir', help='default: --surf-dir')
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--write-white', action='store_true',
                   help='also write the rebuilt ?h.white next to the pial')
    p.add_argument('--preps', nargs='+',
                   help='batch: prep directories or globs. --surf-dir is then a '
                        'SUBDIRECTORY NAME inside each prep, and --subject is '
                        'ignored (each prep\'s directory name is used)')
    p.add_argument('--skip-existing', action='store_true',
                   help='batch: skip preps that already have result-thick-field.csv')
    args = p.parse_args()

    if not args.preps:
        compute(args.prep_dir, args.surf_dir, args.subject, args.out_dir,
                hemis=tuple(args.hemi), write_white=args.write_white)
        return 0

    import glob as _glob
    import time as _time
    preps = []
    for pat in args.preps:
        preps += sorted(_glob.glob(pat)) or [pat]
    ok = fail = skip = 0
    t0 = _time.time()
    for i, d in enumerate(preps, 1):
        name = os.path.basename(os.path.normpath(d))
        sdir = os.path.join(d, args.surf_dir)
        if args.skip_existing and os.path.exists(
                os.path.join(args.out_dir or sdir, 'result-thick-field.csv')):
            print('[%d/%d] %s: skipped' % (i, len(preps), name)); skip += 1
            continue
        t = _time.time()
        try:
            compute(d, sdir, name, args.out_dir, hemis=tuple(args.hemi),
                    write_white=args.write_white, verbose=False)
        except Exception as exc:
            print('[%d/%d] %s: FAILED %s: %s'
                  % (i, len(preps), name, type(exc).__name__, exc), file=sys.stderr)
            fail += 1
        else:
            print('[%d/%d] %s: %.0fs' % (i, len(preps), name, _time.time() - t),
                  flush=True)
            ok += 1
    print('%d ok, %d failed, %d skipped in %.0fs' % (ok, fail, skip, _time.time() - t0))
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
