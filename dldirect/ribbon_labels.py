"""Carry surface labels from the WM/GM boundary out through the ribbon.

STATUS: research prototype. Nothing imports it; run it directly.

Given a parcel label per white-surface vertex, label every grey-matter voxel
by the nearest seed measured GEODESICALLY WITHIN THE RIBBON -- a multi-source
Voronoi partition on the voxel graph, not a Euclidean nearest-vertex lookup.

WHY NOT NEAREST VERTEX
----------------------
The two banks of a sulcus are ~2 mm apart in R^3 and tens of millimetres
apart along the cortex. A Euclidean assignment lets a label jump the gap, so
a voxel on one bank can take the parcel of the bank opposite -- which is
precisely the error a cortical parcellation must not make. Restricting the
propagation to steps between adjacent GM voxels makes that jump impossible:
to reach the far bank the front must travel down to the fundus and back up.

METHOD
------
    nodes  = grey-matter voxels
    edges  = 26-neighbours, weight = the physical step length in mm
    seeds  = voxels holding a white-surface vertex, carrying its label
    label(v) = label of the seed minimising geodesic distance to v

scipy's dijkstra(..., min_only=True) computes exactly this in one pass over
the whole seed set, returning for each node the source that reached it first,
so the cost is one Dijkstra rather than one per label.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import nibabel as nib
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ribbon_graph(gm, spacing):
    """(graph, node_index, coords) over GM voxels, 26-connected, mm weights."""
    idx = np.full(gm.shape, -1, np.int64)
    coords = np.array(np.nonzero(gm)).T
    idx[gm] = np.arange(len(coords))
    sp_ = np.asarray(spacing, float)
    rows, cols, wts = [], [], []
    offs = [(dz, dy, dx)
            for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)]
    # only half the offsets: the graph is symmetric and dijkstra is told so,
    # which halves both the build time and the memory
    offs = [o for o in offs if o > (0, 0, 0)]
    for o in offs:
        q = coords + np.asarray(o)
        ok = np.all((q >= 0) & (q < np.asarray(gm.shape)), axis=1)
        j = np.full(len(coords), -1, np.int64)
        j[ok] = idx[q[ok, 0], q[ok, 1], q[ok, 2]]
        good = j >= 0
        rows.append(np.nonzero(good)[0])
        cols.append(j[good])
        wts.append(np.full(good.sum(), np.linalg.norm(np.asarray(o) * sp_)))
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    wts = np.concatenate(wts)
    n = len(coords)
    G = sp.csr_matrix((wts, (rows, cols)), shape=(n, n))
    return G, idx, coords


def seed_voxels(vert_vox, vert_labels, gm, idx, max_snap=2):
    """{node: label} for the voxels holding white-surface vertices.

    A white vertex sits ON the WM/GM interface, so its own voxel is often
    white matter rather than grey. Such a seed is snapped to the nearest GM
    voxel within `max_snap`; one that cannot reach grey matter at all is
    dropped rather than forced somewhere arbitrary.

    Where several vertices land in one voxel the majority label wins, which
    matters at the medial wall where parcels meet densely.
    """
    base = np.rint(np.asarray(vert_vox)).astype(int)
    shape = np.asarray(gm.shape)
    base = np.clip(base, 0, shape - 1)
    votes = {}
    snapped = dropped = 0
    offs = [(0, 0, 0)] + [(dz, dy, dx)
                          for r in range(1, max_snap + 1)
                          for dz in range(-r, r + 1)
                          for dy in range(-r, r + 1)
                          for dx in range(-r, r + 1)
                          if max(abs(dz), abs(dy), abs(dx)) == r]
    for v, lab in zip(base, vert_labels):
        if lab <= 0:
            continue
        hit = None
        for k, o in enumerate(offs):
            q = np.clip(v + np.asarray(o), 0, shape - 1)
            if gm[q[0], q[1], q[2]]:
                hit = idx[q[0], q[1], q[2]]
                if k:
                    snapped += 1
                break
        if hit is None or hit < 0:
            dropped += 1
            continue
        votes.setdefault(int(hit), Counter())[int(lab)] += 1
    out = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    return out, snapped, dropped


def propagate(gm, spacing, vert_vox, vert_labels, max_snap=2, verbose=True):
    """Label every GM voxel by geodesic nearest seed. Returns (labels, dist)."""
    G, idx, coords = _ribbon_graph(gm, spacing)
    seeds, snapped, dropped = seed_voxels(vert_vox, vert_labels, gm, idx, max_snap)
    if not seeds:
        raise SystemExit('no seed landed in grey matter')
    nodes = np.fromiter(seeds.keys(), np.int64)
    slab = np.fromiter((seeds[k] for k in nodes), np.int64)
    if verbose:
        print('  ribbon %d voxels, %d seed voxels (%d vertices snapped to GM, '
              '%d dropped)' % (len(coords), len(nodes), snapped, dropped))
    dist, _pred, src = dijkstra(G, directed=False, indices=nodes,
                                min_only=True, return_predecessors=True)
    # src gives the SEED NODE that reached each voxel first
    lookup = np.full(G.shape[0], -1, np.int64)
    lookup[nodes] = slab
    lab_nodes = np.where(src >= 0, lookup[np.clip(src, 0, None)], -1)
    out = np.zeros(gm.shape, np.int32)
    out[coords[:, 0], coords[:, 1], coords[:, 2]] = np.where(lab_nodes > 0,
                                                             lab_nodes, 0)
    dvol = np.zeros(gm.shape, np.float32)
    good = np.isfinite(dist)
    dvol[coords[good, 0], coords[good, 1], coords[good, 2]] = dist[good]
    if verbose:
        unreached = int((lab_nodes <= 0).sum())
        print('  propagated: %d voxels unreached (%.2f%%), geodesic distance '
              'median %.2f mm p95 %.2f mm'
              % (unreached, 100.0 * unreached / len(coords),
                 float(np.median(dist[good])), float(np.percentile(dist[good], 95))))
    return out, dvol


def propagate_by_field(gm, velocity, vert_tkr, vert_labels, tovox, totkr,
                       rounds=None, step_scale=None, verbose=True):
    """Label each ribbon voxel by tracing the DiReCT field back to the white.

    The pipeline already rides this field the other way: propagate_pial carries
    a white vertex outward with `pos - v` for ROUNDS steps. Running the same
    integrator with `pos + v` carries a grey-matter voxel INWARD, landing it
    near the white-surface vertex whose column it belongs to -- the same
    correspondence the thickness is measured along, so the ribbon is
    partitioned exactly as the surfaces are.

    Preferred over a geodesic walk on the voxel graph. At a fundus the ribbon
    is one or two voxels thick and a 26-connected diagonal step can cut
    straight through it, so a graph path jumps the fundus almost as cheaply as
    a Euclidean one. The field cannot: its gated smoothing was built to refuse
    to average across opposing banks, so a trajectory stays on its own bank by
    construction.
    """
    from dldirect import pial_clean as pc
    from scipy.ndimage import map_coordinates
    from scipy import spatial
    rounds = pc.ROUNDS if rounds is None else rounds
    step_scale = pc.STEP_SCALE if step_scale is None else step_scale
    coords = np.array(np.nonzero(gm)).T
    cur = totkr(coords.astype(float))          # voxel centres, in tkrRAS
    for _ in range(rounds):
        pos = tovox(cur)
        v = np.stack([map_coordinates(velocity[..., k], pos.T, order=1,
                                      mode='nearest') for k in range(3)], axis=1)
        cur = cur + (totkr(pos + v) - totkr(pos)) * step_scale
    keep = np.asarray(vert_labels) > 1000
    d, j = spatial.cKDTree(np.asarray(vert_tkr)[keep]).query(cur)
    lab = np.asarray(vert_labels)[keep][j]
    out = np.zeros(gm.shape, np.int32)
    out[coords[:, 0], coords[:, 1], coords[:, 2]] = lab
    if verbose:
        moved = np.linalg.norm(cur - totkr(coords.astype(float)), axis=1)
        print('  field trace: %d rounds, voxel travel median %.2f mm p95 %.2f; '
              'landing %.2f mm from the nearest white vertex (median)'
              % (rounds, float(np.median(moved)), float(np.percentile(moved, 95)),
                 float(np.median(d))))
    return out


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY NOT')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case')
    ap.add_argument('--run', default='field_pial_sigma0.65')
    ap.add_argument('--annot', default=None,
                    help='?h.annot giving the per-vertex labels (default: take '
                         'them from the case parcellation by nearest voxel)')
    ap.add_argument('--fs-dir', default=None, help='recon-all dir, for scoring')
    ap.add_argument('--hemi', nargs='+', default=['lh', 'rh'])
    ap.add_argument('--logit-dir', default=None,
                    help="seed from the model's per-parcel logits at the white "
                         'vertices instead of resampling the existing '
                         'parcellation')
    ap.add_argument('--field', default=None,
                    help='pial_Velocity.nii.gz: trace each ribbon voxel back '
                         "along DiReCT's own field instead of walking the "
                         'voxel graph')
    ap.add_argument('--euclidean', action='store_true',
                    help='ablation: nearest seed in R^3 instead of through the '
                         'ribbon, i.e. allowed to jump a sulcus')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    from dldirect.field_pial_prototype import make_transforms
    from dldirect import regional_stats as rs
    fsio = nib.freesurfer.io
    ref = nib.load(os.path.join(args.case, 'mri', 'aparc.atlas+aseg.nii.gz'))
    parc = np.asarray(ref.dataobj).astype(np.int32)
    spacing = tuple(float(z) for z in ref.header.get_zooms()[:3])
    tovox, _ = make_transforms(ref)
    lut, valid, _ = rs.get_labels()

    gm = np.zeros(parc.shape, bool)
    seg = np.asarray(nib.load(os.path.join(args.case, 'softmax_seg.nii.gz')
                              ).dataobj).astype(np.int32)
    gm |= (parc > 1000)                 # the existing ribbon defines the domain
    print('%s: ribbon = %d voxels' % (os.path.basename(args.case), gm.sum()))

    all_v, all_l = [], []
    for h in args.hemi:
        w, wf = fsio.read_geometry(os.path.join(args.case, args.run, '%s.white' % h))
        wv = tovox(np.asarray(w, float))
        if args.logit_dir:
            # Seeds independent of the parcellation we are trying to correct:
            # the segmentation model's own per-parcel logits, sampled at the
            # white vertices. Every other seeding here resamples the source,
            # so it can only carry the source's errors back out.
            from dldirect.mesh_crf import model_unary
            hn = [k for k in valid if k.startswith('%s-' % h) and lut[k] > 1000]
            _u, pr = model_unary(wv, args.logit_dir, hn)
            hl = np.array([lut[k] for k in hn])
            vl = hl[pr.argmax(1)]
        elif args.annot:
            lab, _ct, nm = fsio.read_annot(args.annot.replace('?h', h))
            nm = [n.decode() if isinstance(n, bytes) else n for n in nm]
            key = {n.replace('%s-' % h, ''): lut[n] for n in valid
                   if n.startswith('%s-' % h) and lut[n] > 1000}
            m = np.array([key.get(n, 0) for n in nm], np.int64)
            vl = m[np.clip(lab, 0, len(nm) - 1)]
        else:
            vl = rs.nearest_parcel(np.rint(wv).astype(int), parc)
        all_v.append(wv); all_l.append(vl)
    vv = np.vstack(all_v); vl = np.concatenate(all_l)
    print('  %d vertices, %d carrying a cortical label' % (len(vv), (vl > 1000).sum()))

    if args.field:
        vel = np.asarray(nib.load(args.field).dataobj, dtype=np.float32)
        if vel.shape[:3] != parc.shape:
            raise SystemExit('velocity grid %s != parcellation grid %s'
                             % (vel.shape[:3], parc.shape))
        _tovox, totkr = make_transforms(ref)
        vt = np.vstack([np.asarray(fsio.read_geometry(
            os.path.join(args.case, args.run, '%s.white' % h))[0], float)
            for h in args.hemi])
        out = propagate_by_field(gm, vel, vt, vl, tovox, totkr)
    elif args.euclidean:
        from scipy import spatial
        coords = np.array(np.nonzero(gm)).T
        keep = vl > 1000
        d, j = spatial.cKDTree(vv[keep] * np.asarray(spacing)).query(
            coords * np.asarray(spacing))
        out = np.zeros(gm.shape, np.int32)
        out[coords[:, 0], coords[:, 1], coords[:, 2]] = vl[keep][j]
        print('  EUCLIDEAN ablation: nearest vertex in R^3')
    else:
        out, _dist = propagate(gm, spacing, vv, vl)

    if args.fs_dir:
        fsa = nib.load(os.path.join(args.fs_dir, 'mri', 'aparc+aseg.mgz'))
        fsd = np.asarray(fsa.dataobj).astype(np.int32)
        if fsd.shape != out.shape:
            # FreeSurfer's aparc+aseg is on its own 256^3 conform. Resample it
            # onto our grid through SCANNER RAS: our conformed volumes carry
            # the tkr affine, so their world position comes from
            # mri/conform_vox2ras.txt, never from the header.
            from scipy.ndimage import map_coordinates
            ours_to_world = np.loadtxt(os.path.join(args.case, 'mri',
                                                    'conform_vox2ras.txt'))
            M = np.linalg.inv(fsa.affine) @ ours_to_world
            gc = np.array(np.nonzero(np.ones(out.shape, bool))).T
            q = nib.affines.apply_affine(M, gc.astype(float))
            samp = map_coordinates(fsd.astype(np.float32), q.T, order=0,
                                   mode='constant', cval=0)
            fsd = samp.reshape(out.shape).astype(np.int32)
            print('  resampled FreeSurfer aparc+aseg onto our grid '
                  '(%d cortical voxels)' % int((fsd > 1000).sum()))
        if True:
            labs = [l for l in np.unique(out) if l > 1000]
            ds = []
            for l in labs:
                A, B = out == l, fsd == l
                s = A.sum() + B.sum()
                if s:
                    ds.append(2.0 * (A & B).sum() / s)
            print('  mean Dice vs FreeSurfer aparc+aseg: %.4f over %d parcels'
                  % (np.mean(ds), len(ds)))
            base = []
            for l in labs:
                A, B = parc == l, fsd == l
                s = A.sum() + B.sum()
                if s:
                    base.append(2.0 * (A & B).sum() / s)
            print('  (existing aparc.atlas+aseg for reference: %.4f)' % np.mean(base))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        img = nib.Nifti1Image(out, ref.affine)
        img.header['xyzt_units'] = 2
        nib.save(img, args.out)
        print('  wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
