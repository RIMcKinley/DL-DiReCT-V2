"""Sulcal depth as distance from the white surface to a hull over the pial.

STATUS: research prototype. Nothing imports it; run it directly.

The idea
--------
FreeSurfer's ?h.sulc is the integrated displacement of a vertex during
inflation -- a quantity with no length interpretation and no reference
surface. A geometric alternative: wrap the pial in a hull that bridges the
sulci and follows the gyral crowns, then measure how far each WHITE vertex
lies beneath it.

    hull   = closing_R(pial)        a sulcus narrower than 2R is bridged,
                                    because the dilation joins its banks and
                                    the erosion cannot reopen a gap that is
                                    no longer there; a crown is restored.
    depth(v) = distance from v to the hull's SURFACE, in mm

A crown vertex sits just under the hull and scores ~0; a fundus vertex sits
R-deep or more. Unlike the inflation measure this is in millimetres, has an
explicit reference, and R states outright what counts as "a sulcus".

WHY THE HULL COMES FROM THE PIAL MESH, not the segmentation
-----------------------------------------------------------
outer_surface.cortical_envelope closes the ribbon taken from the label
volume, which is the right input when no surface exists yet. Here one does:
closing the rasterised PIAL ties the hull to the surface this pipeline
actually produced, so the depth is measured against our own geometry rather
than against a segmentation that the pial may have departed from.

WHAT THIS IS NOT
----------------
Not a replacement for ?h.sulc until it is shown to agree with, or to beat,
it on something. The script therefore reports the correlation against an
inflation-derived sulc on the SAME vertices, and against FreeSurfer's own
?h.sulc where a recon-all run is available, rather than asserting the
measure is good.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RADIUS_MM = 3.0          # ball radius; sulci narrower than 2R are bridged


# ---------------------------------------------------------------------------

def rasterize(verts_vox, faces, shape, supersample=3):
    from dldirect.field_pial_prototype import rasterize_mesh_pv
    return rasterize_mesh_pv(np.asarray(verts_vox, float), np.asarray(faces),
                             shape, supersample) > 0.5


def close_by_ball(mask, radius_mm, spacing):
    """Morphological closing by a true ball of `radius_mm`.

    Exact Euclidean distance transforms rather than a discrete structuring
    element, so the radius is in millimetres and is not quantised to the
    voxel lattice. Copied in spirit from outer_surface.close_by_ball; kept
    local so this prototype does not drag in pymeshlab.
    """
    dil = ndi.distance_transform_edt(~mask, sampling=spacing) <= radius_mm
    ero = ndi.distance_transform_edt(dil, sampling=spacing) > radius_mm
    return ndi.binary_fill_holes(ero)


def hull_depth_field(pial_mask, radius_mm, spacing):
    """(depth, hull). depth is mm below the hull surface, 0 outside the hull.

    distance_transform_edt of the hull gives, at every voxel inside it, the
    distance to the nearest voxel outside -- i.e. to the hull's surface.
    That is the quantity wanted, and it is a true point-to-surface distance
    rather than a nearest-VERTEX one.
    """
    hull = close_by_ball(pial_mask, radius_mm, spacing)
    depth = ndi.distance_transform_edt(hull, sampling=spacing).astype(np.float32)
    return depth, hull


def geodesic_smooth(values, verts, faces, iters, adjacency=None):
    """Smooth a per-vertex field ALONG THE SURFACE, not through space.

    Repeated 1-ring averaging on the mesh graph -- a discrete heat kernel.
    The point of doing it on the graph is that two vertices on opposite banks
    of a sulcus may be 2 mm apart in R^3 and 40 mm apart along the cortex; a
    3-D Gaussian mixes them and destroys exactly the contrast this is meant to
    measure, while the graph kernel gives the far bank zero weight however
    close it is in space.

    `iters` sets the width. The kernel's spread grows like
    mean_edge_length * sqrt(iters), so the effective geodesic sigma is
    reported rather than assumed -- see calibrate_geodesic_sigma.
    """
    from dldirect.field_pial_prototype import _mesh_adjacency
    if adjacency is None:
        _, Wm, deg = _mesh_adjacency(np.asarray(verts, float), np.asarray(faces))
    else:
        Wm, deg = adjacency
    x = np.asarray(values, np.float64).copy()
    for _ in range(int(iters)):
        # average of the vertex and its 1-ring, which keeps a constant field
        # constant and so does not shrink the mean
        x = (x + (Wm @ x) / deg) * 0.5
    return x.astype(np.float32)


def geodesic_zscore(values, verts, faces, iters, adjacency=None, floor=None):
    """(z, mean, std): how unusual a vertex is against its own neighbourhood.

    Both moments are taken with the SAME geodesic kernel, so

        z = (d - E[d]) / sd[d],   sd = sqrt(E[d^2] - E[d]^2)

    Preferred over a percentage deviation (d - E[d]) / E[d]. The local MEAN
    goes small on a gyral crown -- that is what a crown is -- so a ratio to it
    explodes exactly where the cortex is most ordinary. The local SD does not:
    in folded cortex every neighbourhood spans crown and fundus, so it stays
    bounded away from zero. Dividing by variability also asks the question one
    actually wants: not "how much deeper in mm", which scales with how deep the
    region is overall, but "how far from typical for here".

    `floor` guards the flat patches where the SD genuinely does collapse (a
    broad crown, the medial wall). Default: the 1st percentile of the SD, so
    the guard is set by the surface rather than by a constant.
    """
    from dldirect.field_pial_prototype import _mesh_adjacency
    if adjacency is None:
        _, Wm, deg = _mesh_adjacency(np.asarray(verts, float), np.asarray(faces))
    else:
        Wm, deg = adjacency
    d = np.asarray(values, np.float64)
    m1 = np.asarray(geodesic_smooth(d, verts, faces, iters, (Wm, deg)), float)
    m2 = np.asarray(geodesic_smooth(d * d, verts, faces, iters, (Wm, deg)), float)
    var = np.maximum(m2 - m1 * m1, 0.0)
    sd = np.sqrt(var)
    if floor is None:
        floor = float(np.percentile(sd[sd > 0], 1)) if (sd > 0).any() else 1e-6
    z = (d - m1) / np.maximum(sd, floor)
    return z.astype(np.float32), m1.astype(np.float32), sd.astype(np.float32)


def crf_edge_weights(z, verts, faces, theta=1.0, floor=0.05):
    """Pairwise weights for a contrast-sensitive Potts term on the mesh.

    Returns (edges[E, 2], w[E]) with w in (floor, 1]. Intended use is a
    smoothness cost that is WEAKENED where an edge crosses a sulcal fundus:

        cost(i, j) = w_ij * [label_i != label_j]

    so neighbours are pulled together inside a gyral bank and left free to
    disagree across a fundus -- the mesh analogue of an edge-aware CRF in
    images, with sulcal depth playing the part of the image gradient.

    The driving quantity is the geodesic z-score, not raw depth. Raw depth is
    high across a whole sulcal BASIN, which would loosen the prior over both
    banks and the fundus alike; the z-score is high only where depth exceeds
    its own neighbourhood, which is the fundus LINE. Measured on OAS30072,
    AUC for picking out fundus-line vertices: z-score 0.856/0.845 against raw
    depth 0.748/0.759 (and FreeSurfer's inflation sulc 0.851/0.845).

    An edge takes the MAXIMUM of its endpoints' z: a fundus line is one vertex
    wide in places, and an edge stepping across it should be loosened even if
    only one end sits on the line.

    `theta` sets how fast the weight decays; `floor` keeps the term from
    vanishing entirely, so a fundus never licenses a completely free label
    flip.
    """
    f = np.asarray(faces)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    zz = np.asarray(z, float)
    s = np.maximum(zz[e[:, 0]], zz[e[:, 1]])
    w = np.exp(-np.maximum(s, 0.0) / float(theta))
    return e, np.maximum(w, floor).astype(np.float32)


def calibrate_geodesic_sigma(verts, faces, iters, n_probe=8, seed=0):
    """Effective geodesic sigma of `iters` smoothing steps, in mm.

    Measured, not derived: diffuse a delta from a few probe vertices and take
    the kernel-weighted RMS graph distance. Also returns how much of the
    kernel's mass sits on vertices that are NEAR IN SPACE but far along the
    surface -- the opposite-bank leak a 3-D kernel would suffer and this one
    should not.
    """
    from scipy.sparse.csgraph import dijkstra
    from dldirect.field_pial_prototype import _mesh_adjacency
    v = np.asarray(verts, float)
    _, Wm, deg = _mesh_adjacency(v, np.asarray(faces))
    rng = np.random.default_rng(seed)
    probes = rng.choice(len(v), size=n_probe, replace=False)
    # edge-length-weighted graph, for true geodesic distance
    Wl = Wm.tocoo()
    d = np.linalg.norm(v[Wl.row] - v[Wl.col], axis=1)
    import scipy.sparse as sp
    G = sp.csr_matrix((d, (Wl.row, Wl.col)), shape=Wm.shape)
    sig, leak = [], []
    for pidx in probes:
        delta = np.zeros(len(v)); delta[pidx] = 1.0
        k = geodesic_smooth(delta, v, faces, iters, adjacency=(Wm, deg))
        k = np.asarray(k, float)
        if k.sum() <= 0:
            continue
        k = k / k.sum()
        gd = dijkstra(G, indices=pidx, limit=200.0)
        ok = np.isfinite(gd) & (k > 0)
        sig.append(np.sqrt((k[ok] * gd[ok] ** 2).sum()))
        near3d = np.linalg.norm(v - v[pidx], axis=1) < 3.0
        far_geo = near3d & (gd > 10.0)          # other bank: close in R^3, far on the sheet
        leak.append(k[far_geo].sum())
    return float(np.mean(sig)), float(np.mean(leak)), len(probes)


def sample(vol, pos_vox):
    return ndi.map_coordinates(vol, np.asarray(pos_vox).T, order=1,
                               mode='nearest')


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('WHY THE HULL')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', help='an OASIS-style case directory')
    ap.add_argument('--run', default='field_pial_sigma0.65',
                    help='run directory holding ?h.white and ?h.pial')
    ap.add_argument('--out', default=None, help='write ?h.hulldepth here')
    ap.add_argument('--radius', type=float, nargs='+', default=[RADIUS_MM],
                    help='ball radius in mm; several sweeps them')
    ap.add_argument('--hemi', nargs='+', default=['lh', 'rh'])
    ap.add_argument('--threshold', type=float, default=7.5,
                    help='depth in mm above which a vertex is called sulcal '
                         '(default %(default)s, the value that best matched an '
                         'inflation sulc on the development subject)')
    ap.add_argument('--deviation-iters', type=int, default=None,
                    help='also write ?h.hulldev: the depth minus its geodesic '
                         'local mean over this many 1-ring smoothing steps. '
                         'Positive = deeper than its surroundings (fundus), '
                         'negative = shallower (crown). See geodesic_smooth.')
    ap.add_argument('--write-hull', action='store_true',
                    help='also write the hull itself as ?h.hull, to see what '
                         'the closing actually bridged')
    ap.add_argument('--sulc', default=None,
                    help='a ?h.sulc on the SAME white mesh (from mris_inflate) '
                         'to correlate against')
    ap.add_argument('--fs-dir', default=None,
                    help="a recon-all subject dir, to compare with its ?h.sulc")
    args = ap.parse_args(argv)

    from dldirect.field_pial_prototype import make_transforms
    fsio = nib.freesurfer.io
    ref = nib.load(os.path.join(args.case, 'mri', 'aparc.atlas+aseg.nii.gz'))
    spacing = tuple(float(z) for z in ref.header.get_zooms()[:3])
    shape = tuple(ref.shape[:3])
    tovox, _ = make_transforms(ref)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    for h in args.hemi:
        w, wf = fsio.read_geometry(os.path.join(args.case, args.run, '%s.white' % h))
        p, pf = fsio.read_geometry(os.path.join(args.case, args.run, '%s.pial' % h))
        w_vox = tovox(np.asarray(w, float))
        pial_mask = rasterize(tovox(np.asarray(p, float)), pf, shape)
        print('%s: pial encloses %d voxels' % (h, int(pial_mask.sum())))

        for R in args.radius:
            depth_vol, hull = hull_depth_field(pial_mask, R, spacing)
            d = sample(depth_vol, w_vox).astype(np.float32)
            bridged = int(hull.sum() - pial_mask.sum())
            print('  R=%.1f mm: hull adds %d voxels (%.1f%%) | depth mean %.2f '
                  'median %.2f p95 %.2f max %.2f mm'
                  % (R, bridged, 100.0 * bridged / max(1, int(pial_mask.sum())),
                     d.mean(), np.median(d), np.percentile(d, 95), d.max()))
            if args.sulc:
                s = fsio.read_morph_data(args.sulc.replace('?h', h))
                if len(s) == len(d):
                    print('     vs inflation sulc on the same mesh: r = %+.3f'
                          % np.corrcoef(d, s)[0, 1])
            if args.fs_dir:
                fp = os.path.join(args.fs_dir, 'surf', '%s.sulc' % h)
                fw = os.path.join(args.fs_dir, 'surf', '%s.white' % h)
                if os.path.exists(fp) and os.path.exists(fw):
                    from scipy import spatial
                    from dldirect.compare_surfaces import tkr_to_world
                    fsv, _ = fsio.read_geometry(fw)
                    fsref = nib.load(os.path.join(args.fs_dir, 'mri', 'orig.mgz'))
                    M = np.linalg.inv(tkr_to_world(
                        ref, np.loadtxt(os.path.join(args.case, 'mri',
                                                     'conform_vox2ras.txt')))) \
                        @ tkr_to_world(fsref)
                    fsv = nib.affines.apply_affine(M, np.asarray(fsv, float))
                    j = spatial.cKDTree(fsv).query(np.asarray(w, float))[1]
                    fsulc = fsio.read_morph_data(fp)[j]
                    print("     vs FreeSurfer's own ?h.sulc (nearest vertex): "
                          'r = %+.3f' % np.corrcoef(d, fsulc)[0, 1])
            if args.out and len(args.radius) == 1:
                path = os.path.join(args.out, '%s.hulldepth' % h)
                fsio.write_morph_data(path, d, fnum=len(wf))
                print('     wrote %s' % path)
                # a binary sulcus/gyrus label at the chosen threshold, so the
                # partition can be inspected rather than inferred from a ramp
                lab = (d > args.threshold).astype(np.float32)
                fsio.write_morph_data(os.path.join(args.out, '%s.sulcuslabel' % h),
                                      lab, fnum=len(wf))
                print('     wrote %s.sulcuslabel (%.1f%% sulcal at %.2f mm)'
                      % (h, 100.0 * lab.mean(), args.threshold))
                if args.deviation_iters:
                    sm = geodesic_smooth(d, w, wf, args.deviation_iters)
                    dev = (d - sm).astype(np.float32)
                    sig, leak, _n = calibrate_geodesic_sigma(
                        w, wf, args.deviation_iters, n_probe=4)
                    fsio.write_morph_data(
                        os.path.join(args.out, '%s.hulldev' % h), dev, fnum=len(wf))
                    print('     wrote %s.hulldev: %d steps = geodesic sigma '
                          '%.2f mm (other-bank leak %.1e) | dev p5 %+.2f '
                          'median %+.2f p95 %+.2f mm'
                          % (h, args.deviation_iters, sig, leak,
                             np.percentile(dev, 5), np.median(dev),
                             np.percentile(dev, 95)))
                if args.deviation_iters:
                    z, _m1, zsd = geodesic_zscore(d, w, wf, args.deviation_iters)
                    fsio.write_morph_data(
                        os.path.join(args.out, '%s.hullz' % h), z, fnum=len(wf))
                    print('     wrote %s.hullz: local SD min %.3f p1 %.3f '
                          'median %.2f mm | z p1 %+.2f median %+.2f p99 %+.2f'
                          % (h, zsd.min(), np.percentile(zsd, 1),
                             np.median(zsd), np.percentile(z, 1),
                             np.median(z), np.percentile(z, 99)))
                if args.write_hull:
                    from skimage import measure
                    _, totkr = make_transforms(ref)
                    hv, hf, _, _ = measure.marching_cubes(
                        hull.astype(np.float32), level=0.5,
                        spacing=(1.0, 1.0, 1.0))
                    hp = os.path.join(args.out, '%s.hull' % h)
                    fsio.write_geometry(hp, totkr(hv).astype(np.float32), hf,
                                        create_stamp=None, volume_info=None)
                    print('     wrote %s (%d vertices)' % (hp, len(hv)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
