#!/usr/bin/env python
"""Trace two vertices on opposite sides of a pial self-intersection back to their origin.

Re-runs the propagation with `return_trajectory=True` (reusing the run's own saved
velocity field, so nothing upstream is re-solved and the surface is bit-identical to
the one the pipeline wrote), finds an exact triangle-triangle crossing, picks the
closest vertex from each of the two crossing faces, and plots both trajectories from
the white surface to the pial.

The pair is chosen to be far apart ON THE MESH: two faces that cross while sitting
next to each other in the graph are a local fold, and their vertices share an origin,
which is not the case worth looking at. Sorting candidates by mesh hop distance picks
out crossings between patches that started far apart and travelled into each other.

Usage:
    python -m dldirect.visualize_tangle_origin --run-dir <dl+direct-nofs output> \
        --out-dir <figure directory> [--hemi lh] [--rank 0]
"""

import argparse
import os

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.sparse.csgraph import dijkstra

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates, distance_transform_edt

from scipy import spatial

from .field_pial_prototype import (
    load_gm_wm_probability, build_seg_maps, make_transforms, rasterize_mesh,
    rasterize_mesh_pv, reconcile_seg_with_surface, build_no_push_mask,
    build_pin_mask, propagate, _mesh_adjacency, _self_intersecting_faces,
    _crossing_pairs, BEST_CONFIG)
from .visualize_field_travel import integrate as integrate_field


# ---------------------------------------------------------------------------
# Crossing pairs
# ---------------------------------------------------------------------------

def _seg_tri_hit(O, Dv, V0, E1, E2, eps=1e-9):
    """Vectorised Moller-Trumbore: does segment (O -> O+Dv) pass through the
    triangle (V0, V0+E1, V0+E2)? All arrays (M, 3); returns (M,) bool."""
    P = np.cross(Dv, E2)
    det = np.einsum('ij,ij->i', E1, P)
    ok = np.abs(det) >= 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    T = O - V0
    u = np.einsum('ij,ij->i', T, P) * inv
    ok &= (u >= -eps) & (u <= 1 + eps)
    Q = np.cross(T, E1)
    v = np.einsum('ij,ij->i', Dv, Q) * inv
    ok &= (v >= -eps) & (u + v <= 1 + eps)
    t = np.einsum('ij,ij->i', E2, Q) * inv
    return ok & (t >= -eps) & (t <= 1 + eps)


def crossing_pairs_fast(verts, faces, cand_faces, radius=None):
    """Same result as field_pial_prototype._crossing_pairs, vectorised.

    pymeshlab's exact per-face test says WHICH faces self-intersect but not
    which face crosses which, and pymeshlab exposes no pair-wise filter. So the
    pairing is done here, restricted to the faces pymeshlab already selected:
    a KD-tree over their centroids gives the candidate pairs, and one
    vectorised Moller-Trumbore pass over all of them at once replaces the
    per-pair Python loop. Verified against _crossing_pairs in main().
    """
    verts = np.asarray(verts, float); faces = np.asarray(faces)
    cand = np.asarray(cand_faces)
    cent_all = verts[faces].mean(1)
    if radius is None:
        e = np.linalg.norm(verts[faces[:, 1]] - verts[faces[:, 0]], axis=1)
        radius = float(np.percentile(e, 99) * 2.0)

    # Candidate pairs: both faces selected, centroids within `radius`. A crossing
    # always has BOTH its faces in pymeshlab's selection, so restricting to the
    # selection loses nothing.
    tree = spatial.cKDTree(cent_all[cand])
    ij = np.asarray(list(tree.query_pairs(radius)), dtype=np.int64)
    if len(ij) == 0:
        return np.empty((0, 2), np.int64)
    fa, fb = cand[ij[:, 0]], cand[ij[:, 1]]

    A, B = faces[fa], faces[fb]
    shared = (A[:, :, None] == B[:, None, :]).any(axis=(1, 2))
    fa, fb = fa[~shared], fb[~shared]
    if len(fa) == 0:
        return np.empty((0, 2), np.int64)

    TA, TB = verts[faces[fa]], verts[faces[fb]]
    hit = np.zeros(len(fa), bool)
    for T, U in ((TA, TB), (TB, TA)):
        V0 = U[:, 0]; E1 = U[:, 1] - V0; E2 = U[:, 2] - V0
        for k in range(3):
            O = T[:, k]; Dv = T[:, (k + 1) % 3] - O
            hit |= _seg_tri_hit(O, Dv, V0, E1, E2)
    return np.stack([fa[hit], fb[hit]], axis=1)


# ---------------------------------------------------------------------------
# Reproduce the pipeline's propagation, keeping the trajectory
# ---------------------------------------------------------------------------

def load_anat(run_dir, shape):
    """The cropped T1 on the segmentation's own grid, or None."""
    for name in ('T1w_norm_noskull_cropped.nii.gz', 'T1w_norm_cropped.nii.gz'):
        f = os.path.join(run_dir, name)
        if os.path.exists(f):
            a = np.asarray(nib.load(f).dataobj).astype(np.float32)
            if a.shape == tuple(shape):
                return a
    return None


def rebuild_cached(run_dir, hemi, cache_path, velocity_path=None):
    """rebuild(), memoised to an .npz. The propagation is deterministic given the
    saved velocity field, so a cache hit is the same surface -- main() still
    checks it against the written pial either way."""
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        print('trajectory cache: %s' % cache_path)
        D = {k: z[k] for k in z.files}
        ref = nib.load(os.path.join(run_dir, 'seg_Left-Cerebral-Cortex.nii.gz'))
        D['tovox'], D['totkr'] = make_transforms(ref)
        D['anat'] = load_anat(run_dir, D['seg'].shape)
        D['velocity'] = np.asarray(nib.load(
            velocity_path or os.path.join(run_dir, 'field_pial',
                                          'best_Velocity.nii.gz')).dataobj)
        return D
    D = rebuild(run_dir, hemi, velocity_path)
    if cache_path:
        np.savez_compressed(
            cache_path, white=D['white'], faces=D['faces'], pial=D['pial'],
            traj=D['traj'], seg=D['seg'], pin=D['pin'], no_push=D['no_push'])
        print('wrote trajectory cache: %s' % cache_path)
    return D


def rebuild(run_dir, hemi, velocity_path=None):
    """Everything main() sets up, then propagate() with the trajectory kept.

    Uses the saved velocity field rather than re-solving: nothing that
    distinguishes the propagation is an input to the solve, so reusing it makes
    this provably the same field the written pial rode.
    """
    cfg = BEST_CONFIG
    if velocity_path is None:
        velocity_path = os.path.join(run_dir, 'field_pial', 'best_Velocity.nii.gz')

    gm_prob, wm_prob, ref_img = load_gm_wm_probability(run_dir)
    seg, gmT, wmT = build_seg_maps(gm_prob, wm_prob)
    tovox, totkr = make_transforms(ref_img)
    velocity = np.asarray(nib.load(velocity_path).dataobj)

    soft_seg = np.asarray(nib.load(os.path.join(run_dir, 'softmax_seg.nii.gz')).dataobj)
    _df = pd.read_csv(os.path.join(run_dir, 'label_def.csv'))
    id_map = {r.LABEL: int(r.ID) for _, r in _df.iterrows()}

    surf_dir = os.path.join(run_dir, 'surf')
    hemi_verts = {}
    for h in ('lh', 'rh'):
        hemi_verts[h] = nib.freesurfer.io.read_geometry(
            os.path.join(surf_dir, '%s.white' % h))

    # WM from the white surface (the default), at partial volume — main()'s block.
    shape = tuple(ref_img.shape[:3])
    m = np.zeros(shape, np.float32)
    crisp = np.zeros(shape, bool)
    for h in ('lh', 'rh'):
        v, f = hemi_verts[h]
        crisp |= rasterize_mesh(tovox(v), f, shape)
        m = m + rasterize_mesh_pv(tovox(v), f, shape, 3)
    m = np.clip(m, 0.0, 1.0)
    seg, gmT, wmT, _, _ = reconcile_seg_with_surface(seg, gmT, wmT, m, label_mask=crisp)

    white, faces = hemi_verts[hemi]
    mask = build_no_push_mask(white, faces, seg, soft_seg, id_map, tovox,
                              rings=cfg.pin_rings)
    pin = build_pin_mask(mask, white, faces, seg, soft_seg, id_map, tovox,
                         scope=cfg.pin_scope, rings=cfg.pin_rings)

    pial, traj = propagate(
        white, faces, velocity, seg, tovox, totkr,
        mode=cfg.propagation_mode, rounds=cfg.propagation_rounds,
        floor=cfg.constrained_floor, lam=cfg.constrained_lam,
        use_escape=cfg.use_escape, floor_after=cfg.constrained_floor_after,
        floor_after_round=cfg.constrained_floor_after_round,
        no_push=mask, pin_mask=pin, pin_feather=cfg.pin_feather,
        escape_in_no_push=cfg.escape_in_no_cortex, step_scale=cfg.step_scale,
        return_trajectory=True)
    return dict(white=white, faces=faces, pial=pial, traj=traj, seg=seg,
                tovox=tovox, totkr=totkr, ref_img=ref_img, pin=pin, no_push=mask,
                anat=load_anat(run_dir, seg.shape), velocity=velocity)


# ---------------------------------------------------------------------------
# Starting from the WM voxel centre instead of the surface vertex
# ---------------------------------------------------------------------------

def wm_voxel_centres(pos_tkr, seg, tovox, wm_label=3):
    """Voxel-index centres of the WM voxel each position sits in.

    A white-surface vertex lies ON the WM/GM interface, so the voxel it rounds
    into is not always labelled WM. Where it is not, the nearest WM voxel is
    used -- that is the voxel DiReCT would have seeded, since its contour is
    defined on WM voxels adjacent to GM.

    Returns (centres_vox, snapped_bool) where `snapped` marks the positions that
    had to move to a neighbouring voxel rather than their own.
    """
    vox = tovox(np.atleast_2d(pos_tkr))
    idx = np.rint(vox).astype(int)
    idx = np.clip(idx, 0, np.array(seg.shape) - 1)
    inside = seg[tuple(idx.T)] == wm_label
    if not inside.all():
        # Nearest WM voxel for every position that did not land in one.
        _, ind = distance_transform_edt(seg != wm_label, return_indices=True)
        near = np.stack([ind[k][tuple(idx.T)] for k in range(3)], axis=1)
        idx = np.where(inside[:, None], idx, near)
    return idx.astype(float), ~inside


# ---------------------------------------------------------------------------
# Pick the pair
# ---------------------------------------------------------------------------

HOP_LIMIT = 80.0


def find_pair(pial, faces, pin, rank=0, sample=400, seed=0):
    """An exact crossing whose two faces are far apart on the mesh.

    Returns (ia, ib, fa, fb, hops, all_ranked).
    """
    sel = _self_intersecting_faces(pial, faces)
    if sel is None:
        raise SystemExit('pymeshlab unavailable: cannot locate crossings exactly')
    cand = np.flatnonzero(sel)
    pairs = crossing_pairs_fast(pial, faces, cand)
    if len(pairs) == 0:
        raise SystemExit('no exact triangle-triangle crossings found')
    print('%d intersecting faces, %d exact crossing pairs' % (len(cand), len(pairs)))

    rng = np.random.default_rng(seed)
    if len(pairs) > sample:
        pairs = pairs[rng.choice(len(pairs), sample, replace=False)]

    # Drop pairs touching a pinned vertex: a pinned vertex never left the white
    # surface, so "tracking it back" would show nothing.
    if pin is not None:
        keep = ~(pin[faces[pairs[:, 0]]].any(1) | pin[faces[pairs[:, 1]]].any(1))
        pairs = pairs[keep]
        if len(pairs) == 0:
            raise SystemExit('every sampled crossing involves a pinned vertex')

    # Mesh hop distance, one batched Dijkstra over the unique sources. `limit`
    # prunes the search: anything beyond it comes back inf, which is exactly the
    # "far apart on the mesh" end we are selecting for, so it is reported as
    # ">limit" rather than treated as an error.
    _, Wm, _ = _mesh_adjacency(pial, faces)
    src = np.unique(faces[pairs[:, 0], 0])
    row = {int(v): i for i, v in enumerate(src)}
    dist = dijkstra(Wm, directed=False, unweighted=True, indices=src, limit=HOP_LIMIT)

    ranked = []
    for fa, fb in pairs:
        va, vb = faces[fa], faces[fb]
        d = dist[row[int(va[0])], vb]
        hops = float(np.nanmin(d))
        if not np.isfinite(hops):
            hops = HOP_LIMIT + 1             # beyond the search limit
        # The two closest vertices across the crossing.
        D = np.linalg.norm(pial[va][:, None] - pial[vb][None], axis=2)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        ranked.append((hops, int(va[i]), int(vb[j]), int(fa), int(fb)))

    ranked.sort(key=lambda r: -r[0])
    hops, ia, ib, fa, fb = ranked[rank]
    return ia, ib, fa, fb, hops, ranked


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def oblique_plane(*paths):
    """Orthonormal (e1, e2, origin, rms_out_of_plane) of the plane best
    containing every path passed. Fitting it to only some of the paths would
    project the rest at an angle and misrepresent their length."""
    pts = np.vstack(paths)
    o = pts.mean(0)
    u, sv, vt = np.linalg.svd(pts - o, full_matrices=False)
    rms = float(np.sqrt(np.mean(((pts - o) @ vt[2]) ** 2)))
    return vt[0], vt[1], o, rms


def hop_str(h):
    return '>%d' % int(HOP_LIMIT) if h > HOP_LIMIT else '%d' % int(h)


def voxel_traces(D, ia, ib, steps=10):
    """Ten unconstrained steps of the field from each vertex's WM voxel centre.

    Same integration as the boundary-voxel figures (p <- p - v(p), trilinear
    sample at the current position) and NONE of the propagation's machinery --
    no relaxation, no signed-distance floor, no pin. So the difference between
    this and the mesh trajectory is exactly what those constraints did.
    """
    starts_tkr = np.stack([D['traj'][0, ia], D['traj'][0, ib]])
    c_vox, snapped = wm_voxel_centres(starts_tkr, D['seg'], D['tovox'])
    tr_vox = integrate_field(c_vox, D['velocity'], steps=steps)
    tr_tkr = np.stack([D['totkr'](tr_vox[t]) for t in range(len(tr_vox))])
    offset = np.linalg.norm(tr_tkr[0] - starts_tkr, axis=1)
    return tr_tkr, offset, snapped


def figure(out, D, ia, ib, fa, fb, hops, half=12.0, vox=None):
    white, faces, pial, traj = D['white'], D['faces'], D['pial'], D['traj']
    seg, tovox, anat = D['seg'], D['tovox'], D.get('anat')
    ta, tb = traj[:, ia], traj[:, ib]
    paths = [ta, tb] + ([vox[0][:, 0], vox[0][:, 1]] if vox is not None else [])
    e1, e2, o, rms = oblique_plane(*paths)

    fig = plt.figure(figsize=(19, 8.2))

    # --- left: the anatomy in the plane the two paths span ------------------
    ax = fig.add_subplot(1, 3, 1)
    n = 361
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g, indexing='ij')
    P = o + X[..., None] * e1 + Y[..., None] * e2          # tkrRAS mm
    V = tovox(P.reshape(-1, 3)).reshape(n, n, 3)
    lab = map_coordinates(seg.astype(np.float32), V.reshape(-1, 3).T,
                          order=0, mode='nearest').reshape(n, n)
    ext = (-half, half, -half, half)
    if anat is not None:
        # Nearest-neighbour would alias badly on an oblique cut; the intensities
        # are only a backdrop, so interpolate them.
        mri = map_coordinates(anat, V.reshape(-1, 3).T, order=1,
                              mode='nearest').reshape(n, n)
        inb = mri[mri > 0]
        lo, hi = np.percentile(inb, [1, 99]) if inb.size else (0.0, 1.0)
        ax.imshow(mri.T, origin='lower', extent=ext, cmap='gray',
                  vmin=lo, vmax=hi, interpolation='bilinear')
        # The segmentation stays, as boundaries rather than as fill, so the
        # tissue the paths cross is still readable against the intensities.
        ax.contour(lab.T >= 2.5, levels=[0.5], colors='#22d3ee',
                   linewidths=0.9, extent=ext)
        ax.contour(lab.T >= 1.5, levels=[0.5], colors='#fb923c',
                   linewidths=0.9, extent=ext)
        legend_note = 'cyan = WM boundary, orange = GM/CSF boundary'
    else:
        # One colour per label value 0..3, with bin edges at the half-integers.
        # A 3-colour map over vmin=0/vmax=3 puts GM (2) and WM (3) in one bin.
        ax.imshow(lab.T, origin='lower', extent=ext,
                  cmap=matplotlib.colors.ListedColormap(
                      ['#0b1020', '#0b1020', '#6b7280', '#f8fafc']),
                  vmin=-0.5, vmax=3.5, interpolation='nearest')
        legend_note = 'dark = background/CSF, grey = GM, white = WM'

    def to2d(p):
        q = np.atleast_2d(p) - o
        return np.stack([q @ e1, q @ e2], axis=1)

    if vox is not None:
        for t, col, name in ((vox[0][:, 0], '#fdba74', 'A from WM voxel centre'),
                             (vox[0][:, 1], '#a5f3fc', 'B from WM voxel centre')):
            q = to2d(t)
            ax.plot(q[:, 0], q[:, 1], '-', color='k', lw=3.0, alpha=0.5, zorder=2)
            ax.plot(q[:, 0], q[:, 1], '--s', color=col, ms=4.5, lw=1.5,
                    markevery=[0], mfc='none', mew=1.5, label=name, zorder=3)
    for t, col, name in ((ta, '#f97316', 'vertex A %d' % ia),
                         (tb, '#22d3ee', 'vertex B %d' % ib)):
        q = to2d(t)
        ax.plot(q[:, 0], q[:, 1], '-', color='k', lw=3.2, alpha=0.55, zorder=4)
        ax.plot(q[:, 0], q[:, 1], '-o', color=col, ms=3.4, lw=1.7, label=name,
                zorder=5)
        ax.plot(q[0, 0], q[0, 1], 'o', color='w', ms=8, mec=col, mew=1.8,
                zorder=6)
        ax.annotate('start %s' % name.split()[1], q[0], textcoords='offset points',
                    xytext=(7, 7), color=col, fontsize=8, zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='k')])
    # The two ends coincide by construction (that is the crossing), so labelling
    # both just stacks two words on one point.
    mid = to2d(0.5 * (ta[-1] + tb[-1]))[0]
    ax.plot(mid[0], mid[1], 'x', color='w', ms=9, mew=2.0, zorder=7)
    ax.annotate('they cross here', mid, textcoords='offset points',
                xytext=(9, -14), color='w', fontsize=8, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='k')])
    ax.set_xlabel('mm along the plane the two paths span')
    ax.set_ylabel('mm')
    ax.set_title('Paths cut through the MRI in the plane they span '
                 '(RMS out of plane %.2f mm)\n%s' % (rms, legend_note), fontsize=9)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85)

    # --- middle: 3D, with the two crossing triangles ------------------------
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    for t, col in ((ta, '#f97316'), (tb, '#22d3ee')):
        ax.plot(t[:, 0], t[:, 1], t[:, 2], '-o', color=col, ms=3, lw=1.6)
        ax.scatter(*t[0], color='w', edgecolor=col, s=55, depthshade=False)
    for f, col in ((fa, '#f97316'), (fb, '#22d3ee')):
        ax.add_collection3d(Poly3DCollection([pial[faces[f]]], facecolor=col,
                                             alpha=0.55, edgecolor='k', lw=0.6))
    pts = np.vstack([ta, tb])
    c = pts.mean(0); r = max(np.abs(pts - c).max(), 2.0) * 1.25
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title('The two crossing triangles and the paths into them\n'
                 '(white marker = origin on the white surface)', fontsize=10)
    ax.tick_params(labelsize=6)

    # --- right: separation and travel, round by round -----------------------
    ax = fig.add_subplot(1, 3, 3)
    sep = np.linalg.norm(ta - tb, axis=1)
    ax.plot(np.arange(len(sep)), sep, '-o', color='#6366f1', ms=4,
            label='A to B (mesh vertices)')
    if vox is not None:
        vsep = np.linalg.norm(vox[0][:, 0] - vox[0][:, 1], axis=1)
        ax.plot(np.arange(len(vsep)), vsep, '--s', color='#a855f7', ms=3.5,
                label='A to B (from WM voxel centres)')
    ax.axhline(0, color='#94a3b8', lw=0.8)
    ax.set_xlabel('propagation round')
    ax.set_ylabel('mm')
    ax2 = ax.twinx()
    for t, col, name in ((ta, '#f97316', 'A travelled'), (tb, '#22d3ee', 'B travelled')):
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))])
        ax2.plot(np.arange(len(cum)), cum, '--', color=col, lw=1.4, label=name)
    ax2.set_ylabel('cumulative path length (mm)')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center left')
    ax.set_title('They start %.2f mm apart and end %.2f mm apart\n'
                 '%s mesh hops apart on the surface'
                 % (sep[0], sep[-1], hop_str(hops)), fontsize=10)

    fig.suptitle('Two vertices on opposite sides of a pial self-intersection, '
                 'traced back to their origin', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return sep


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--hemi', default='lh', choices=['lh', 'rh'])
    p.add_argument('--velocity', default=None)
    p.add_argument('--rank', type=int, default=0,
                   help='which crossing to plot, 0 = the most mesh-distant (default)')
    p.add_argument('--sample', type=int, default=400,
                   help='crossing pairs to rank (default 400)')
    p.add_argument('--no-voxel-start', action='store_true',
                   help='omit the unconstrained trace from each vertex\'s WM '
                        'voxel centre')
    p.add_argument('--half', type=float, default=12.0,
                   help='half-width in mm of the MRI cut (default 12)')
    p.add_argument('--no-cache', action='store_true',
                   help='always re-propagate instead of reusing traj_<hemi>.npz '
                        'in --out-dir')
    p.add_argument('--verify-pairs', type=int, default=300, metavar='N',
                   help='cross-check the vectorised crossing test against '
                        '_crossing_pairs on N of the selected faces (0 = skip)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache = (None if args.no_cache else
             os.path.join(args.out_dir, 'traj_%s.npz' % args.hemi))
    D = rebuild_cached(args.run_dir, args.hemi, cache, args.velocity)

    # The re-propagation must reproduce the written surface exactly, or the
    # crossing located here is not the crossing that is in the file.
    written = os.path.join(args.run_dir, 'field_pial', '%s.pial.field_best' % args.hemi)
    if os.path.exists(written):
        wv, _ = nib.freesurfer.io.read_geometry(written)
        dmax = float(np.abs(wv - D['pial']).max())
        print('max |re-propagated - written pial| = %.3e mm %s'
              % (dmax, '(identical)' if dmax < 1e-4 else '*** DIFFERS ***'))

    if args.verify_pairs:
        sel = _self_intersecting_faces(D['pial'], D['faces'])
        sub = np.flatnonzero(sel)[:args.verify_pairs]
        ref = {tuple(sorted(p)) for p in _crossing_pairs(D['pial'], D['faces'], sub)}
        fastp = crossing_pairs_fast(D['pial'], D['faces'], np.flatnonzero(sel))
        got = {tuple(sorted(map(int, p))) for p in fastp}
        # _crossing_pairs over a SUBSET finds every crossing with at least one
        # face in that subset, so its result must be contained in the full run.
        missing = ref - got
        print('pair test vs _crossing_pairs on %d faces: %d reference pairs, '
              '%d missing from the vectorised result %s'
              % (len(sub), len(ref), len(missing), '(agree)' if not missing else '*** DIFFER ***'))

    ia, ib, fa, fb, hops, ranked = find_pair(D['pial'], D['faces'], D['pin'],
                                             rank=args.rank, sample=args.sample)
    print('chose vertices %d and %d (faces %d, %d), %s mesh hops apart'
          % (ia, ib, fa, fb, hop_str(hops)))
    print('top crossings by mesh distance: %s'
          % ', '.join('%s' % hop_str(r[0]) for r in ranked[:8]))

    out = os.path.join(args.out_dir, 'tangle_origin_%s_rank%d.png' % (args.hemi, args.rank))
    vox = voxel_traces(D, ia, ib) if not args.no_voxel_start else None
    sep = figure(out, D, ia, ib, fa, fb, hops, half=args.half, vox=vox)

    ta, tb = D['traj'][:, ia], D['traj'][:, ib]
    for name, t in (('A', ta), ('B', tb)):
        arc = np.linalg.norm(np.diff(t, axis=0), axis=1).sum()
        print('  vertex %s: travelled %.3f mm, net %.3f mm'
              % (name, arc, np.linalg.norm(t[-1] - t[0])))
    da = ta[-1] - ta[0]; db = tb[-1] - tb[0]
    cos = float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db)))
    print('  angle between the two net travel directions: %.1f deg'
          % np.degrees(np.arccos(np.clip(cos, -1, 1))))
    print('  separation start %.3f mm -> end %.3f mm' % (sep[0], sep[-1]))
    if vox is not None:
        tr, offset, snapped = vox
        vsep = np.linalg.norm(tr[:, 0] - tr[:, 1], axis=1)
        print('from the WM voxel centre (unconstrained, %d steps):' % (len(tr) - 1))
        for k, name in ((0, 'A'), (1, 'B')):
            arc = np.linalg.norm(np.diff(tr[:, k], axis=0), axis=1).sum()
            print('  %s: centre is %.3f mm from the vertex%s; travelled %.3f mm, '
                  'net %.3f mm' % (name, offset[k],
                                   ' (nearest WM voxel)' if snapped[k] else '',
                                   arc, np.linalg.norm(tr[-1, k] - tr[0, k])))
        print('  separation start %.3f mm -> end %.3f mm' % (vsep[0], vsep[-1]))
    print('wrote', out)


if __name__ == '__main__':
    main()
