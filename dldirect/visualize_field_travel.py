#!/usr/bin/env python
"""Visualise where the WM/GM boundary voxels travel in the DiReCT velocity field.

This is the transport DiReCT's thickness image is built from. `extract_wm_contours`
in direct_cuda.py marks every WM voxel with GM in its 26-neighbourhood; the solver
then carries those voxels through the velocity field for `num_integration_points`
steps and records how far each one got. The thickness at a GM voxel is the mean
distance travelled by the contour points that swept through it
(`total_image / hit_image`).

The stepping here is the same one `field_pial_prototype.propagate` applies to
surface vertices, minus the mesh relaxation, the signed-distance floor and the pin:

    p <- p - v(p)      ten times, v sampled trilinearly at the current position

`- v` because DiReCT's velocity convention points from GM into WM, so the outward
direction is its negative (see `propagate.outward_step_tkr`). The field read is
`<field_pial>/best_Velocity.nii.gz`, which is the PER-INTEGRATION-POINT field the
solve composes `num_integration_points` times -- so ten steps of it is the whole
deformation, exactly as ten propagation rounds are for the surface.

Everything is done on the cropped grid at 1mm, in voxel index order (d, h, w),
which is the order the velocity components are stored in.

Usage:
    python -m dldirect.visualize_field_travel --run-dir <dl+direct-nofs output> \
        --out-dir <figure directory> [--slice N] [--steps 10]
"""

import argparse
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import (map_coordinates, binary_dilation, binary_closing,
                           distance_transform_edt, uniform_filter)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_run(run_dir, velocity=None):
    """Load the reconciled segmentation, the velocity field and the anatomy.

    `seg.nii.gz` is the segmentation the solve actually used (written by
    run_pipeline via save_like_direct), i.e. after the WM-from-surface
    reconciliation -- not the raw argmax. Using it means the contour seeded here
    is the same set of voxels the solver seeded.
    """
    if velocity is None:
        velocity = os.path.join(run_dir, 'field_pial', 'best_Velocity.nii.gz')
    seg_img = nib.load(os.path.join(run_dir, 'seg.nii.gz'))
    seg = np.asarray(seg_img.dataobj).astype(np.uint8)
    vel = np.asarray(nib.load(velocity).dataobj).astype(np.float32)
    if vel.shape[:3] != seg.shape:
        raise SystemExit('velocity grid %s does not match seg grid %s'
                         % (vel.shape[:3], seg.shape))

    anat = None
    for name in ('T1w_norm_noskull_cropped.nii.gz', 'T1w_norm_cropped.nii.gz'):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            a = np.asarray(nib.load(p).dataobj).astype(np.float32)
            if a.shape == seg.shape:
                anat = a
                break

    thick = None
    p = os.path.join(run_dir, 'T1w_thickmap.nii.gz')
    if os.path.exists(p):
        t = np.asarray(nib.load(p).dataobj).astype(np.float32)
        if t.shape == seg.shape:
            thick = t

    zooms = np.asarray(seg_img.header.get_zooms()[:3], float)
    return seg, vel, anat, thick, zooms


def wm_contour_mask(seg, gm_label=2, wm_label=3):
    """WM voxels with GM in the 26-neighbourhood.

    Reproduces direct_cuda.extract_wm_contours, whose 3x3x3 max_pool on the GM
    mask is a full-connectivity dilation.
    """
    gm_dilated = binary_dilation(seg == gm_label, structure=np.ones((3, 3, 3), bool))
    return (seg == wm_label) & gm_dilated


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def integrate(seeds, vel, steps=10):
    """Step `seeds` through `vel` for `steps` steps, returning (steps+1, N, 3).

    Identical in form to propagate()'s field step: trilinear sample of the
    per-integration-point field at the CURRENT position, applied as -v, with no
    constraint of any kind imposed between steps.
    """
    pos = np.asarray(seeds, float).copy()
    traj = np.empty((steps + 1, len(pos), 3), float)
    traj[0] = pos
    for t in range(steps):
        v = np.stack([map_coordinates(vel[..., k], pos.T, order=1, mode='nearest')
                      for k in range(3)], axis=1)
        pos = pos - v
        traj[t + 1] = pos
    return traj


def path_lengths(traj, zooms):
    """Cumulative arc length in mm, and straight-line displacement in mm."""
    d = np.diff(traj, axis=0) * zooms
    arc = np.linalg.norm(d, axis=2).sum(axis=0)
    net = np.linalg.norm((traj[-1] - traj[0]) * zooms, axis=1)
    return arc, net


def wm_normals(seg, pos, wm_label=3):
    """Outward unit normal of the WM boundary at each position.

    Built the way propagate() builds it: the signed distance to WM, whose
    normalised gradient points out of WM. At a border WM voxel this is the
    local surface normal of the WM/GM interface.
    """
    wmb = (seg == wm_label)
    sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)
    g = np.stack(np.gradient(sdt), axis=-1)
    gu = g / np.maximum(np.linalg.norm(g, axis=-1), 1e-9)[..., None]
    n = np.stack([map_coordinates(gu[..., k], pos.T, order=1, mode='nearest')
                  for k in range(3)], axis=1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def tangential_travel(seg, traj, zooms):
    """Split each voxel's net displacement into normal and tangential parts.

    A boundary voxel that travels straight out of WM crosses the ribbon; one that
    slides ALONG the interface does not, and contributes its path length to the
    thickness of wherever it ends up instead. Returns (tangential_mm, normal_mm,
    tangential_fraction).
    """
    n = wm_normals(seg, traj[0])
    d = (traj[-1] - traj[0]) * zooms
    along = np.einsum('ij,ij->i', d, n)
    tang = d - along[:, None] * n
    tmag = np.linalg.norm(tang, axis=1)
    dmag = np.linalg.norm(d, axis=1)
    return tmag, along, tmag / np.maximum(dmag, 1e-9)


def most_tangential_location(seg, traj, tmag, radius=7, min_count=60):
    """Voxel index where the local mean tangential travel is largest.

    Averaged over a box of half-width `radius` around each voxel, and only where
    at least `min_count` boundary voxels fall in that box -- otherwise the
    winner is a single stray seed with a large sideways step.
    """
    shape = seg.shape
    idx = np.rint(traj[0]).astype(int)
    ok = ((idx >= 0) & (idx < np.array(shape))).all(axis=1)
    flat = np.ravel_multi_index(idx[ok].T, shape)
    tot = np.zeros(shape, float).reshape(-1)
    cnt = np.zeros(shape, float).reshape(-1)
    np.add.at(tot, flat, tmag[ok])
    np.add.at(cnt, flat, 1.0)
    size = 2 * radius + 1
    tot = uniform_filter(tot.reshape(shape), size) * size ** 3
    cnt = uniform_filter(cnt.reshape(shape), size) * size ** 3
    mean = np.where(cnt >= min_count, tot / np.maximum(cnt, 1e-9), -1.0)
    i = int(np.argmax(mean))
    c = np.unravel_index(i, shape)
    return np.array(c), float(mean[c]), float(cnt[c])


def sample_labels(seg, pos):
    """Segmentation label at each position (nearest voxel, out-of-bounds -> 0)."""
    idx = np.rint(pos).astype(int)
    ok = ((idx >= 0) & (idx < np.array(seg.shape))).all(axis=1)
    out = np.zeros(len(pos), np.uint8)
    out[ok] = seg[tuple(idx[ok].T)]
    return out


def sweep_counts(traj, shape):
    """How many times a transported point lands in each voxel, over every step.

    This is the geometry behind DiReCT's `hit_image`: the solver accumulates the
    warped contour indicator at GM voxels once per integration point, and divides
    the accumulated distance by it. A GM voxel that no path sweeps gets no hit and
    so takes no thickness from this transport, which is what the figure shows.
    """
    counts = np.zeros(shape, np.int32)
    idx = np.rint(traj).astype(int)
    ok = ((idx >= 0) & (idx < np.array(shape))).all(axis=2)
    flat = np.ravel_multi_index(idx[ok].T, shape)
    np.add.at(counts.reshape(-1), flat, 1)
    return counts


# ---------------------------------------------------------------------------
# Locating a sulcus
# ---------------------------------------------------------------------------

def _ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (z * z + y * y + x * x) <= r * r


def deepest_sulcus(seg, close_radius=6, gm_within=2, exclude_midline=10):
    """Voxel index of the deepest CORTICAL sulcus.

    Sulcal CSF is continuous with the subarachnoid space, so hole-filling does
    not isolate it; a morphological closing of the brain mask bridges the sulcal
    openings instead, and CSF-labelled voxels inside that hull are candidates.

    Two further conditions are needed, and without them this picks the wrong
    thing. "CSF inside the hull" also matches the VENTRICLES, which are label 0
    and sit deeper than any sulcus, so candidates must have cortex within
    `gm_within` voxels -- ventricles are walled by WM and drop out. And the
    deepest remaining pocket is then the INTERHEMISPHERIC FISSURE, which is not
    a sulcus and whose banks are the medial wall the pipeline pins, so a slab of
    `exclude_midline` voxels either side of the brain's mid-sagittal plane is
    dropped too. Set exclude_midline=0 to keep it.

    Returns (index, depth_mm, hull).
    """
    brain = seg > 0
    hull = binary_closing(brain, structure=_ball(close_radius))
    depth = distance_transform_edt(hull)
    sulcal = hull & (seg == 0) & binary_dilation(seg == 2, structure=_ball(gm_within))
    if exclude_midline:
        # axis 0 is L in this LIA grid, so the mid-sagittal plane is a slab in it
        mid = float(np.argwhere(brain)[:, 0].mean())
        off = np.abs(np.arange(seg.shape[0]) - mid) > exclude_midline
        sulcal &= off[:, None, None]
    if not sulcal.any():
        raise SystemExit('no cortical sulcal CSF found')
    cand = np.argwhere(sulcal)
    d = depth[tuple(cand.T)]
    i = int(np.argmax(d))
    return cand[i], float(d[i]), hull


def narrowest_sulcus(seg, close_radius=6, gm_within=2, exclude_midline=10,
                     depth_pct=75):
    """Voxel index of the narrowest CSF gap among the deeper cortical sulci.

    Same candidate set as deepest_sulcus (CSF inside the closed brain hull, with
    cortex within `gm_within`, off the midline). Ranking differs: the local
    half-width of the CSF band, `distance_transform_edt(seg == 0)`, is minimised
    instead of the depth being maximised -- so this picks the folds whose banks
    nearly touch. Only candidates in the top `depth_pct` of depth are eligible,
    otherwise the winner is a shallow nick at the brain surface rather than a
    sulcus. Ties on width (there are many at one voxel) go to the deeper one.

    Returns (index, depth_mm, width_mm).
    """
    brain = seg > 0
    hull = binary_closing(brain, structure=_ball(close_radius))
    depth = distance_transform_edt(hull)
    width = distance_transform_edt(seg == 0)
    sulcal = hull & (seg == 0) & binary_dilation(seg == 2, structure=_ball(gm_within))
    if exclude_midline:
        mid = float(np.argwhere(brain)[:, 0].mean())
        off = np.abs(np.arange(seg.shape[0]) - mid) > exclude_midline
        sulcal &= off[:, None, None]
    if not sulcal.any():
        raise SystemExit('no cortical sulcal CSF found')
    cand = np.argwhere(sulcal)
    d = depth[tuple(cand.T)]
    w = width[tuple(cand.T)]
    keep = d >= np.percentile(d, depth_pct)
    cand, d, w = cand[keep], d[keep], w[keep]
    i = int(np.lexsort((-d, w))[0])          # narrowest, deepest to break ties
    return cand[i], float(d[i]), float(w[i])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
#
# The grid is LIA: axis 0 = L, axis 1 = I, axis 2 = A. A coronal slice fixes
# axis 2. In plane we draw axis 0 across (subject's left to the image right,
# neurological convention) and axis 1 down (increasing = inferior), which is what
# imshow(vol[:, :, s].T) gives with the default origin='upper'.

def coronal(vol, s):
    return vol[:, :, s].T


def draw_background(ax, anat, seg, s, extent=None):
    if anat is not None:
        bg = coronal(anat, s)
        lo, hi = np.percentile(bg[bg > 0], [1, 99]) if (bg > 0).any() else (0, 1)
        ax.imshow(bg, cmap='gray', vmin=lo, vmax=hi, interpolation='bilinear')
    else:
        ax.imshow(coronal((seg > 0).astype(float), s), cmap='gray')
    # WM boundary in cyan, GM outer boundary in orange
    ax.contour(coronal((seg == 3).astype(float), s), levels=[0.5],
               colors='#22d3ee', linewidths=0.6)
    ax.contour(coronal((seg >= 2).astype(float), s), levels=[0.5],
               colors='#fb923c', linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([])


def draw_paths(ax, traj, values, norm, cmap='plasma', lw=0.5, alpha=0.85):
    """Trajectories projected into the coronal plane, coloured by `values`."""
    segs = np.stack([traj[:, :, 0], traj[:, :, 1]], axis=2)      # (T, N, 2)
    segs = np.transpose(segs, (1, 0, 2))                          # (N, T, 2)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, alpha=alpha)
    lc.set_array(np.asarray(values))
    ax.add_collection(lc)
    return lc


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_overview(out, traj, arc, seg, anat, vel, s, slab_sel, zooms, sweep):
    """Whole coronal slice: the field, the paths it carries the contour along,
    and where those paths sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    norm = Normalize(0, np.percentile(arc, 99))

    ax = axes[0]
    draw_background(ax, anat, seg, s)
    lc = draw_paths(ax, traj[:, slab_sel], arc[slab_sel], norm, lw=0.45)
    ax.plot(traj[0, slab_sel, 0], traj[0, slab_sel, 1], '.', ms=0.7,
            color='#ffffff', alpha=0.7)
    ax.set_title('WM/GM boundary voxels, %d steps of the velocity field\n'
                 'coronal slice %d, %d seeds in the slab; dots = start'
                 % (len(traj) - 1, s, slab_sel.sum()), fontsize=10)
    cb = fig.colorbar(lc, ax=ax, fraction=0.045)
    cb.set_label('path length (mm)')

    ax = axes[1]
    draw_background(ax, anat, seg, s)
    # The field itself, as the direction a point travels (-v), on a coarse lattice.
    step = 3
    d, h = np.meshgrid(np.arange(0, seg.shape[0], step),
                       np.arange(0, seg.shape[1], step), indexing='ij')
    u = -vel[d, h, s, 0]
    w = -vel[d, h, s, 1]
    mag = np.hypot(u, w)
    keep = mag > 1e-4
    q = ax.quiver(d[keep], h[keep], u[keep], w[keep], mag[keep],
                  cmap='viridis', scale=12, width=0.0016,
                  headwidth=4, headlength=5, alpha=0.9)
    ax.set_title('the per-integration-point field it rides (-v), every %d voxels\n'
                 'one step; the solve composes it %d times'
                 % (step, len(traj) - 1), fontsize=10)
    cb = fig.colorbar(q, ax=ax, fraction=0.045)
    cb.set_label('|v| per step (mm)')

    ax = axes[2]
    sw = coronal(sweep, s).astype(float)
    gm = coronal((seg == 2).astype(float), s)
    im = ax.imshow(np.ma.masked_where(sw == 0, sw), cmap='inferno',
                   vmin=1, vmax=max(2, np.percentile(sw[sw > 0], 99)),
                   interpolation='nearest')
    # GM voxels no path ever touches take no thickness from this transport.
    missed = (gm > 0) & (sw == 0)
    ax.imshow(np.ma.masked_where(~missed, missed.astype(float)),
              cmap=matplotlib.colors.ListedColormap(['#22d3ee']),
              interpolation='nearest', alpha=0.9)
    ax.contour(coronal((seg == 3).astype(float), s), levels=[0.5],
               colors='#ffffff', linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('where the transported points land (all seeds, all steps)\n'
                 'cyan = GM voxels no path sweeps: %d of %d in this slice (%.1f%%)'
                 % (missed.sum(), int((gm > 0).sum()),
                    100.0 * missed.sum() / max(1, int((gm > 0).sum()))), fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.045)
    cb.set_label('trajectory samples per voxel')

    fig.suptitle('Where the boundary voxels go — bert, coronal slice %d '
                 '(subject left on the image right)' % s, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_zooms(out, traj, arc, seg, anat, s, slab_sel, centres, half=22):
    """Close-ups where individual paths are legible."""
    n = len(centres)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.4))
    axes = np.atleast_1d(axes)
    norm = Normalize(0, np.percentile(arc, 99))

    idx = np.flatnonzero(slab_sel)
    for ax, (cd, ch, label) in zip(axes, centres):
        draw_background(ax, anat, seg, s)
        start = traj[0, idx]
        near = ((np.abs(start[:, 0] - cd) <= half) &
                (np.abs(start[:, 1] - ch) <= half))
        sub = idx[near]
        # Markers UNDER the lines and small: at this zoom a 2pt marker is wider
        # than a sub-0.5mm path, so drawing them on top erases exactly the
        # short-travel population the picture is supposed to show.
        ax.plot(traj[0, sub, 0], traj[0, sub, 1], '.', ms=1.4, color='w',
                zorder=2, alpha=0.9)
        ax.plot(traj[-1, sub, 0], traj[-1, sub, 1], '.', ms=1.4, color='#ef4444',
                zorder=2, alpha=0.9)
        lc = draw_paths(ax, traj[:, sub], arc[sub], norm, lw=0.9, alpha=0.95)
        lc.set_zorder(3)
        ax.set_xlim(cd - half, cd + half)
        ax.set_ylim(ch + half, ch - half)
        short = int((arc[sub] < 0.5).sum())
        ax.set_title('%s\n%d paths, %d under 0.5 mm\nwhite = start, red = after %d steps'
                     % (label, len(sub), short, len(traj) - 1), fontsize=9)
    cb = fig.colorbar(lc, ax=axes.tolist(), fraction=0.03)
    cb.set_label('path length (mm)')
    fig.suptitle('Individual boundary-voxel paths, coronal slice %d' % s, fontsize=12)
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)


def figure_diagnostics(out, traj, arc, net, end_lab, seg, thick, zooms, steps):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(arc, bins=120, range=(0, np.percentile(arc, 99.5)),
            color='#6366f1', alpha=0.85)
    ax.axvline(np.median(arc), color='#ef4444', lw=1.2,
               label='median %.3f mm' % np.median(arc))
    if thick is not None:
        gm = thick[(seg == 2) & (thick > 0)]
        ax.axvline(np.median(gm), color='#22c55e', lw=1.2, ls='--',
                   label="solve's thickmap median %.3f mm" % np.median(gm))
    ax.set_xlabel('path length over %d steps (mm)' % steps)
    ax.set_ylabel('boundary voxels')
    ax.set_title('How far a boundary voxel travels')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    hb = ax.hexbin(arc, net, gridsize=90, bins='log', cmap='magma',
                   extent=(0, np.percentile(arc, 99.5), 0, np.percentile(arc, 99.5)))
    lim = np.percentile(arc, 99.5)
    ax.plot([0, lim], [0, lim], color='#22d3ee', lw=1, ls='--', label='straight path')
    ax.set_xlabel('arc length (mm)'); ax.set_ylabel('net displacement (mm)')
    ax.set_title('Path length vs straight-line displacement')
    ax.legend(fontsize=8)
    fig.colorbar(hb, ax=ax, fraction=0.045, label='count')

    ax = axes[1, 0]
    d = np.linalg.norm(np.diff(traj, axis=0) * zooms, axis=2)     # (steps, N)
    med = np.median(d, axis=1)
    q1, q3 = np.percentile(d, [25, 75], axis=1)
    x = np.arange(1, steps + 1)
    ax.plot(x, med, '-o', color='#6366f1', ms=4, label='median')
    ax.fill_between(x, q1, q3, color='#6366f1', alpha=0.25, label='IQR')
    ax.set_xlabel('integration step'); ax.set_ylabel('displacement this step (mm)')
    ax.set_title('Step size along the integration')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    names = {0: 'background / CSF', 2: 'grey matter', 3: 'white matter'}
    labs, counts = np.unique(end_lab, return_counts=True)
    frac = 100.0 * counts / counts.sum()
    ax.bar([names.get(int(l), str(l)) for l in labs], frac,
           color=['#94a3b8', '#fb923c', '#22d3ee'][:len(labs)])
    for i, (f, c) in enumerate(zip(frac, counts)):
        ax.text(i, f + 0.8, '%.1f%%\n(%d)' % (f, c), ha='center', fontsize=8)
    ax.set_ylabel('% of boundary voxels')
    ax.set_title('Tissue the voxel ends up in after %d steps' % steps)
    ax.set_ylim(0, max(frac) * 1.25)

    fig.suptitle('Boundary-voxel transport, whole brain (%d seeds)' % traj.shape[1],
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_sulcus(out, traj, arc, seg, anat, centre, half, zooms, slab,
                  traces_only=False, title=None, vmax=None, select='start',
                  values=None, value_label='path length (mm)', window=(1, 99)):
    """Every border WM voxel in one sulcus, on the native slice.

    No reslicing: the background is the acquired coronal plane and the paths are
    the 3D trajectories projected into it, so the out-of-plane component is
    reported rather than removed.
    """
    cd, ch, s = int(centre[0]), int(centre[1]), int(centre[2])
    seeds = traj[0]
    # Which end of the path decides membership. select='end' answers "which
    # boundary voxels DELIVER thickness into this slice", and pulls in paths
    # that started elsewhere; select='start' answers "where do this slice's
    # boundary voxels go", and lets them leave. They are different questions and
    # the two sets are not the same size.
    ref = traj[0] if select == 'start' else traj[-1]
    sel = ((np.abs(ref[:, 2] - s) <= slab) &
           (np.abs(ref[:, 0] - cd) <= half) &
           (np.abs(ref[:, 1] - ch) <= half))
    idx = np.flatnonzero(sel)

    panels = (True,) if traces_only else (False, True)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(8.0 * len(panels), 8))
    axes = np.atleast_1d(axes)
    # A shared scale makes two fields comparable; without it each panel
    # normalises to its own 99th percentile and the colours mean different mm.
    val = arc if values is None else values
    norm = Normalize(0, np.percentile(val, 99) if vmax is None else vmax)

    for ax, show_paths in zip(axes, panels):
        bg = coronal(anat, s) if anat is not None else coronal((seg > 0).astype(float), s)
        # Window over the WHOLE volume's in-brain intensities, not this slice's,
        # so the greyscale means the same thing in every figure. Raising the
        # black point above GM's low tail darkens grey matter while leaving WM
        # bright; the default 1-99 puts GM at about 65% brightness here.
        inb = anat[anat > 0] if anat is not None else bg[bg > 0]
        lo, hi = np.percentile(inb, window) if inb.size else (0.0, 1.0)
        ax.imshow(bg, cmap='gray', vmin=lo, vmax=hi, interpolation='bilinear')
        ax.contour(coronal((seg == 3).astype(float), s), levels=[0.5],
                   colors='#22d3ee', linewidths=0.8)
        ax.contour(coronal((seg >= 2).astype(float), s), levels=[0.5],
                   colors='#fb923c', linewidths=0.8)
        if show_paths:
            lc = draw_paths(ax, traj[:, idx], val[idx], norm, lw=1.0, alpha=0.95)
            lc.set_zorder(4)
            # Every integration point, not just the endpoints: the ten steps are
            # what the transport actually is, and their spacing along a path is
            # the thing to read off it.
            P = traj[:, idx]                                  # (steps+1, n, 3)
            ax.scatter(P[:, :, 0].ravel(), P[:, :, 1].ravel(),
                       c=np.tile(val[idx], P.shape[0]), cmap='plasma', norm=norm,
                       s=3.0, linewidths=0, zorder=5)
            ax.plot(seeds[idx, 0], seeds[idx, 1], '.', ms=2.6, color='w', zorder=6)
            ax.plot(traj[-1, idx, 0], traj[-1, idx, 1], '.', ms=2.6,
                    color='#ef4444', zorder=6)
        ax.plot(cd, ch, 'x', color='#a855f7', ms=11, mew=2.2, zorder=6)
        ax.set_xlim(cd - half, cd + half)
        ax.set_ylim(ch + half, ch - half)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(('%d border WM voxels %s this slice, %d steps each '
                      '(one dot per step)\nwhite = start, red = end'
                      % (len(idx), 'ending in' if select == 'end'
                         else 'starting in', len(traj) - 1))
                     if show_paths else
                     'the sulcus, coronal slice %d\n'
                     'cyan = WM boundary, orange = GM/CSF boundary' % s, fontsize=10)

    drift = np.abs(traj[-1, idx, 2] - traj[0, idx, 2])
    cb = fig.colorbar(lc, ax=axes.tolist(), fraction=0.03)
    cb.set_label(value_label)
    fig.suptitle('%s — out-of-plane drift median %.2f, p95 %.2f voxels'
                 % (title or 'Border WM voxels of one sulcus, native slice '
                              '(no reslicing)',
                    np.median(drift), np.percentile(drift, 95)), fontsize=11)
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return idx, drift


# ---------------------------------------------------------------------------

def pick_zoom_centres(traj, arc, seg, s, slab_sel):
    """Three in-plane centres: the longest paths, the shortest, and a sulcal fundus.

    'Sulcal fundus' is taken as the in-slab seed whose starting voxel is deepest
    inside WM relative to the outer GM boundary -- i.e. the bottom of a fold.
    """
    idx = np.flatnonzero(slab_sel)
    start = traj[0, idx]
    a = arc[idx]

    long_i = idx[np.argsort(a)[-max(1, len(a) // 200):]]
    short_i = idx[np.argsort(a)[:max(1, len(a) // 200)]]

    dist_out = distance_transform_edt(seg[:, :, s] >= 2)
    ij = np.rint(start[:, :2]).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, seg.shape[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, seg.shape[1] - 1)
    depth = dist_out[ij[:, 0], ij[:, 1]]

    return [
        (float(np.median(traj[0, long_i, 0])), float(np.median(traj[0, long_i, 1])),
         'longest paths in this slice'),
        (float(start[np.argmax(depth), 0]), float(start[np.argmax(depth), 1]),
         'deepest fold in this slice'),
        (float(np.median(traj[0, short_i, 0])), float(np.median(traj[0, short_i, 1])),
         'shortest paths in this slice'),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True,
                   help='a dl+direct-nofs output directory (needs seg.nii.gz and '
                        'field_pial/best_Velocity.nii.gz)')
    p.add_argument('--velocity', default=None,
                   help='velocity field to use instead of field_pial/best_Velocity.nii.gz')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--steps', type=int, default=10,
                   help='integration steps (default 10, matching the solve\'s '
                        'num_integration_points and the propagation\'s rounds)')
    p.add_argument('--slice', type=int, default=None,
                   help='coronal slice (axis 2). Default: the slice carrying the '
                        'most boundary voxels.')
    p.add_argument('--slab', type=float, default=0.5,
                   help='half-thickness in voxels of the slab of seeds drawn on the '
                        'slice figures (default 0.5, i.e. that slice only)')
    p.add_argument('--sulcus-at', default=None, metavar='D,H,S',
                   help='voxel index to centre the sulcus figure on '
                        '(default: the deepest sulcal CSF voxel in the volume)')
    p.add_argument('--sulcus-mode', choices=('deep', 'narrow'), default='deep',
                   help="how to auto-pick the sulcus: 'deep' = the most buried "
                        "fold, 'narrow' = the thinnest CSF gap among the deeper "
                        "folds (default deep). Ignored with --sulcus-at.")
    p.add_argument('--window', default='1,99', metavar='LO,HI',
                   help='greyscale window as two percentiles of the in-brain '
                        'intensities (default 1,99). Raise LO to darken grey '
                        'matter: GM sits near the 28th percentile on this data.')
    p.add_argument('--colour-by', choices=('length', 'tangential', 'tangential-frac'),
                   default='length',
                   help="colour the traces by total path length (default), by "
                        "the component of net displacement PERPENDICULAR to the "
                        "local WM normal in mm, or by that as a fraction of the "
                        "total displacement")
    p.add_argument('--find-tangential', action='store_true',
                   help='ignore --sulcus-at and centre the sulcus figure where '
                        'the local mean tangential travel is largest')
    p.add_argument('--select-by', choices=('start', 'end'), default='start',
                   help="select paths by where they START in the slice (default) "
                        "or by where they END there. 'end' shows every boundary "
                        "voxel delivering thickness into the slice, including "
                        "those that started outside it.")
    p.add_argument('--traces-only', action='store_true',
                   help='drop the plain-anatomy panel and plot only the traces')
    p.add_argument('--sulcus-title', default=None,
                   help='title for the sulcus figure')
    p.add_argument('--sulcus-vmax', type=float, default=None,
                   help='fix the path-length colour scale (mm) so two runs are '
                        'directly comparable')
    p.add_argument('--include-midline', action='store_true',
                   help='allow the interhemispheric fissure to be chosen as the '
                        '"sulcus" (excluded by default; it is not one, and its '
                        'banks are the pinned medial wall)')
    p.add_argument('--exclude-midline', type=float, default=10, metavar='N',
                   help='drop candidates within N voxels of the brain\'s '
                        'mid-sagittal plane (default 10). The default only '
                        'removes the fissure itself; raise it to force a '
                        'laterally placed sulcus.')
    p.add_argument('--sulcus-half', type=float, default=26,
                   help='half-width in voxels of the sulcus figure (default 26)')
    p.add_argument('--sulcus-slab', type=float, default=0.5,
                   help='half-thickness in voxels of the seed slab (default 0.5)')
    p.add_argument('--zoom-half', type=float, default=22,
                   help='half-width in voxels of the close-up boxes (default 22)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seg, vel, anat, thick, zooms = load_run(args.run_dir, args.velocity)
    if args.velocity and os.path.realpath(args.velocity).startswith(
            os.path.realpath(args.run_dir) + os.sep) is False:
        # T1w_thickmap.nii.gz in --run-dir is the output of THAT directory's own
        # solve. With a velocity field from elsewhere it describes a different
        # solve entirely, so reporting it as "the solve's own" would compare two
        # unrelated runs. Drop it rather than mislabel it.
        if thick is not None:
            print('note: --velocity comes from outside --run-dir, so the '
                  'thickness map in --run-dir belongs to a different solve; '
                  'it is not reported.')
        thick = None
    contour = wm_contour_mask(seg)
    seeds = np.stack(np.nonzero(contour), axis=1).astype(float)
    print('grid %s at %s mm; %d WM/GM boundary voxels'
          % (seg.shape, tuple(zooms), len(seeds)))

    traj = integrate(seeds, vel, steps=args.steps)
    arc, net = path_lengths(traj, zooms)
    end_lab = sample_labels(seg, traj[-1])

    print('path length over %d steps: mean %.3f  median %.3f  p95 %.3f mm'
          % (args.steps, arc.mean(), np.median(arc), np.percentile(arc, 95)))
    print('net displacement:          mean %.3f  median %.3f  p95 %.3f mm'
          % (net.mean(), np.median(net), np.percentile(net, 95)))
    for lab, name in ((0, 'background/CSF'), (2, 'GM'), (3, 'WM')):
        n = int((end_lab == lab).sum())
        print('  ends in %-15s %7d  (%5.1f%%)' % (name, n, 100.0 * n / len(end_lab)))
    if thick is not None:
        gm = thick[(seg == 2) & (thick > 0)]
        print("solve's own thickmap over GM: mean %.3f  median %.3f mm"
              % (gm.mean(), np.median(gm)))

    s = args.slice
    if s is None:
        s = int(np.argmax(contour.sum(axis=(0, 1))))
    slab_sel = np.abs(seeds[:, 2] - s) <= args.slab
    print('coronal slice %d: %d seeds in the slab' % (s, slab_sel.sum()))
    drift = np.abs(traj[-1, slab_sel, 2] - traj[0, slab_sel, 2])
    print('  out-of-plane drift of those paths: median %.2f, p95 %.2f voxels'
          % (np.median(drift), np.percentile(drift, 95)))

    sweep = sweep_counts(traj, seg.shape)
    gm_missed = int(((seg == 2) & (sweep == 0)).sum())
    print('GM voxels no path ever sweeps: %d of %d (%.1f%%)'
          % (gm_missed, int((seg == 2).sum()),
             100.0 * gm_missed / max(1, int((seg == 2).sum()))))

    f1 = os.path.join(args.out_dir, 'travel_overview.png')
    figure_overview(f1, traj, arc, seg, anat, vel, s, slab_sel, zooms, sweep)
    f2 = os.path.join(args.out_dir, 'travel_zooms.png')
    figure_zooms(f2, traj, arc, seg, anat, s, slab_sel,
                 pick_zoom_centres(traj, arc, seg, s, slab_sel), half=args.zoom_half)
    tmag, along, tfrac = tangential_travel(seg, traj, zooms)
    print('tangential travel: mean %.3f  median %.3f  p95 %.3f mm '
          '(fraction of displacement: median %.3f, p95 %.3f)'
          % (tmag.mean(), np.median(tmag), np.percentile(tmag, 95),
             np.median(tfrac), np.percentile(tfrac, 95)))
    values, vlabel = arc, 'path length (mm)'
    if args.colour_by == 'tangential':
        values, vlabel = tmag, 'tangential travel (mm)'
    elif args.colour_by == 'tangential-frac':
        values, vlabel = tfrac, 'tangential fraction of displacement'

    if args.find_tangential:
        centre, mval, ncnt = most_tangential_location(seg, traj, tmag)
        print('most tangential neighbourhood: %s, local mean %.3f mm over %d '
              'boundary voxels' % (centre.tolist(), mval, int(ncnt)))
    elif args.sulcus_at:
        centre = np.array([int(x) for x in args.sulcus_at.split(',')])
        print('sulcus centre (given): %s' % centre.tolist())
    else:
        mid = 0 if args.include_midline else args.exclude_midline
        if args.sulcus_mode == 'narrow':
            centre, depth, width = narrowest_sulcus(seg, exclude_midline=mid)
            print('narrowest cortical sulcus: %s, CSF half-width %.2f mm, '
                  '%.2f mm inside the brain hull'
                  % (centre.tolist(), width, depth))
        else:
            centre, depth, _ = deepest_sulcus(seg, exclude_midline=mid)
            print('deepest sulcal CSF voxel: %s, %.2f mm inside the brain hull'
                  % (centre.tolist(), depth))
    _mid = float(np.argwhere(seg > 0)[:, 0].mean())
    print('sulcus is %.1f voxels off the mid-sagittal plane (centroid %.1f)'
          % (abs(centre[0] - _mid), _mid))

    f4 = os.path.join(args.out_dir, 'travel_sulcus.png')
    sidx, sdrift = figure_sulcus(f4, traj, arc, seg, anat, centre,
                                 args.sulcus_half, zooms, args.sulcus_slab,
                                 traces_only=args.traces_only,
                                 title=args.sulcus_title, vmax=args.sulcus_vmax,
                                 select=args.select_by,
                                 values=values, value_label=vlabel,
                                 window=tuple(float(x) for x in args.window.split(',')))
    print('sulcus figure: %d border WM voxels (%s in the slab); path length '
          'median %.3f mm, p95 %.3f mm; out-of-plane drift median %.2f voxels'
          % (len(sidx), 'ending' if args.select_by == 'end' else 'starting',
             np.median(arc[sidx]), np.percentile(arc[sidx], 95),
             np.median(sdrift)))
    if args.sulcus_slab > 0.5 and len(sidx):
        # With a slab wider than one slice, say how the paths distribute across
        # it -- otherwise "in this slice" silently means "within N of it".
        _ref = traj[0] if args.select_by == 'start' else traj[-1]
        _off = np.rint(_ref[sidx, 2] - int(centre[2])).astype(int)
        for o in np.unique(_off):
            m = _off == o
            print('    slice %+d: %4d voxels, path length median %.3f mm'
                  % (o, int(m.sum()), np.median(arc[sidx][m])))

    f3 = os.path.join(args.out_dir, 'travel_diagnostics.png')
    figure_diagnostics(f3, traj, arc, net, end_lab, seg, thick, zooms, args.steps)
    for f in (f1, f2, f3, f4):
        print('wrote', f)


if __name__ == '__main__':
    main()
