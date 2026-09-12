#!/usr/bin/env python
"""Solve the DiReCT velocity field and propagate a surface along it, in one call.

This is the two stages of pial_clean.py behind a single entry point, with the
surface an optional in-memory input and the propagation available on either
device:

  propagate_on='cpu'    numpy/scipy, float64, the reference implementation
  propagate_on='cuda'   torch, the field STAYS in GPU memory from the solve
                        through the propagation -- it is never copied to the
                        host unless the caller asks for it

Equivalence of the two paths
----------------------------
The CPU round is

    cur = build_constrained_white(cur + step, ..., floor=-inf, iters=k)

and with floor=-inf the signed-distance constraint can never fire, so the round
is a plain Taubin relaxation; the sdt and its gradient are computed and unused.
Two further facts let the GPU path drop every per-round affine:

  * totkr is affine, so with A the vox2ras_tkr linear block,
        step = (totkr(pos - v) - totkr(pos)) * S = A @ (-v) * S
    and applying it to a tkrRAS point is exactly `pos -= v * S` in voxel space.
  * the pin blend w*start + (1-w)*cur is an affine combination (the weights sum
    to one), so it commutes with the affine.

The GPU path therefore runs the whole loop in voxel coordinates and converts
once at the end. It is the same arithmetic, not an approximation; the residual
difference against the CPU path is float32-vs-float64 rounding. Run this module
with --check to measure it.

Sampling matches too: scipy's map_coordinates(order=1, mode='nearest') is
trilinear with the edge value held outside, which is grid_sample's
mode='bilinear', padding_mode='border', align_corners=True.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from . import pial_clean as pc
from . import wm_surface
from .field_pial_prototype import (_pin_weights, build_no_push_mask,
                                   build_pin_mask, evaluate_surface, get_vox2ras_tkr)
from .surface_frames import volume_info_from_image


# ---------------------------------------------------------------------------
# the CUDA propagation
# ---------------------------------------------------------------------------
def adjacency_from_faces(n_vertices, faces):
    """(W, deg): the vertex neighbour-indicator matrix straight from the faces.

    Identical structure and degrees to the trimesh-based builder in
    field_pial_prototype, but that walks vertex_neighbors in Python, which
    measured 1.30s against 0.06s here on a 134k-vertex hemisphere -- roughly 90%
    of a GPU propagation. Only the matrix is needed on this path, not the mesh.
    """
    F = np.asarray(faces)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])                 # undirected
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                      shape=(n_vertices, n_vertices)).tocsr()
    W.sum_duplicates()
    W.data[:] = 1.0                                     # indicator, not multiplicity
    deg = np.asarray(W.sum(1)).ravel()
    deg[deg == 0] = 1
    return W, deg


def _sparse_adjacency(Wm, device, dtype):
    """scipy CSR neighbour-indicator matrix -> torch sparse CSR on `device`."""
    Wm = Wm.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(Wm.indptr.astype(np.int64)).to(device),
        torch.from_numpy(Wm.indices.astype(np.int64)).to(device),
        torch.from_numpy(Wm.data.astype(np.float64)).to(device=device, dtype=dtype),
        size=Wm.shape)


def _sample_trilinear(field, pos):
    """field [1, 3, D, H, W], pos [N, 3] in (d, h, w) voxel indices -> [N, 3].

    Equivalent to map_coordinates(order=1, mode='nearest') per component.
    """
    D, H, W = field.shape[2:]
    size = pos.new_tensor([D, H, W])
    # align_corners=True: index i maps to 2i/(n-1) - 1. grid's last axis is
    # (x, y, z) = (w, h, d), the REVERSE of the index order.
    g = 2.0 * pos / (size - 1).clamp(min=1) - 1.0
    grid = g.flip(-1).reshape(1, -1, 1, 1, 3)
    out = F.grid_sample(field, grid, mode='bilinear', padding_mode='border',
                        align_corners=True)
    return out[0, :, :, 0, 0].transpose(0, 1)


def propagate_pial_torch(white_verts, faces, velocity, tovox_affine, totkr_affine,
                         pin_mask=None, device=None, dtype=torch.float32,
                         rounds=pc.ROUNDS, step_scale=pc.STEP_SCALE,
                         relax_iters=pc.RELAX_ITERS,
                         relax_iters_final=pc.RELAX_ITERS_FINAL,
                         relax_lambda=pc.RELAX_LAMBDA, pin_feather=pc.PIN_FEATHER):
    """Carry a surface along the velocity field, entirely on the GPU.

    `velocity` is either the [1, 3, D, H, W] tensor solve_velocity_field_t
    returns (used in place, nothing is copied) or a [D, H, W, 3] array, which is
    uploaded once. `tovox_affine` / `totkr_affine` are the 4x4 matrices, not the
    callables -- the loop needs the linear block, not a host round-trip.

    Returns the propagated vertices as a float64 numpy array in tkrRAS.
    """
    if device is None:
        device = velocity.device if torch.is_tensor(velocity) else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    if torch.is_tensor(velocity):
        field = velocity.to(device=device, dtype=dtype)
    else:
        field = torch.from_numpy(np.ascontiguousarray(velocity)).to(device=device, dtype=dtype)
        field = field.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    if field.ndim != 5 or field.shape[1] != 3:
        raise ValueError('velocity must be [1, 3, D, H, W] or [D, H, W, 3], got %s'
                         % (tuple(field.shape),))

    Wm, deg = adjacency_from_faces(len(np.asarray(white_verts)), faces)
    Wt = _sparse_adjacency(Wm, device, dtype)
    degt = torch.from_numpy(deg).to(device=device, dtype=dtype).unsqueeze(1)

    Ainv = torch.from_numpy(np.asarray(tovox_affine, np.float64)).to(device=device, dtype=dtype)
    A = torch.from_numpy(np.asarray(totkr_affine, np.float64)).to(device=device, dtype=dtype)
    v_tkr = torch.from_numpy(np.asarray(white_verts, np.float64)).to(device=device, dtype=dtype)
    pos = v_tkr @ Ainv[:3, :3].T + Ainv[:3, 3]          # voxel indices
    start = pos.clone()

    pin_w = None
    if pin_mask is not None and np.asarray(pin_mask).any():
        pin_w = torch.from_numpy(_pin_weights(np.asarray(pin_mask), Wm, pin_feather)) \
            .to(device=device, dtype=dtype).unsqueeze(1)

    for rnd in range(rounds):
        pos = pos - _sample_trilinear(field, pos) * step_scale
        iters = relax_iters if rnd < rounds - 1 else relax_iters_final
        for i in range(iters):
            neighbour_mean = torch.sparse.mm(Wt, pos) / degt
            mu = -0.53 if i % 2 else relax_lambda
            pos = pos + mu * (neighbour_mean - pos)
        if pin_w is not None:
            pos = pin_w * start + (1.0 - pin_w) * pos

    out = pos @ A[:3, :3].T + A[:3, 3]
    return out.double().cpu().numpy()


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def propagate(white_verts, faces, velocity, seg, tovox, totkr, ref_img=None,
              pin_mask=None, on='cuda', device=None, dtype=torch.float32):
    """Propagate on 'cpu' (numpy reference) or 'cuda' (field resident on GPU).

    On the CPU path `velocity` must be, or be convertible to, the [D, H, W, 3]
    array; a GPU tensor is brought across once.
    """
    on = str(on).lower()
    if on not in ('cpu', 'cuda'):
        raise ValueError("propagate_on must be 'cpu' or 'cuda', got %r" % (on,))
    if on == 'cuda' and not torch.cuda.is_available() and device is None:
        print('WARNING: propagate_on="cuda" but no CUDA device; running torch on CPU',
              file=sys.stderr)
    if on == 'cpu':
        if torch.is_tensor(velocity):
            velocity = pc.velocity_to_numpy(velocity)
        return pc.propagate_pial(white_verts, faces, velocity, seg, tovox, totkr,
                                 pin_mask=pin_mask)
    if ref_img is None:
        raise ValueError('the cuda path needs ref_img for the vox2ras_tkr affine')
    A = get_vox2ras_tkr(ref_img)
    return propagate_pial_torch(white_verts, faces, velocity, np.linalg.inv(A), A,
                                pin_mask=pin_mask, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# solve + propagate
# ---------------------------------------------------------------------------
def reconstruct(prep_dir, surf_dir=None, surfaces=None, hemis=('lh', 'rh'),
                propagate_on='cuda', velocity=None, pin=True, out_dir=None,
                verbose=True, report=None, compute_thickness=True,
                build_white=None, nsmooth=wm_surface.NSMOOTH_DEFAULT,
                dtype=torch.float32, device=None):
    """Solve the field and propagate, returning the propagated surfaces.

    prep_dir        a --space cropped prep (seg_<Label>.nii.gz, softmax_seg.nii.gz,
                    label_def.csv)
    surf_dir        directory holding ?h.white; ignored if `surfaces` is given
    surfaces        optional {hemi: (verts, faces)} already in the cropped tkrRAS
                    frame. Both hemispheres are required (the WM label is
                    reconciled against their union), but only `hemis` are
                    propagated.
    build_white     build the white surfaces here from the prep's segmentation
                    (nighres topology correction + marching cubes + `nsmooth`
                    Taubin steps) instead of reading ?h.white. Defaults to True
                    when neither surf_dir nor surfaces is given, so the whole
                    chain segmentation -> white -> field -> pial runs in one
                    process with nothing going through disk.
    propagate_on    'cuda' keeps the field in GPU memory from solve to surface;
                    'cpu' uses the numpy reference implementation
    velocity        reuse a field instead of solving (tensor or [D,H,W,3] array)
    pin             pin the medial wall (needs softmax_seg.nii.gz + label_def.csv)
    compute_thickness
                    False skips the DiReCT thickness map (and with it the
                    THICKNESS_PRIOR velocity cap -- see solve_velocity_field_t).
                    'thickness' comes back None.
    report          print evaluate_surface metrics per hemisphere. Defaults to
                    `verbose`, but it costs ~4s per 130k vertices and scales with
                    vertex count, so pass report=False in a loop.

    Returns {'surfaces': {hemi: (pial_verts, faces)}, 'white': {hemi: (v, f)},
             'velocity', 'thickness', 'seg', 'ref_img', 'tovox', 'totkr'}.
    """
    import pandas as pd

    if build_white is None:
        build_white = surfaces is None and not surf_dir
    if build_white:
        if surfaces is not None:
            raise ValueError('build_white=True and surfaces= are mutually exclusive')
        if verbose:
            print('building the white surfaces (%d Taubin steps)...' % nsmooth)
        # Both hemispheres regardless of `hemis`: the WM label is reconciled
        # against their union.
        surfaces = wm_surface.build_white_surfaces(prep_dir, regions=('lh', 'rh'),
                                                   nsmooth=nsmooth, verbose=verbose)
        surf_dir = None

    d = pc.prepare(prep_dir, surf_dir, hemis=tuple(hemis), surfaces=surfaces)
    seg, tovox, totkr, ref_img = d['seg'], d['tovox'], d['totkr'], d['ref_img']
    report = verbose if report is None else report
    on_gpu = str(propagate_on).lower() == 'cuda'

    thickness = None
    if velocity is None:
        if verbose:
            print('solving the velocity field (%d iterations)...' % pc.MAX_ITERATIONS)
        vel_t, thick_t, _dev = pc.solve_velocity_field_t(
            seg, d['gmT'], d['wmT'], ref_img, verbose=verbose, device=device,
            compute_thickness=compute_thickness)
        thickness = thick_t.squeeze().cpu().numpy() if thick_t is not None else None
        # Only leave the GPU if something actually needs the host copy.
        velocity = vel_t if (on_gpu and not out_dir) else pc.velocity_to_numpy(vel_t)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            img = nib.Nifti1Image(velocity, ref_img.affine)
            img.header['xyzt_units'] = 10
            nib.save(img, os.path.join(out_dir, 'pial_Velocity.nii.gz'))
            if on_gpu:
                velocity = vel_t            # keep using the resident tensor

    soft_seg = id_map = None
    if pin:
        soft = os.path.join(prep_dir, 'softmax_seg.nii.gz')
        ldef = os.path.join(prep_dir, 'label_def.csv')
        if os.path.exists(soft) and os.path.exists(ldef):
            soft_seg = np.asarray(nib.load(soft).dataobj)
            id_map = {r.LABEL: int(r.ID) for _, r in pd.read_csv(ldef).iterrows()}
        else:
            print('WARNING: no softmax_seg.nii.gz / label_def.csv; the medial wall will '
                  'not be pinned and will be dragged outward.', file=sys.stderr)

    out = {}
    for hemi in hemis:
        white, faces = d['surfaces'][hemi]
        pin_mask = None
        if soft_seg is not None:
            no_push = build_no_push_mask(white, faces, seg, soft_seg, id_map, tovox, rings=0)
            pin_mask = build_pin_mask(no_push, white, faces, seg, soft_seg, id_map, tovox,
                                      scope='medial-wall', rings=0)
            if verbose:
                print('%s: %d/%d vertices with no cortex to move into, %d pinned'
                      % (hemi, no_push.sum(), len(no_push), pin_mask.sum()))
        pial = propagate(white, faces, velocity, seg, tovox, totkr, ref_img=ref_img,
                         pin_mask=pin_mask, on=propagate_on, device=device, dtype=dtype)
        out[hemi] = (pial, faces)
        if out_dir:
            vinfo = volume_info_from_image(ref_img, prep_dir)
            nib.freesurfer.io.write_geometry(os.path.join(out_dir, '%s.pial' % hemi),
                                             pial, faces, create_stamp=None,
                                             volume_info=vinfo)
        if report:
            m = evaluate_surface(white, pial, faces, seg, tovox,
                                 no_push=pin_mask if (pin_mask is not None
                                                      and pin_mask.any()) else None)
            print('%s: displacement %.3f mm, crossed_csf %d, self-intersections %d, '
                  'flipped %.4f%%  [%s]'
                  % (hemi, m['mean_displacement_mm'], m['crossed_csf_count'],
                     m['self_intersections'], m['flipped_face_pct'], propagate_on))
    return dict(surfaces=out, white=d['surfaces'], velocity=velocity,
                thickness=thickness, seg=seg, ref_img=ref_img,
                tovox=tovox, totkr=totkr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True)
    p.add_argument('--surf-dir', help='directory holding ?h.white; omit to build them')
    p.add_argument('--build-white', action='store_true',
                   help='build the white surfaces from the segmentation in-process')
    p.add_argument('--nsmooth', type=int, default=wm_surface.NSMOOTH_DEFAULT,
                   help='Taubin steps for the white surface (default %d)'
                        % wm_surface.NSMOOTH_DEFAULT)
    p.add_argument('--out-dir')
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--propagate-on', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--no-thickness', action='store_true',
                   help='skip the DiReCT thickness map; also drops the THICKNESS_PRIOR '
                        'velocity cap, which does not bind on validated data')
    p.add_argument('--float64', action='store_true',
                   help='run the cuda propagation in double precision')
    p.add_argument('--check', action='store_true',
                   help='propagate BOTH ways off one solve and report the difference')
    args = p.parse_args()
    dtype = torch.float64 if args.float64 else torch.float32

    if not args.check:
        reconstruct(args.prep_dir, args.surf_dir, hemis=tuple(args.hemi),
                    propagate_on=args.propagate_on, out_dir=args.out_dir, dtype=dtype,
                    compute_thickness=not args.no_thickness,
                    build_white=args.build_white or None, nsmooth=args.nsmooth)
        return

    import time
    # pin=False so both paths see identical inputs; the pin blend is exercised
    # separately below.
    r = reconstruct(args.prep_dir, args.surf_dir, hemis=tuple(args.hemi),
                    propagate_on='cpu', out_dir=None, verbose=True, pin=False,
                    build_white=args.build_white or None, nsmooth=args.nsmooth)
    for hemi in args.hemi:
        white, faces = r['white'][hemi]
        cpu = r['surfaces'][hemi][0]
        for dt, nm in ((torch.float32, 'float32'), (torch.float64, 'float64')):
            t = time.time()
            gpu = propagate(white, faces, r['velocity'], r['seg'], r['tovox'], r['totkr'],
                            ref_img=r['ref_img'], pin_mask=None, on='cuda', dtype=dt)
            dt_s = time.time() - t
            e = np.linalg.norm(gpu - cpu, axis=1)
            print('%s cuda/%s vs cpu: median %.3e  mean %.3e  max %.3e mm   (%.2fs)'
                  % (hemi, nm, np.median(e), e.mean(), e.max(), dt_s))
        # and once WITH the medial-wall pin, to exercise that branch too
        import pandas as pd
        soft = os.path.join(args.prep_dir, 'softmax_seg.nii.gz')
        ldef = os.path.join(args.prep_dir, 'label_def.csv')
        if os.path.exists(soft) and os.path.exists(ldef):
            ss = np.asarray(nib.load(soft).dataobj)
            im = {row.LABEL: int(row.ID) for _, row in pd.read_csv(ldef).iterrows()}
            npsh = build_no_push_mask(white, faces, r['seg'], ss, im, r['tovox'], rings=0)
            pm = build_pin_mask(npsh, white, faces, r['seg'], ss, im, r['tovox'],
                                scope='medial-wall', rings=0)
            a = pc.propagate_pial(white, faces,
                                  pc.velocity_to_numpy(r['velocity'])
                                  if torch.is_tensor(r['velocity']) else r['velocity'],
                                  r['seg'], r['tovox'], r['totkr'], pin_mask=pm)
            b = propagate(white, faces, r['velocity'], r['seg'], r['tovox'], r['totkr'],
                          ref_img=r['ref_img'], pin_mask=pm, on='cuda', dtype=torch.float64)
            e = np.linalg.norm(b - a, axis=1)
            print('%s pinned, cuda/float64 vs cpu: median %.3e  max %.3e mm'
                  % (hemi, np.median(e), e.max()))


if __name__ == '__main__':
    main()
