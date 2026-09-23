"""Compare the eikonal thickness prototype against a propagated pial surface.

Companion to eikonal_thickness.py. Produces two views of the same
disagreement:

  --vertex   per-vertex overlays on the shared white surface, for freeview:
             ?h.eik_mid, ?h.eik_base, ?h.iso_travel, the two differences,
             and ?h.pial_to_envelope -- the signed mm gap between the two
             methods' OUTER boundaries, which is the surface-to-surface
             comparison proper rather than a thickness comparison.
  --slices   coronal panels with the eikonal thickness map underneath and the
             two OUTER boundaries drawn over it -- the propagated pial as its
             own vertices, the eikonal's GM envelope as a contour -- plus the
             shock set Sigma.

Why a vertex-paired comparison is available at all
--------------------------------------------------
The two methods start from the SAME white surface. The isowhite arm was run
with --reuse-white, so its pial was propagated from field_pial_sigma0.65's
?h.white, and eikonal_thickness --surface-pv rasterises that same mesh as
its WM boundary. Vertex i of ?h.white therefore corresponds to vertex i of
?h.pial, and the difference below is a true paired difference rather than a
nearest-neighbour approximation between two unrelated meshes.

Frames
------
Nothing here invents a transform. The white and pial surfaces are already in
the solve's tkrRAS, and the solve grid is build_surface_segmentation's
ref_img, so `tovox` from that same call takes vertices to voxel indices.
check_frame_alignment is run before anything is sampled and the numbers are
printed: a white surface sits ON the WM/GM interface, so its mean distance
to the WM boundary must be a small fraction of a voxel. If that number is
not small, stop -- everything downstream is shifted.

Reading the two vertex overlays
-------------------------------
The eikonal field is a voxel map and a white vertex sits exactly on its
inner boundary, where sampling is ill-posed. Two independent readouts are
written rather than one, because their agreement is itself the test of the
method's central assumption -- that T_wm + T_out is constant along a
cortical column:

  ?h.eik_mid   sampled at the MIDPOINT of the white->pial segment, i.e. in
               the middle of the ribbon where the field is unambiguous. Uses
               the pial only to locate where to read, never for the value.
  ?h.eik_base  sampled at the nearest GM voxel to the white vertex, using no
               pial information at all.

Where the assumption holds these two agree. Where they diverge, the column
is one the first-arrival approximation does not describe -- and that is
worth looking at before reading anything into eik-minus-iso there.

?h.pial_to_envelope is the one to look at first if the question is about
the surfaces rather than about thickness: it samples the signed distance to
the tissue/CSF interface at each pial vertex, so + means the pial stopped
inside the eikonal's envelope and - means it pushed out past it.

?h.iso_travel is |pial_vertex - white_vertex|, the `travel` definition.
Note the package documents its own thickness definitions as separating by
0.12-0.70 mm region-dependently, so a non-zero difference map is expected
and is not by itself evidence that either side is wrong.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import eikonal_thickness as E   # noqa: E402


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_case(case, white_run, nsmooth=50, topology='nighres', crop=True,
              verbose=True):
    """PV maps, the solve grid, the GM envelope meshes and tovox, in one call."""
    from dldirect import surface_seg, pial_pipeline
    from dldirect.field_pial_prototype import make_transforms
    reused = pial_pipeline._load_white_for_reuse(
        white_run, case, hemis=('lh', 'rh'), nsmooth=nsmooth,
        topology=topology, crop=crop, verbose=False)
    sd = surface_seg.build_surface_segmentation(
        case, hemis=('lh', 'rh'), nsmooth=nsmooth, crop=crop, topology=topology,
        wm_surfaces=reused, correct_ribbon=True, verbose=verbose)
    tovox, totkr = make_transforms(sd['ref_img'])
    return sd, tovox, totkr


def run_eikonal(p_gm, p_wm, spacing, div_k, shock):
    """The prototype's own pipeline, on maps already in memory."""
    ns = argparse.Namespace(phi='mask', speed='binary', eps=1e-2,
                            gm_thr=0.5, wm_thr=0.5)
    gm, wm = E.build_masks(p_gm, p_wm, ns.gm_thr, ns.wm_thr)
    t_wm, _ = E.solve_t_wm(p_gm, p_wm, gm, wm, spacing, ns)
    unit, _ = E.characteristic_directions(t_wm, gm, wm, spacing)
    sigma = None
    if shock != 'none':
        sig, _ = E.shock_by_divergence(unit, gm, spacing, div_k)
        sigma = sig & ~E.exposed_rim(gm, wm)
    t_out, _ = E.solve_t_out(p_gm, gm, wm, sigma, spacing, ns)
    thick = np.where(gm, t_wm + t_out, 0.0).astype(np.float32)
    return dict(gm=gm, wm=wm, t_wm=t_wm, t_out=t_out, thickness=thick,
                sigma=(np.zeros_like(gm) if sigma is None else sigma))


def read_pair(case, iso_run, white_run, hemi):
    """(white, pial, faces) for one hemisphere, with the pairing checked."""
    fsio = nib.freesurfer.io
    wp = os.path.join(case, white_run, '%s.white' % hemi)
    pp = os.path.join(case, iso_run, '%s.pial' % hemi)
    for p in (wp, pp):
        if not os.path.exists(p):
            raise SystemExit('missing %s' % p)
    w, wf = fsio.read_geometry(wp)
    p, pf = fsio.read_geometry(pp)
    if w.shape != p.shape or wf.shape != pf.shape:
        raise SystemExit(
            '%s: white %s and pial %s are not the same mesh -- a paired '
            'comparison needs the pial propagated from THIS white (see the '
            'module docstring on --reuse-white)' % (hemi, w.shape, p.shape))
    return w.astype(np.float64), p.astype(np.float64), wf


# ---------------------------------------------------------------------------
# the vertex view
# ---------------------------------------------------------------------------

def sample_at(vol, pos_vox, order=1):
    return map_coordinates(vol, pos_vox.T, order=order, mode='nearest')


def nearest_gm_value(vol, gm, pos_vox, spacing):
    """vol read at the nearest GM voxel to each position.

    distance_transform_edt on ~gm gives, for every voxel, the index of the
    closest GM voxel; looking the positions up in that index map is a
    nearest-GM lookup that never reads the zeros outside the ribbon, which a
    trilinear sample on the boundary would.
    """
    _, idx = ndi.distance_transform_edt(~gm, sampling=spacing,
                                        return_indices=True)
    ijk = np.rint(pos_vox).astype(int)
    for a in range(3):
        ijk[:, a] = np.clip(ijk[:, a], 0, vol.shape[a] - 1)
    src = tuple(idx[a][ijk[:, 0], ijk[:, 1], ijk[:, 2]] for a in range(3))
    return vol[src]


def envelope_signed_distance(tissue, spacing):
    """Signed distance to the BINARISED tissue mask: + inside, - outside.

    Kept only as the voxel-grid counterpart of mesh_gap below, because the
    difference between the two is worth seeing. This one quantises the
    boundary to the mask staircase, so it cannot resolve better than about
    half a voxel -- the very thing the partial-volume maps exist to avoid.
    Prefer mesh_gap.
    """
    din = ndi.distance_transform_edt(tissue, sampling=spacing)
    dout = ndi.distance_transform_edt(~tissue, sampling=spacing)
    return (din - dout).astype(np.float32)


def mesh_gap(pial, gm_verts, pv_tissue, tovox):
    """Signed mm distance from each pial vertex to the GM ENVELOPE MESH.

    The envelope is a real marching-cubes surface -- surface_seg meshes the
    topology-corrected ribbon with osf.mesh_envelope, and the PV maps this
    module solves on are a rasterisation OF that mesh. Measuring against the
    mesh therefore compares two surfaces; measuring against the rasterised,
    then thresholded, mask compares one surface to a staircase.

    Magnitude is the nearest-VERTEX distance, the same approximation
    regional_stats' `nn` metric uses. It slightly overestimates the true
    point-to-surface distance, by up to about half an edge length, since the
    nearest point on a triangle is generally not one of its vertices.

    Sign comes from the partial-volume occupancy interpolated at the vertex
    (inside where GM+WM > 0.5), which is sub-voxel, rather than from a
    voxel-centre inside/outside test. Positive = the pial stopped inside the
    envelope, negative = it pushed out past it.
    """
    from scipy import spatial
    d = spatial.cKDTree(np.asarray(gm_verts, float)).query(np.asarray(pial, float))[0]
    inside = sample_at(pv_tissue, tovox(pial)) > 0.5
    return (np.where(inside, d, -d)).astype(np.float32)


def vertex_view(case, out_dir, sd, tovox, eik, spacing, iso_run, white_run,
                hemis=('lh', 'rh')):
    from dldirect.field_pial_prototype import check_frame_alignment
    fsio = nib.freesurfer.io
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for h in hemis:
        w, p, f = read_pair(case, iso_run, white_run, h)
        w_vox = tovox(w)
        d_mm, wm_frac, inb = check_frame_alignment(w, sd['seg'], tovox)
        print('%s frame check: white vertices %.3f mm from the WM boundary, '
              '%.0f%% land in WM, %.0f%% in bounds' % (h, d_mm, 100 * wm_frac,
                                                       100 * inb))
        if d_mm > 1.5:
            raise SystemExit(
                '%s: white vertices sit %.2f mm from the WM boundary -- that '
                'is a frame mismatch, not a result. Refusing to sample.'
                % (h, d_mm))

        iso = np.linalg.norm(p - w, axis=1).astype(np.float32)

        # How far apart the two OUTER boundaries actually are, per vertex --
        # against the envelope MESH, with the mask-staircase version printed
        # beside it so the cost of binarising is visible.
        gm_verts = sd['gm_surfaces'][h][0]
        pv_tissue = (np.asarray(sd['gmT'], np.float32)
                     + np.asarray(sd['wmT'], np.float32))
        gap = mesh_gap(p, gm_verts, pv_tissue, tovox)
        mask_gap = sample_at(envelope_signed_distance(eik['gm'] | eik['wm'],
                                                      spacing),
                             tovox(p)).astype(np.float32)
        for lab, v in (('mesh ', gap), ('mask ', mask_gap)):
            print('  %s: pial to envelope (%s) median %+.3f  mean %+.3f  '
                  'p5 %+.3f  p95 %+.3f mm;  %.1f%% inside'
                  % (h, lab, np.median(v), v.mean(), np.percentile(v, 5),
                     np.percentile(v, 95), 100.0 * (v > 0).mean()))
        mid_vox = tovox(0.5 * (w + p))
        eik_mid = sample_at(eik['thickness'], mid_vox).astype(np.float32)
        eik_base = nearest_gm_value(eik['thickness'], eik['gm'], w_vox,
                                    spacing).astype(np.float32)

        # A midpoint can fall outside the ribbon where the pial is very close
        # to the white; those samples read the zero background, so flag them
        # rather than letting a zero masquerade as a thin cortex.
        mid_gm = sample_at(eik['gm'].astype(np.float32), mid_vox) > 0.5
        print('  %s: %d of %d midpoints (%.1f%%) fall outside the GM ribbon'
              % (h, int((~mid_gm).sum()), mid_gm.size,
                 100.0 * (~mid_gm).sum() / mid_gm.size))
        eik_mid = np.where(mid_gm, eik_mid, np.nan).astype(np.float32)

        ok = np.isfinite(eik_mid)
        agree = eik_mid[ok] - eik_base[ok]
        print('  %s: eik_mid vs eik_base  mean %+.3f  median %+.3f  '
              'p95|.| %.3f mm   (the constant-along-column assumption)'
              % (h, agree.mean(), np.median(agree),
                 np.percentile(np.abs(agree), 95)))
        for name, v in (('iso_travel', iso), ('eik_mid', eik_mid),
                        ('eik_base', eik_base), ('pial_to_envelope', gap),
                        ('pial_to_envelope_mask', mask_gap),
                        ('diff_mid', eik_mid - iso),
                        ('diff_base', eik_base - iso)):
            d = np.nan_to_num(v, nan=0.0).astype(np.float32)
            path = os.path.join(out_dir, '%s.%s' % (h, name))
            fsio.write_morph_data(path, d, fnum=len(f))
            written.append(path)
        v = (eik_base - iso)
        print('  %s: eik_base - iso_travel  mean %+.3f  median %+.3f  '
              'p5 %+.3f  p95 %+.3f mm'
              % (h, v.mean(), np.median(v), np.percentile(v, 5),
                 np.percentile(v, 95)))
        src = os.path.join(case, white_run, '%s.white' % h)
        dst = os.path.join(out_dir, '%s.white' % h)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    return written


def freeview_command(out_dir, hemis):
    parts = ['freeview']
    for h in hemis:
        parts.append('-f %s/%s.white:overlay=%s/%s.diff_base:'
                     'overlay_threshold=-1,1:overlay_color=colorwheel'
                     % (out_dir, h, out_dir, h))
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# the slice view
# ---------------------------------------------------------------------------

def mesh_slice_segments(verts_vox, faces, axis, level, show):
    """Line segments where a mesh crosses the plane `axis == level`.

    The exact cross-section, not an approximation: every triangle straddling
    the plane contributes one segment, found by linearly interpolating along
    its two edges whose endpoints fall on opposite sides. Returned as an
    (N, 2, 2) array of 2-D endpoints in the `show` display axes, ready for a
    LineCollection.

    Drawing vertices instead -- which is what this figure did first -- shows
    only the vertices that happen to lie within half a voxel of the plane, so
    a surface running obliquely through the slice appears as a broken cloud
    and one running tangentially appears as a solid band. Neither is the
    cross-section.
    """
    v = np.asarray(verts_vox, float)
    f = np.asarray(faces, int)
    d = v[:, axis] - float(level)
    dv = d[f]                                     # (F, 3)
    pos = dv > 0
    n = pos.sum(1)
    keep = (n == 1) | (n == 2)                    # straddles the plane
    if not keep.any():
        return np.empty((0, 2, 2))
    f, dv = f[keep], dv[keep]
    pos = pos[keep]

    pts = np.empty((f.shape[0], 3, 3))            # per edge, the crossing point
    hit = np.zeros((f.shape[0], 3), bool)
    for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        da, db = dv[:, a], dv[:, b]
        hit[:, k] = pos[:, a] != pos[:, b]
        den = da - db
        t = np.where(np.abs(den) < 1e-12, 0.5, da / np.where(den == 0, 1, den))
        va, vb = v[f[:, a]], v[f[:, b]]
        pts[:, k] = va + t[:, None] * (vb - va)

    # exactly two of the three edges cross; take those two per triangle
    order = np.argsort(~hit, axis=1, kind='stable')[:, :2]
    rows = np.arange(f.shape[0])[:, None]
    two = pts[rows, order]                        # (F, 2, 3)
    return two[:, :, show]


def coronal_axis(ref_img):
    """Array axis running anterior-posterior, and whether it increases A."""
    codes = nib.aff2axcodes(ref_img.affine)
    for a, c in enumerate(codes):
        if c in ('A', 'P'):
            return a, (c == 'A')
    return 1, True


def load_background(case, ref_img):
    """The cropped T1 that shares the solve grid, or None."""
    path = os.path.join(case, 'T1w_norm_noskull_cropped.nii.gz')
    if not os.path.exists(path):
        return None
    img = nib.load(path)
    if tuple(img.shape[:3]) != tuple(ref_img.shape[:3]):
        print('  background %s is %s, solve grid is %s -- skipping'
              % (os.path.basename(path), img.shape[:3], ref_img.shape[:3]))
        return None
    return img.get_fdata(dtype=np.float32)


def slice_view(out_dir, sd, tovox, eik, case, iso_run, white_run, hemi='lh',
               n_slices=4, out_name='eikonal_vs_pial_coronal.png', win=56,
               show_sigma=True, background='t1', eikonal_pial=None,
               show_envelope=True):
    """Coronal panels comparing the two OUTER boundaries on the anatomy.

      blue    the white surface, shared by both methods
      red     the propagated pial, drawn as the exact cross-section of the
              mesh with the slice plane (mesh_slice_segments), not rasterised
              and not a cloud of nearby vertices
      cyan    the GM ENVELOPE MESH -- surface_seg's marching-cubes surface
              over the topology-corrected ribbon, drawn as its own vertices
              as its true cross-section through the slice plane, exactly
              like the pial, so this is a mesh-to-mesh comparison.
              It is what the eikonal's inward front is seeded on; the fast
              marching itself produces no surface, it inherits this one.
              An earlier version drew a matplotlib contour of the binarised
              GM u WM mask instead, which is that mesh rasterised, then
              thresholded, then staircased -- three lossy steps, and not
              something to measure a sub-voxel offset against.
      yellow  Sigma, the shock set, which acts as additional outer boundary
              wherever the segmentation has closed a sulcus

    Panels are ZOOMED to a `win`-voxel window centred on the shock in that
    slice and restricted to the plotted hemisphere. A whole coronal slice is
    ~140 voxels across, at which the ~1 mm separation this figure exists to
    show is sub-pixel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ax_ap, _ = coronal_axis(sd['ref_img'])
    gm, sig = eik['gm'], eik['sigma']
    w, p, w_faces = read_pair(case, iso_run, white_run, hemi)
    p_vox, w_vox = tovox(p), tovox(w)
    g_verts, g_faces = sd['gm_surfaces'][hemi]
    g_vox = tovox(np.asarray(g_verts, float))
    e_vox = None
    if eikonal_pial:
        ep = os.path.join(eikonal_pial, '%s.pial_eikonal' % hemi)
        if os.path.exists(ep):
            ev, _ = nib.freesurfer.io.read_geometry(ep)
            e_vox = tovox(np.asarray(ev, float))
        else:
            print('  no %s, skipping the eikonal pial curve' % ep)
    anat = load_background(case, sd['ref_img']) if background == 't1' else None

    # Restrict to this hemisphere's own extent along L-R, so a panel cannot
    # centre on the other hemisphere's shock.
    lr = [a for a in range(3) if a != ax_ap][0]
    lo = int(np.floor(np.percentile(p_vox[:, lr], 0.5)))
    hi = int(np.ceil(np.percentile(p_vox[:, lr], 99.5)))
    hemi_mask = np.zeros(gm.shape, bool)
    sel = [slice(None)] * 3
    sel[lr] = slice(max(0, lo), min(gm.shape[lr], hi + 1))
    hemi_mask[tuple(sel)] = True
    sig_h = sig & hemi_mask

    per = sig_h.sum(axis=tuple(a for a in range(3) if a != ax_ap))
    if per.sum() == 0:
        per = (gm & hemi_mask).sum(axis=tuple(a for a in range(3) if a != ax_ap))
    order = np.argsort(per)[::-1]
    picked = []
    for q in order:
        if all(abs(int(q) - r) >= 8 for r in picked):
            picked.append(int(q))
        if len(picked) == n_slices:
            break
    picked.sort()

    show = [a for a in range(3) if a != ax_ap]
    vmax = np.percentile(anat[anat > 0], 99.5) if anat is not None else 1.0
    fig, axes = plt.subplots(1, len(picked), figsize=(5.2 * len(picked), 5.6),
                             facecolor='black')
    axes = np.atleast_1d(axes)
    for ax, q in zip(axes, picked):
        sl = [slice(None)] * 3
        sl[ax_ap] = q
        sl = tuple(sl)
        ax.set_facecolor('black')
        if anat is not None:
            ax.imshow(anat[sl].T, origin='lower', cmap='gray', vmin=0,
                      vmax=vmax, interpolation='bilinear')
        ss = sig_h[sl]
        if show_sigma and ss.any():
            yy, xx = np.nonzero(ss)
            ax.scatter(yy, xx, s=6, c='#ffe14d', marker='s', linewidths=0,
                       alpha=0.85)
        curves = []
        if show_envelope:
            curves.append((g_vox, g_faces, '#38f5d0', 0.8))
        curves += [(w_vox, w_faces, '#4da3ff', 0.9),
                   (p_vox, w_faces, '#ff2d55', 1.15)]
        if e_vox is not None:
            curves.append((e_vox, w_faces, '#7CFC00', 1.15))
        for V, F, col, lw in curves:
            seg = mesh_slice_segments(V, F, ax_ap, q, show)
            if len(seg):
                ax.add_collection(LineCollection(seg, colors=col,
                                                 linewidths=lw))
        src = ss if ss.any() else (gm & hemi_mask)[sl]
        yy, xx = np.nonzero(src)
        cx, cy = float(yy.mean()), float(xx.mean())
        ax.set_xlim(cx - win / 2.0, cx + win / 2.0)
        ax.set_ylim(cy - win / 2.0, cy + win / 2.0)
        ax.set_title('coronal %d   Sigma %d in view' % (q, int(ss.sum())),
                     fontsize=9, color='white')
        ax.set_xticks([]); ax.set_yticks([])
    h0 = [plt.Line2D([], [], color='#ff2d55', lw=1.6,
                     label='propagated pial (%s)' % iso_run),
          plt.Line2D([], [], color='#4da3ff', lw=1.6,
                     label='white (shared start)'),

          plt.Line2D([], [], color='#ffe14d', marker='s', ls='', ms=4,
                     label='Sigma (shock set)')]
    if show_envelope:
        h0.insert(2, plt.Line2D([], [], color='#38f5d0', lw=1.6,
                                label='GM envelope mesh'))
    if eikonal_pial:
        h0.insert(1, plt.Line2D([], [], color='#7CFC00', lw=1.6,
                                label='eikonal pial (flow-propagated)'))
    leg = fig.legend(handles=h0, loc='lower center', ncol=4, frameon=False,
                     fontsize=9)
    for t in leg.get_texts():
        t.set_color('white')
    fig.suptitle('outer boundaries on the T1 \u2014 %s' % hemi, fontsize=11,
                 color='white')
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = os.path.join(out_dir, out_name)
    fig.savefig(path, dpi=170, facecolor='black')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('Why a vertex')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', help='an OASIS case directory')
    ap.add_argument('out_dir')
    ap.add_argument('--iso-run', default='field_pial_isowhite',
                    help='run directory holding ?h.pial (default %(default)s)')
    ap.add_argument('--white-run', default='field_pial_sigma0.65',
                    help='run directory holding ?h.white, i.e. the mesh the '
                         'pial was propagated from (default %(default)s)')
    ap.add_argument('--hemi', nargs='+', default=['lh', 'rh'],
                    choices=['lh', 'rh'])
    ap.add_argument('--div-k', type=float, default=1.0)
    ap.add_argument('--shock', choices=('div', 'none'), default='div')
    ap.add_argument('--slices', type=int, default=4)
    ap.add_argument('--no-sigma', action='store_true',
                    help='leave the shock set off the slice panels')
    ap.add_argument('--background', choices=('t1', 'none'), default='t1')
    ap.add_argument('--win', type=int, default=56,
                    help='zoom window in voxels per panel (default %(default)s; '
                         'a whole coronal slice is ~140 across and the '
                         'difference is then sub-pixel)')
    ap.add_argument('--no-vertex', action='store_true')
    ap.add_argument('--no-slices', action='store_true')
    ap.add_argument('--no-envelope', action='store_true',
                    help='do not draw the GM envelope mesh')
    ap.add_argument('--eikonal-pial', default=None,
                    help='directory holding ?h.pial_eikonal from '
                         'eikonal_thickness.py --propagate; drawn as a fourth '
                         'curve')
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    sd, tovox, totkr = load_case(args.case, args.white_run)
    ref = sd['ref_img']
    spacing = tuple(float(z) for z in ref.header.get_zooms()[:3])
    p_gm = np.clip(np.asarray(sd['gmT'], np.float32), 0, 1)
    p_wm = np.clip(np.asarray(sd['wmT'], np.float32), 0, 1)
    print('eikonal solve (shock=%s, div-k=%g)' % (args.shock, args.div_k))
    eik = run_eikonal(p_gm, p_wm, spacing, args.div_k, args.shock)
    E.save_img(eik['thickness'],
               os.path.join(args.out_dir, 'eikonal_thickmap.nii.gz'), ref)

    if not args.no_vertex:
        print('vertex view')
        vertex_view(args.case, args.out_dir, sd, tovox, eik, spacing,
                    args.iso_run, args.white_run, tuple(args.hemi))
        print('\n  ' + freeview_command(args.out_dir, args.hemi) + '\n')
    if not args.no_slices:
        print('slice view')
        for h in args.hemi:
            path = slice_view(args.out_dir, sd, tovox, eik, args.case,
                              args.iso_run, args.white_run, hemi=h,
                              n_slices=args.slices, win=args.win,
                              show_sigma=not args.no_sigma,
                              background=args.background,
                              eikonal_pial=args.eikonal_pial,
                              show_envelope=not args.no_envelope,
                              out_name='eikonal_vs_pial_%s.png' % h)
            print('  wrote %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
