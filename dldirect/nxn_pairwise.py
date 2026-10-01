"""End-to-end model-ensemble uncertainty for a PAIR of T1 scans.

    python -m dldirect.nxn_pairwise --t1 scanA.nii.gz scanB.nii.gz --out DIR

Segments each scan with several models, builds ONE white surface shared by every
model and both scans, solves each configuration on its own native data, and
reports per-vertex and per-parcel uncertainty.

WHY IT IS BUILT THIS WAY.  Each decision below was measured, and the obvious
alternative was tried and rejected:

  Segment NATIVELY, never resample the T1.  Resampling changes what the
  segmenter sees.  Only the LEVELSETS travel between spaces, and the MESH comes
  back by affine, which is exact.

  Register to an unbiased MIDPOINT (mri_robust_template), not A->B.  Mapping B
  onto A interpolates B alone; the smoothing asymmetry appears directly as a
  thickness difference.  No --iscale: it alters intensities the segmenter reads.

  NO GAUSSIAN on the levelset.  Blurring a signed distance field is not a
  surface smoother -- it moves the zero level asymmetrically in concave
  geometry, closing sulci whose banks are ~2 voxels apart.  Measured against the
  directly-built surface: sigma 1.0 keeps 77.5% of its area, sigma 0.25 and
  sigma 0 both keep 93.2%, at unchanged mesh edge length.  It was missing
  surface, not coarser sampling.

  The levelset comes from the MESHED, Taubin-smoothed, FILLED WM (surface-pv),
  not from the posterior.  The posterior is not white matter in the ventricles
  or basal ganglia, so its 0.5 level carves through them -- 10% of vertices
  landed >2 mm from the label-built surface.  And `d = 0.5 - pv` is only valid
  for true occupancy: applied to a posterior it rewrites the distance field
  across the whole ribbon and manufactures spurious zero crossings.

  Topology is corrected ONCE, on the joint mean, in the common space.  The
  projected surfaces inherit it because the mesh is moved, never re-cut, so
  per-scan correction is wasted work.  Skipping it per scan (topology='none',
  correct_ribbon=False) also matches the shipped euclidean pipeline, which
  reuses white surfaces and corrects nothing.

  DIAGONAL cells by default.  The 30 crossed cells estimate the WM/GM
  attribution and nothing else: they make the per-vertex uncertainty ~4% SMALLER
  (they dilute) and their pattern correlates 0.956 with the diagonal's.  Use
  --full only for attribution, and note the one-row-plus-one-column shortcut is
  NOT adequate for it (the WM share swings 36-69% depending on the anchor).

  crop_safe on the solve, GPU propagation.  Together ~6x and ~166x; the
  numerical results are unchanged to the fourth decimal.

The shared surface buys CORRESPONDENCE, not reproducibility. The controlled arm
(`nxn_pilot/reliability/independent_arm.py`: the same 40 rescan pairs, the same
solve, only the surface differs) is a null -- parcel-wise eps_mu 1.0444% shared
vs 1.0128% independent, ratio 1.03, shared better in 17/40, Wilcoxon p=0.115.
An earlier "~35% parcel / 3.8x global" claim here came from an uncontrolled
comparison on one pair and is WITHDRAWN; so is a 6-7% deficit seen in the same
arm at n=25, which regressed to the null by n=40. Score any such comparison on
parcels or vertices: the global eps_mu is the solve's voxelwise thickness field,
which never touches the surface, so both arms agree exactly and the test is
vacuous.

What the shared surface does deliver is vertex correspondence across timepoints,
and with it the cluster-mass statistic against an empirical null. It also fits
each native segmentation somewhat better than a per-scan mesh (0.27 vs 0.36 mm
mean deviation, 12/12 scans), for reasons that are untested.

Resumable: every stage writes to `out/` and is skipped if its output is present.
"""
import argparse, os, subprocess, sys, time, json
import numpy as np

DEFAULT_MODELS = ('v0_f1', 'v0_f2', 'v6_f1', 'v6_f2', 'v7_f1', 'v7_f2')
FS_HOME = '/data/disk2/freesurfer'


def log(msg, t0=None):
    el = '' if t0 is None else '  [%.1f min]' % ((time.time() - t0) / 60)
    print('%s  %s%s' % (time.strftime('%H:%M:%S'), msg, el), flush=True)


# --------------------------------------------------------------------------
# stage 1: segmentation, natively, one prep per (scan, model)
# --------------------------------------------------------------------------

def segment_all(t1_paths, models, out, t0):
    import dldirect
    sys.argv[0] = os.path.join(os.path.dirname(dldirect.__file__),
                               'DeepSCAN_Anatomy_Newnet_apply.py')
    from dldirect.DeepSCAN_Anatomy_Newnet_apply import segment, write_segmentation
    here = os.path.dirname(os.path.abspath(dldirect.__file__))
    preps = {}
    for ti, t1 in enumerate(t1_paths):
        for m in models:
            d = os.path.join(out, 'prep', 't%d_%s' % (ti + 1, m))
            preps[(ti, m)] = d
            if os.path.exists(os.path.join(d, 'mri', 'aparc.atlas+aseg.nii.gz')):
                continue
            os.makedirs(d, exist_ok=True)
            s = segment(t1, model=m, verbose=False)
            write_segmentation(s, d, 't%d_%s' % (ti + 1, m), save_logits=None)
            for sub in ('mri', 'label', 'surf'):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
            loc = os.path.join(d, 'T1w_norm_noskull_cropped.nii.gz')
            if not os.path.exists(loc):
                os.symlink(os.path.abspath(t1), loc)
            subprocess.run([sys.executable, os.path.join(here, 'preparedata.py'),
                            '-inputpath', d, '--space', 'cropped'],
                           check=True, stdout=subprocess.DEVNULL)
            log('segmented t%d/%s' % (ti + 1, m), t0)
    return preps


# --------------------------------------------------------------------------
# stage 2: unbiased midpoint registration
# --------------------------------------------------------------------------

def register(t1_paths, out, t0):
    hw = os.path.join(out, 'halfway.mgz')
    ltas = [os.path.join(out, 'scan%d.lta' % (i + 1)) for i in range(len(t1_paths))]
    if not (os.path.exists(hw) and all(os.path.exists(l) for l in ltas)):
        env = dict(os.environ, FREESURFER_HOME=FS_HOME)
        cmd = ([os.path.join(FS_HOME, 'bin', 'mri_robust_template'), '--mov']
               + list(t1_paths) + ['--template', hw, '--satit', '--lta'] + ltas)
        # NOT --iscale: it fits a global intensity scaling, and these volumes are
        # what the segmentation was run on.
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0 or not os.path.exists(hw):
            raise RuntimeError('mri_robust_template failed:\n%s' % r.stderr[-2000:])
        log('registered to an unbiased midpoint', t0)
    return hw, ltas


def read_lta(path):
    L = open(path).readlines()
    i = [k for k, l in enumerate(L) if l.strip() == '1 4 4'][0]
    return np.array([[float(x) for x in L[i + 1 + r].split()] for r in range(4)])


# --------------------------------------------------------------------------
# stage 3: native segmentations and levelsets
# --------------------------------------------------------------------------

def signed_distance(pv):
    """Signed distance of a TRUE OCCUPANCY map, negative inside. NO smoothing."""
    from scipy.ndimage import distance_transform_edt
    ins = pv > 0.5
    d = (distance_transform_edt(~ins).astype(np.float32)
         - distance_transform_edt(ins).astype(np.float32))
    band = (pv > 0) & (pv < 1)            # one voxel wide for real occupancy
    d[band] = (0.5 - pv[band]).astype(np.float32)
    return d


def build_natives(preps, out, t0, nsmooth=50):
    """Native PV maps, plus ONE LEVELSET PER HEMISPHERE.

    surface_seg ALREADY builds the two hemispheres separately (`build_hemisphere`
    per hemi, returned as `surfaces={'lh':..., 'rh':...}`); it is the single line
    `pv_wm += rasterize_mesh_pv(...)` that sums them into one volume. Taking the
    levelset of that combined wmT threw the separation away: the corpus callosum
    is white matter, so the union is one connected object and meshes to a single
    genus-0 whole-brain surface.

    That is topologically fine but useless downstream -- `mris_register` aligns
    to fsaverage's per-hemisphere spheres and has no target for a whole-brain
    one. So rasterise each hemisphere's own surface, which surface_seg has
    already handed us, and keep them apart.

    The COMBINED wmT is still what the solve consumes -- the velocity field is
    volumetric and the ribbon is whole-brain. Only the template is per-hemisphere.
    """
    from dldirect import surface_seg
    from dldirect.field_pial_prototype import make_transforms, rasterize_mesh_pv
    SD, LV = {}, {}
    for k, prep in preps.items():
        d = surface_seg.build_surface_segmentation(
            prep, hemis=('lh', 'rh'), nsmooth=nsmooth, crop=True,
            topology='none', correct_ribbon=False, verbose=False)
        SD[k] = d
        tovox, _ = make_transforms(d['ref_img'])
        shape = np.asarray(d['seg']).shape
        LV[k] = {}
        for h in ('lh', 'rh'):
            hv, hf = d['surfaces'][h]
            pv = np.clip(rasterize_mesh_pv(tovox(np.asarray(hv)), np.asarray(hf),
                                           shape, 3).astype(np.float32), 0, 1)
            LV[k][h] = signed_distance(pv)
    log('%d native segmentations, levelsets per hemisphere' % len(SD), t0)
    return SD, LV


# --------------------------------------------------------------------------
# stage 4: ONE joint template
# --------------------------------------------------------------------------

def joint_template(SD, LV, hw_path, ltas, out, t0):
    import nibabel as nib, nighres
    from scipy.ndimage import map_coordinates
    cache = os.path.join(out, 'template.npz')
    hw = nib.load(hw_path); Ah = hw.affine; SH = tuple(hw.shape[:3])
    if os.path.exists(cache):
        z = np.load(cache)
        if 'hemi' not in z:            # pre per-hemisphere cache: rebuild
            os.remove(cache)
        else:
            log('joint template loaded: %d vertices' % len(z['tv']), t0)
            return z['tv'], z['tf'].astype(int), Ah, SH, z['hemi']
    g = np.indices(SH).reshape(3, -1)
    hom = np.vstack([g, np.ones(g.shape[1])])
    TV, TF, hemi, off = [], [], [], 0
    for h in ('lh', 'rh'):
        acc = np.zeros(SH, np.float64)
        for (ti, m), lv in LV.items():
            Mat = np.linalg.inv(SD[(ti, m)]['ref_img'].affine) @ np.linalg.inv(ltas[ti]) @ Ah
            acc += map_coordinates(lv[h], (Mat @ hom)[:3], order=1,
                                   mode='nearest').reshape(SH)
        mean = (acc / len(LV)).astype(np.float64)
        tc = nighres.shape.topology_correction(
            nib.Nifti1Image(mean, np.eye(4)), 'signed_distance_function',
            minimum_distance=1e-5, propagation='background->object', connectivity='6/18')
        r = nighres.surface.levelset_to_mesh(tc['corrected'], connectivity='6/18')
        v, f = r['result']['points'], r['result']['faces'].astype(int)
        TV.append(v); TF.append(f + off); hemi.append(np.full(len(v), 0 if h == 'lh' else 1, np.int8))
        off += len(v)
    tv = np.vstack(TV); tf = np.vstack(TF); hemi = np.concatenate(hemi)
    np.savez_compressed(cache, tv=tv, tf=tf, hemi=hemi)
    log('joint template from %d levelsets, each hemisphere corrected and meshed '
        'separately: %d + %d vertices' % (len(LV), len(TV[0]), len(TV[1])), t0)
    return tv, tf, Ah, SH, hemi


def project(v, lv, iters=10, max_step=1.0, grad_floor=0.3):
    """Newton onto the zero level. The floor and the clamp are load-bearing:
    without them a handful of vertices travel hundreds of mm and land elsewhere
    on the same zero level, leaving a perfect residual and destroyed
    correspondence."""
    from scipy.ndimage import map_coordinates
    g = np.stack(np.gradient(lv), 0).astype(np.float32)
    out = np.asarray(v, np.float64).copy()
    for _ in range(iters):
        c = out.T
        s = map_coordinates(lv, c, order=1, mode='nearest')
        gg = np.stack([map_coordinates(g[a], c, order=1, mode='nearest')
                       for a in range(3)], 1)
        n = np.linalg.norm(gg, axis=1, keepdims=True)
        step = -(s[:, None] * gg / np.maximum(n, grad_floor) ** 2)
        ln = np.linalg.norm(step, axis=1, keepdims=True)
        out = out + np.where(ln > max_step, step * (max_step / np.maximum(ln, 1e-9)), step)
    return out


def place_surfaces(SD, LV, tv, ltas, Ah, out, t0, hemi, tol=0.06):
    """Template -> each native space by affine (exact), then project. Verifies
    that wmT reads ~0.5 at the placed vertices; a frame error shows up here as a
    plausible-looking but wrong number, so it is checked, not assumed."""
    import nibabel as nib
    from scipy.ndimage import map_coordinates
    from dldirect.field_pial_prototype import get_vox2ras_tkr
    WV, checks, dev = {}, {}, {}
    for k, d in SD.items():
        ref = d['ref_img']
        Mat = np.linalg.inv(ref.affine) @ np.linalg.inv(ltas[k[0]]) @ Ah
        vn = (Mat @ np.vstack([tv.T, np.ones(len(tv))]))[:3].T
        # BEFORE projection: the levelset value at the consensus position is
        # exactly how far this scan-and-model's WM boundary sits from the joint
        # surface, per vertex, in voxels. This is the single-surface QC -- it
        # needs no other model at analysis time. The spread of the PROJECTED
        # surfaces understates it badly (median 0.023 mm), because projection has
        # already pulled every one of them onto its own valid boundary.
        # each hemisphere is projected onto ITS OWN levelset: the two are
        # separate objects now, and projecting a vertex onto the other
        # hemisphere's boundary would silently cross the midline.
        zz = np.empty(len(vn), np.float64)
        vp = np.empty_like(vn)
        for hi, h in enumerate(('lh', 'rh')):
            m = hemi == hi
            if not m.any():
                continue
            zz[m] = np.abs(map_coordinates(LV[k][h], vn[m].T, order=1, mode='nearest'))
            vp[m] = project(vn[m], LV[k][h])
        dev[k] = np.linalg.norm(vp - vn, axis=1).astype(np.float32)
        dev[('levelset', ) + k] = zz.astype(np.float32)
        WV[k] = nib.affines.apply_affine(get_vox2ras_tkr(ref), vp)
        w = float(np.median(map_coordinates(np.asarray(d['wmT'], np.float32),
                                            vp.T, order=1, mode='nearest')))
        checks['t%d_%s' % (k[0] + 1, k[1])] = w
        if abs(w - 0.5) > tol:
            raise RuntimeError('FRAME CHECK FAILED for t%d/%s: wmT at the placed '
                               'vertices is %.3f, expected ~0.50. The transform '
                               'chain or the levelset is wrong -- refusing to '
                               'solve on it.' % (k[0] + 1, k[1], w))
    dd = [np.median(v) for kk, v in dev.items() if not isinstance(kk[0], str)]
    log('surfaces placed; wmT at vertices %.3f-%.3f (want ~0.50); consensus-to-model '
        'boundary offset %.3f-%.3f voxels (median)'
        % (min(checks.values()), max(checks.values()), min(dd), max(dd)), t0)
    return WV, checks, dev


# --------------------------------------------------------------------------
# stage 4b: rebuild the scan's PV maps from the surface we will propagate from
# --------------------------------------------------------------------------

def match_natives(SD, WV, tf, t0):
    """Re-rasterise seg/gmT/wmT from the PLACED template surface, per scan.

    build_natives rasterises wmT from surface_seg's own mesh, but the
    propagation starts from the template projected onto that scan, so the
    velocity field is shaped around a boundary the starting surface does not sit
    on. surface_seg's own `wm_surfaces` note records that this same mismatch
    previously invalidated a 36-hemisphere comparison. It cannot be avoided in
    build_natives -- the template does not exist until the levelsets have been
    averaged -- so it is corrected here, in a second pass. Not circular: the
    template is built from the original levelsets and only the PV maps the solve
    consumes are rebuilt.

    Measured on a Trio/mMR pair: WM Dice between the two boundaries is 0.989,
    and the 1% that disagrees sits where the template and the per-scan mesh
    diverge most -- the frontal and temporal poles, which is exactly where the
    scanner map disagreed with an independently built surface and with
    FreeSurfer 6 cross-sectional. Matching the boundary closes 92% (lh) and 53%
    (rh) of the frontal-pole gap and moves the undisputed parcels by <=0.017 mm.
    It also raises global thickness ~0.06 mm, so it is not scale-neutral.

    The GM envelope is unchanged: pv_gm is recovered as gmT + wmT from the
    original build and only the WM boundary is replaced.
    """
    from dldirect.field_pial_prototype import make_transforms, rasterize_mesh_pv
    for key, d in SD.items():
        gm = np.clip(np.asarray(d['gmT'], np.float32), 0, 1)
        wm = np.clip(np.asarray(d['wmT'], np.float32), 0, 1)
        pv_gm = np.clip(gm + wm, 0, 1)
        tovox, _ = make_transforms(d['ref_img'])
        shape = np.asarray(d['seg']).shape
        pv_wm = np.clip(rasterize_mesh_pv(tovox(WV[key]), tf, shape, 3)
                        .astype(np.float32), 0, 1)
        d['seg'] = np.where(pv_wm > 0.5, 3, np.where(pv_gm > 0.5, 2, 0)).astype(np.uint8)
        d['gmT'] = np.clip(pv_gm - pv_wm, 0, 1).astype(np.float32)
        d['wmT'] = pv_wm
    log('PV maps re-rasterised from the placed surfaces (%d scans)' % len(SD), t0)
    return SD


# --------------------------------------------------------------------------
# stage 5: the solves
# --------------------------------------------------------------------------

def cell_seg(SD, wm_key, gm_key):
    """wm = A says WM; gm = B's cortex+WM support minus A's WM. Where A==B this
    collapses to the un-crossed segmentation exactly, so the diagonal is a true
    no-op and every cell runs one code path. gmT is handed the reclaimed band's
    occupancy because gmT scales the driving speed."""
    si = np.asarray(SD[wm_key]['seg']); sj = np.asarray(SD[gm_key]['seg'])
    wm = si == 3
    gm = ((sj == 2) | (sj == 3)) & ~wm
    seg = np.where(wm, 3., np.where(gm, 2., 0.)).astype(np.float32)
    recl = gm & (sj == 3)
    g = np.clip(np.asarray(SD[gm_key]['gmT'], np.float32), 0, 1)
    if recl.any():
        g = np.clip(g + np.clip(np.asarray(SD[gm_key]['wmT'], np.float32), 0, 1) * recl, 0, 1)
    w = np.clip(np.asarray(SD[wm_key]['wmT'], np.float32), 0, 1)
    return seg, g, w, int(recl.sum())


def solve_cells(SD, WV, tf, models, out, t0, full=False, crop_margin=4):
    import torch
    from dldirect import pial_clean as pc, pial_pipeline as pp
    from dldirect.field_pial_prototype import make_transforms
    from dldirect.pial_clean import extract_wm_contours
    V = dict(verbose=False, compute_thickness=True, smoothing='gated',
             blend_beta=1.0, reorient_alpha=0.5, nu_mode='euclidean')
    tps = sorted({k[0] for k in SD})
    NV = len(tf) and len(WV[(tps[0], models[0])])
    n = len(models)
    PIAL = np.zeros((len(tps), n, n, NV, 3), np.float32)
    TRAV = np.zeros((len(tps), n, n, NV), np.float32)
    THK = np.full((len(tps), n, n), np.nan)
    for ti in tps:
        for a, i in enumerate(models):
            ck = os.path.join(out, 'cells2_t%d_%s.npz' % (ti + 1, i))   # cells_* are pre-match_natives
            if os.path.exists(ck):
                z = np.load(ck)
                PIAL[ti, a] = z['PIAL']; TRAV[ti, a] = z['TRAV']; THK[ti, a] = z['THK']
                log('row t%d/%s from checkpoint' % (ti + 1, i), t0)
                continue
            key = (ti, i); ref = SD[key]['ref_img']
            tovox, totkr = make_transforms(ref)
            # PER SCAN: two timepoints from different scanners do not share a
            # grid, and nu is allocated on this scan's own shape.
            shape = tuple(np.asarray(SD[key]['seg']).shape)
            # nu on the crop: it is a function of (seg, wmT) alone, so one build
            # per WM source serves the whole row.
            seg_i = np.asarray(SD[key]['seg'])
            s_i = np.where(seg_i == 3, 3., np.where(seg_i == 2, 2., 0.)).astype(np.float32)
            w_i = np.clip(np.asarray(SD[key]['wmT'], np.float32), 0, 1)
            st = torch.from_numpy(s_i)[None, None]
            act = (((st == 2).float() + extract_wm_contours(st)).clamp(max=1.0).numpy()[0, 0] > 0)
            idx = np.array(np.nonzero(act))
            lo = np.maximum(idx.min(1) - 12, 0)
            hi = np.minimum(idx.max(1) + 13, np.asarray(shape))
            sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
            nu_c = pc.wm_normal_field(s_i[sl], torch.device('cuda'), wmT=w_i[sl],
                                      spacing=ref.header.get_zooms()[:3])
            nu = torch.zeros((1, 3) + shape, device=nu_c.device, dtype=nu_c.dtype)
            nu[(slice(None), slice(None)) + sl] = nu_c
            for b, j in enumerate(models):
                if not full and a != b:
                    continue
                seg, g, w, _ = cell_seg(SD, key, (ti, j))
                vel, th, _ = pc.solve_velocity_field_t(seg, g, w, ref, nu=nu,
                                                       crop_safe=crop_margin, **V)
                p = np.asarray(pp.propagate(WV[key], tf, vel, seg, tovox, totkr,
                                            ref_img=ref, on='cuda'))
                PIAL[ti, a, b] = p
                TRAV[ti, a, b] = np.linalg.norm(p - WV[key], axis=1)
                THK[ti, a, b] = float(th.squeeze().cpu().numpy()[seg == 2].mean())
            np.savez_compressed(ck, PIAL=PIAL[ti, a], TRAV=TRAV[ti, a], THK=THK[ti, a],
                                WV=WV[key].astype(np.float32))
            log('row t%d/%s done  thickness %s' % (ti + 1, i,
                ' '.join('%.4f' % x for x in THK[ti, a] if np.isfinite(x))), t0)
    return PIAL, TRAV, THK


# --------------------------------------------------------------------------
# stage 6: parcellation, by majority vote on the shared vertices
# --------------------------------------------------------------------------

# aparc.atlas+aseg is NOT usable here: it holds DK for v0/v6 and DESTRIEUX for
# v7, under the same filename. Use the explicit files, which carry standard
# FreeSurfer numbering (DK 1001-2035, Destrieux 11101-12175) and so are directly
# comparable across models without a name mapping. Neither model family produces
# both atlases, so the two votes draw on different subsets.
ATLASES = {'DK':        ('aparc.DKatlas+aseg.mgz', 1000, 3000),
           'Destrieux': ('aparc.2009s+aseg.mgz', 11000, 13000)}
DEPTHS = (0.35, 0.5, 0.65)          # sample across mid-thickness, not one plane


def _tovox(img):
    """.mgz is MGH format and get_vox2ras_tkr reads a NIfTI-only header field;
    nibabel supplies the transform natively for MGH, and the two agree exactly
    where both are available (max|diff| 0)."""
    import nibabel as nib
    from dldirect.field_pial_prototype import get_vox2ras_tkr
    A = (img.header.get_vox2ras_tkr() if hasattr(img.header, 'get_vox2ras_tkr')
         else get_vox2ras_tkr(img))
    Ainv = np.linalg.inv(A)
    return lambda q: nib.affines.apply_affine(Ainv, q)


def parcellate(preps, WV, PIAL, models, out, t0):
    """One label per shared vertex, voted across every model and scan carrying
    the atlas. Parcel means are then over IDENTICAL vertex sets for every model
    and both scans, so the parcellation cannot contribute to a between-model or
    between-scan difference."""
    import nibabel as nib, collections
    from scipy.ndimage import map_coordinates
    res = {}
    for name, (fn, lo, hi) in ATLASES.items():
        votes, sources = [], []
        for (ti, m), prep in sorted(preps.items()):
            f = os.path.join(prep, 'mri', fn)
            if not os.path.exists(f):
                continue
            img = nib.load(f); L = np.asarray(img.dataobj); tovox = _tovox(img)
            k = models.index(m)
            w, pl = WV[(ti, m)], PIAL[ti, k, k]
            per = np.stack([map_coordinates(L, tovox(w + d * (pl - w)).T,
                                            order=0, mode='nearest') for d in DEPTHS])
            vv = np.array([collections.Counter(per[:, i]).most_common(1)[0][0]
                           for i in range(per.shape[1])], per.dtype)
            votes.append(vv); sources.append('t%d/%s' % (ti + 1, m))
        if not votes:
            log('%s: no model in this set produces it, skipped' % name, t0)
            continue
        V = np.stack(votes)
        lab = np.zeros(V.shape[1], np.int32); agree = np.zeros(V.shape[1])
        for i in range(V.shape[1]):
            w_, n_ = collections.Counter(V[:, i]).most_common(1)[0]
            lab[i] = w_; agree[i] = n_ / len(votes)
        ctx = (lab >= lo) & (lab < hi)
        res[name] = (lab, agree)
        log('%s: %d voters, %d parcels, %.1f%% of vertices cortical, %.1f%% unanimous there'
            % (name, len(votes), len(set(lab[ctx])), 100 * ctx.mean(),
               100 * (agree[ctx] == 1).mean()), t0)
    return res


# --------------------------------------------------------------------------
# stage 7: QC
# --------------------------------------------------------------------------

def qc(SD, WV, PIAL, TRAV, tf, models, parc, ltas, out, t0, dev=None):
    """Per-scan quality signals, written to qc.json and qc.npz.

    Three groups:
      GEOMETRY   self-intersecting faces, and the white surface's distance to the
                 WM boundary it should lie on -- the established WM-deviation
                 channel, which reaches AUC 0.94 per parcel against 0.84 global.
      ENSEMBLE   per-scan model disagreement, which a single-model pipeline
                 cannot produce. Where the models disagree the measurement is
                 uncertain FOR THAT SCAN, so this is a confidence map rather than
                 a cohort-level error bar. Also the parcellation vote agreement,
                 whose non-unanimous vertices are the parcel borders.
      ACQUISITION  head motion, from the registration.

    Nothing is thresholded here. Flags are 2.8x commoner in AD, so filtering a
    group comparison on them biases it; emit the numbers and let the analysis
    decide.
    """
    import torch
    from dldirect.field_pial_prototype import make_transforms, _self_intersecting_faces
    from scipy.ndimage import distance_transform_edt, map_coordinates
    n = len(models); rec = {}
    arrays = {}
    for (ti, m), d in sorted(SD.items()):
        k = models.index(m); tag = 't%d_%s' % (ti + 1, m)
        seg = np.asarray(d['seg']); ref = d['ref_img']
        tovox, _ = make_transforms(ref)
        wmb = seg == 3
        sdt = (distance_transform_edt(~wmb).astype(np.float32)
               - distance_transform_edt(wmb).astype(np.float32))
        dist = np.abs(map_coordinates(sdt, tovox(WV[(ti, m)]).T, order=1, mode='nearest'))
        pial = PIAL[ti, k, k]
        si = _self_intersecting_faces(np.asarray(pial, np.float64), tf)
        rec[tag] = dict(
            wm_deviation_mean=float(dist.mean()),
            wm_deviation_p95=float(np.percentile(dist, 95)),
            pial_self_int_faces=int(np.asarray(si, bool).sum()) if si is not None else -1,
            pial_self_int_pct=float(100 * np.asarray(si, bool).mean()) if si is not None else -1.0,
            thickness_mean=float(np.nanmean(TRAV[ti, k, k])),
            wm_voxels=int(wmb.sum()), gm_voxels=int((seg == 2).sum()))
    for kk, v in (dev or {}).items():
        if isinstance(kk[0], str):          # the levelset offset, in voxels
            tag = 't%d_%s' % (kk[1] + 1, kk[2])
            rec.setdefault(tag, {}).update(
                consensus_offset_median_vox=float(np.median(v)),
                consensus_offset_p95_vox=float(np.percentile(v, 95)),
                consensus_offset_mean_vox=float(v.mean()))
            arrays['consensus_offset_%s' % tag] = v
    for ti in range(TRAV.shape[0]):
        dd = TRAV[ti, np.arange(n), np.arange(n)]
        sd = dd.std(0, ddof=1) if n > 1 else np.zeros(dd.shape[1])
        arrays['ensemble_sd_t%d' % (ti + 1)] = sd.astype(np.float32)
        rec['t%d_ensemble' % (ti + 1)] = dict(
            model_sd_median=float(np.median(sd)), model_sd_p95=float(np.percentile(sd, 95)),
            model_sd_mean=float(sd.mean()), n_models=n)
    for name, (lab, agree) in (parc or {}).items():
        arrays['%s_agreement' % name] = agree.astype(np.float32)
        rec['parcellation_%s' % name] = dict(
            unanimous_pct=float(100 * (agree == 1).mean()),
            median_agreement=float(np.median(agree)))
    for i, L in enumerate(ltas):
        ang = float(np.degrees(np.arccos(np.clip((np.trace(L[:3, :3]) - 1) / 2, -1, 1))))
        rec['t%d_motion' % (i + 1)] = dict(rotation_deg=ang,
                                           translation_mm=float(np.linalg.norm(L[:3, 3])))
    json.dump(rec, open(os.path.join(out, 'qc.json'), 'w'), indent=2, sort_keys=True)
    if arrays:
        np.savez_compressed(os.path.join(out, 'qc.npz'), **arrays)
    wm = [v['wm_deviation_mean'] for v in rec.values() if 'wm_deviation_mean' in v]
    si = [v['pial_self_int_pct'] for v in rec.values() if 'pial_self_int_pct' in v]
    log('qc: WM deviation %.3f-%.3f mm, pial self-int %.3f-%.3f%% of faces'
        % (min(wm), max(wm), min(si), max(si)), t0)
    return rec


# --------------------------------------------------------------------------
# stage 8: report
# --------------------------------------------------------------------------

def report(TRAV, THK, models, out, parc=None):
    n = len(models); dg = np.arange(n)
    D = TRAV[:, dg, dg]
    lines = ['model-ensemble uncertainty, %d models, %d timepoints' % (n, TRAV.shape[0]), '']
    for ti in range(TRAV.shape[0]):
        s = D[ti].std(0, ddof=1)
        lines.append('  t%d  per-vertex sd  median %.4f  p95 %.4f mm   global mean %.4f +- %.4f'
                     % (ti + 1, np.median(s), np.percentile(s, 95),
                        np.nanmean(THK[ti][dg, dg]), np.nanstd(THK[ti][dg, dg], ddof=1)))
    if TRAV.shape[0] == 2:
        dd = np.abs(D[1] - D[0])
        lines += ['', '  scan-rescan, same model:  |d| median %.4f  p95 %.4f   '
                      'sigma of one measurement %.4f mm'
                  % (np.median(dd), np.percentile(dd, 95),
                     dd.mean() * np.sqrt(np.pi / 2) / np.sqrt(2))]
        dev = D - D.mean(1, keepdims=True)
        r = np.mean([np.corrcoef(dev[0, k], dev[1, k])[0, 1] for k in range(n)])
        chg = (D[1] - D[0]).std(0, ddof=1)
        lines += ['  model deviation persists across scans: r = %+.3f' % r,
                  '  sd across models of the CHANGE %.4f vs %.4f if fully independent'
                  % (np.median(chg), np.sqrt(2) * np.median(D[0].std(0, ddof=1)))]
    for name, (lab, agree) in (parc or {}).items():
        lo, hi = ATLASES[name][1], ATLASES[name][2]
        pars = sorted(set(lab[(lab >= lo) & (lab < hi)]))
        PM = np.stack([D[:, :, lab == p].mean(2) for p in pars], -1)
        mod = PM.std(1, ddof=1).mean(0)
        lines.append('')
        if TRAV.shape[0] == 2:
            meas = np.abs(PM[1] - PM[0]).mean(0) * np.sqrt(np.pi / 2) / np.sqrt(2)
            tot = np.sqrt(mod ** 2 + meas ** 2)
            lines.append('  %s, %d parcels: model %.4f  measurement %.4f  combined %.4f mm '
                         '(median), range %.4f-%.4f'
                         % (name, len(pars), np.median(mod), np.median(meas),
                            np.median(tot), tot.min(), tot.max()))
        else:
            lines.append('  %s, %d parcels: model choice %.4f mm (median), range %.4f-%.4f'
                         % (name, len(pars), np.median(mod), mod.min(), mod.max()))
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(out, 'report.txt'), 'w').write(txt + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--t1', nargs=2, required=True, metavar='NII',
                   help='the two T1 images, in their native spaces')
    p.add_argument('--out', required=True)
    p.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    p.add_argument('--full', action='store_true',
                   help='all n x n cells, for the WM/GM attribution. The default '
                        'runs the diagonal only: the crossed cells make the '
                        'per-vertex uncertainty ~4%% SMALLER and add nothing to it.')
    p.add_argument('--nsmooth', type=int, default=50)
    p.add_argument('--crop-margin', type=int, default=4)
    a = p.parse_args(argv)
    for f in a.t1:
        if not os.path.exists(f):
            sys.exit('missing input: %s' % f)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log('%d models x %d scans, %s cells' % (len(a.models), len(a.t1),
                                            'all' if a.full else 'diagonal'))
    preps = segment_all(a.t1, a.models, a.out, t0)
    hw, ltas_p = register(a.t1, a.out, t0)
    ltas = [read_lta(l) for l in ltas_p]
    for i, L in enumerate(ltas):
        ang = np.degrees(np.arccos(np.clip((np.trace(L[:3, :3]) - 1) / 2, -1, 1)))
        log('  scan%d -> midpoint: rotation %.3f deg, translation %.3f mm'
            % (i + 1, ang, np.linalg.norm(L[:3, 3])))
    keyed = {(ti, m): preps[(ti, m)] for ti in range(len(a.t1)) for m in a.models}
    SD, LV = build_natives(keyed, a.out, t0, nsmooth=a.nsmooth)
    tv, tf, Ah, SH, hemi = joint_template(SD, LV, hw, ltas, a.out, t0)
    WV, checks, dev = place_surfaces(SD, LV, tv, ltas, Ah, a.out, t0, hemi)
    SD = match_natives(SD, WV, tf, t0)
    PIAL, TRAV, THK = solve_cells(SD, WV, tf, a.models, a.out, t0,
                                  full=a.full, crop_margin=a.crop_margin)
    parc = parcellate(keyed, WV, PIAL, list(a.models), a.out, t0)
    qcrec = qc(SD, WV, PIAL, TRAV, tf, list(a.models), parc, ltas, a.out, t0, dev)
    np.savez_compressed(os.path.join(a.out, 'result.npz'), PIAL=PIAL, TRAV=TRAV,
                        THK=THK, tf=tf, tv=tv, models=np.array(a.models),
                        WV=np.stack([np.stack([WV[(ti, m)] for m in a.models])
                                     for ti in range(len(a.t1))]).astype(np.float32),
                        hemi=hemi,
                        **{'%s_labels' % k: v[0] for k, v in parc.items()},
                        **{'%s_agreement' % k: v[1] for k, v in parc.items()})
    json.dump(dict(frame_checks=checks, models=list(a.models), full=bool(a.full),
                   minutes=round((time.time() - t0) / 60, 2),
                   qc_written=True),
              open(os.path.join(a.out, 'provenance.json'), 'w'), indent=2)
    log('done -> %s/result.npz' % a.out, t0)
    report(TRAV, THK, a.models, a.out, parc)


if __name__ == '__main__':
    main()
