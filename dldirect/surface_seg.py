#!/usr/bin/env python
"""Build DiReCT's inputs from SURFACES rather than from the label volume.

Both boundaries are meshed and then rasterized as partial volume onto the
solve's grid:

    wmT = pv(WM surface)
    gmT = pv(GM surface) - pv(WM surface)        the ribbon's occupancy
    seg = 3 where pv(WM) > 0.5, 2 where pv(GM) > 0.5, else 0

Two things this buys.

SUB-THRESHOLD CSF. The GM surface is the isosurface of the cortical ribbon
AFTER a topology correction. The ribbon carries hundreds of handles because
sulcal banks touch wherever the CSF between them fell below detection; the
voxels the confidence-ordered growth declines to add are exactly those bridges
-- tissue by the argmax, CSF by the topology. Meshing the corrected ribbon
therefore hands the solve a GM boundary with those sulci OPEN. Measured on one
hemisphere: 4200 of 498792 voxels (0.84%), and crossed_csf on the resulting
pial fell 594 -> 336.

ONE FRAME CONVERSION. The pipeline's usual route reconciles a voxel
segmentation against the white surface by rasterising it, which on the measured
subject promoted 71550 voxels -- the two representations disagree and one is
forced onto the other. Here both boundaries are surfaces from the start, the
conformed -> cropped mapping is applied once to both, and there is nothing left
to reconcile.

The label volume (mri/aparc.atlas+aseg.nii.gz) is usually on the 256^3 conform
while the solve runs on the cropped grid. That mapping is done here, in one
place, rather than in each caller.

STATUS: validated on one subject, two hemispheres, against the FSR pial --
crossed_csf and median distance improve, self-intersections and end_in_wm get
worse. Not a default.
"""

import os

import numpy as np
import nibabel as nib

from . import outer_surface as osf
from . import wm_labels
from . import wm_surface
from .field_pial_prototype import (get_vox2ras_tkr, load_gm_wm_probability,
                                   make_transforms, rasterize_mesh, rasterize_mesh_pv)
from .retarget_surface import tkr_to_tkr
from .topology_gpu import correct_topology

SUPERSAMPLE = 3            # partial-volume rasterisation, as the white reconciliation uses
N_BANDS = 8                # priority bands for the ribbon correction
PROTECT_ABOVE = 0.6        # never sacrifice tissue the model is this sure about
PAD = 2


def hemisphere_ribbon(seg_labelled, df_labels, region, excluded):
    """The cortical ribbon for one hemisphere: WM fill plus the cortex parcels.

    The WM fill's labels are 'Left-*' / 'Right-*'; a parcellated aseg names the
    cortex per gyrus as 'lh-*' / 'rh-*'. Both are needed -- without the parcels
    this is the WM mask, not the ribbon.
    """
    side = 'Left' if region == 'lh' else 'Right'
    ids = list(wm_labels.hemisphere_labels(df_labels, side, excluded))
    skip = {'%s-%s' % (side, x) for x in excluded}
    ids += [i for n, i in df_labels['ID'].items()
            if n.startswith(region + '-') and n not in skip]
    return np.isin(np.asarray(seg_labelled), sorted(set(ids)))


def tissue_priority(gm_prob, wm_prob, ref_img, label_img):
    """Per-voxel 'how sure is the model this is tissue', on the LABEL grid.

    max(P_wm, P_ctx), a PROBABILITY in [0, 1]: load_gm_wm_probability already
    applies expit() to the per-label logits, so these are sigmoid outputs, not
    logits. On the cortical ribbon the distribution runs median 0.866, p25
    0.707, p10 0.543, p01 0.080.

    High in the interior of either tissue, low at the CSF boundary where both
    are weak. NOT P(WM|WM,cortex) -- that asks WHICH tissue, so on a GM+WM mask
    it ranks confident cortex as low confidence and the growth cuts straight
    through the ribbon (measured: cuts at the ribbon's own median confidence,
    i.e. no selectivity at all).

    Outside the reference image's extent the priority is set to the maximum, so
    those voxels are annexed first rather than sacrificed.
    """
    raw = np.maximum(np.asarray(wm_prob), np.asarray(gm_prob)).astype(np.float32)
    hi = float(raw.max())
    sh = tuple(label_img.shape[:3])
    if sh == tuple(ref_img.shape[:3]) and np.allclose(label_img.affine, ref_img.affine):
        return raw, hi
    from scipy.ndimage import map_coordinates
    g = np.meshgrid(*[np.arange(k) for k in sh], indexing='ij')
    idx = np.stack([g[0].ravel(), g[1].ravel(), g[2].ravel(), np.ones(g[0].size)])
    T = np.linalg.inv(ref_img.affine) @ label_img.affine    # label vox -> ref vox
    cc = (T @ idx)[:3]
    out = map_coordinates(raw, cc, order=1, mode='constant', cval=hi).reshape(sh)
    inb = (cc >= 0).all(0) & (cc <= (np.array(raw.shape) - 1)[:, None]).all(0)
    return np.where(inb.reshape(sh), out, hi).astype(np.float32), hi


def build_surface_segmentation(prep_dir, hemis=('lh', 'rh'), nsmooth=None,
                               n_bands=N_BANDS, supersample=SUPERSAMPLE,
                               correct_ribbon=True, gm_crisp=False,
                               protect_above=PROTECT_ABOVE, verbose=True):
    """Surfaces -> (seg, gmT, wmT) on the solve's grid, plus the WM surfaces.

    Returns a dict with seg/gmT/wmT/ref_img/tovox/totkr, 'surfaces' (the WM
    surfaces, in the cropped tkrRAS frame, ready to propagate), 'gm_surfaces',
    and 'found_csf' (voxels per hemisphere the correction called CSF).

    correct_ribbon=False meshes the raw ribbon instead, i.e. skips the
    sub-threshold CSF detection -- the control for it.

    protect_above is a confidence floor on the ribbon correction: tissue at or
    above it is never sacrificed, so the sulcus is only opened where the model
    was unsure. The topology is then not guaranteed genus 0 -- which does not
    matter here, since the GM surface is allowed defects. In the ribbon's units
    (max of the WM and cortex logits) the ribbon median is ~0.87 and the voxels
    the unprotected correction removes have median ~0.65.

    gm_crisp=True takes the GM occupancy as the voxel-centre-inside test rather
    than the partial-volume fraction. The PV fraction gives gmT a soft outer
    edge, so DiReCT's speed term is still non-zero half a voxel beyond the GM
    surface and the flow can push past it; the crisp mask stops exactly at the
    surface. WM stays partial-volume either way, so the change is isolated to
    the outer boundary.
    """
    nsmooth = wm_surface.NSMOOTH_DEFAULT if nsmooth is None else nsmooth
    gm, wm, ref_img = load_gm_wm_probability(prep_dir)
    tovox, totkr = make_transforms(ref_img)
    shape = tuple(ref_img.shape[:3])
    seg_lab, df, aff_lab, excluded = wm_surface.load_inputs(prep_dir)
    label_img = nib.load(os.path.join(prep_dir, 'mri', 'aparc.atlas+aseg.nii.gz'))
    prio, hi = tissue_priority(gm, wm, ref_img, label_img)
    # the label grid's tkrRAS -> the solve's tkrRAS, applied once, to both surfaces
    M = tkr_to_tkr(prep_dir, ref_img)
    to_ref = lambda v: (M[:3, :3] @ np.asarray(v).T).T + M[:3, 3]

    pv_wm = np.zeros(shape, np.float32)
    pv_gm = np.zeros(shape, np.float32)
    wm_surfaces, gm_surfaces, found = {}, {}, {}
    for h in ('lh', 'rh'):
        ribbon = hemisphere_ribbon(seg_lab, df, h, excluded)
        if correct_ribbon:
            corr, info = correct_topology(np.pad(ribbon, PAD), verbose=False, pair='26-6',
                                          priority=np.pad(prio, PAD, constant_values=hi),
                                          n_bands=n_bands, protect_above=protect_above)
            corr = corr[PAD:-PAD, PAD:-PAD, PAD:-PAD]
            found[h] = int((ribbon & ~corr).sum())
            if verbose:
                print('%s ribbon: %d voxels found as sub-threshold CSF (%.2f%%)'
                      % (h, found[h], 100 * found[h] / max(ribbon.sum(), 1)))
        else:
            corr, found[h] = ribbon, 0
        gv, gf = osf.mesh_envelope(corr, aff_lab, nsmooth=nsmooth)
        wv, wf = wm_surface.build_hemisphere(seg_lab, df, aff_lab, h, excluded,
                                             nsmooth=nsmooth, topology='gpu',
                                             priority=prio)
        gv, wv = to_ref(gv), to_ref(wv)
        gm_surfaces[h] = (gv, np.asarray(gf))
        wm_surfaces[h] = (wv, np.asarray(wf))
        pv_gm += (rasterize_mesh(tovox(gv), np.asarray(gf), shape).astype(np.float32)
                  if gm_crisp else
                  rasterize_mesh_pv(tovox(gv), np.asarray(gf), shape, supersample))
        pv_wm += rasterize_mesh_pv(tovox(wv), np.asarray(wf), shape, supersample)

    pv_gm = np.clip(pv_gm, 0.0, 1.0)
    pv_wm = np.clip(pv_wm, 0.0, 1.0)
    seg = np.where(pv_wm > 0.5, 3, np.where(pv_gm > 0.5, 2, 0)).astype(np.uint8)
    gmT = np.clip(pv_gm - pv_wm, 0.0, 1.0).astype(np.float32)
    wmT = pv_wm.astype(np.float32)
    if verbose:
        print('surface-derived segmentation (%s GM): WM %d, GM %d voxels'
              % ('crisp' if gm_crisp else 'PV',
                 int((seg == 3).sum()), int((seg == 2).sum())))
    return dict(seg=seg, gmT=gmT, wmT=wmT, ref_img=ref_img, tovox=tovox, totkr=totkr,
                surfaces={h: wm_surfaces[h] for h in hemis},
                gm_surfaces={h: gm_surfaces[h] for h in hemis},
                found_csf=found, pv_gm=pv_gm, pv_wm=pv_wm, prep_dir=prep_dir)
