"""Prototype: cortical thickness by eikonal wave propagation, as an
alternative to DiReCT's diffeomorphic WM->pial propagation.

STATUS: research prototype. Nothing else in this package imports it and it
is not wired into any pipeline script. Run it directly against an existing
DL+DiReCT-V2 prep or output directory and it writes a new set of volumes
alongside them. By default it consumes only the tissue probabilities that
directory already carries, and needs nothing beyond numpy/scipy/nibabel and
scikit-fmm; `--surface-pv` instead rasterises the pipeline's own surfaces as
partial volume, which is a better input and pulls in the pipeline's
environment (see load_surface_pv).

The idea
--------
DiReCT measures thickness by transporting the WM boundary outwards along a
velocity field and recording how far each voxel travelled. Here the same
correspondence is built with two first-arrival (eikonal) solves instead:

  T_wm   distance/travel time from the WM/GM interface, marching OUTWARD
         through grey matter.
  Sigma  the shock set of that first solve -- where characteristics of T_wm
         collide. Physically this is where two cortical banks meet with no
         resolvable CSF between them: a buried sulcus that the segmentation
         has closed. These voxels have to act as an outer boundary, or the
         wave from one bank walks straight into the other and thickness is
         reported as the sum of two cortices.
  T_out  distance/travel time from (pial boundary U Sigma), marching INWARD
         through grey matter.

  thickness = T_wm + T_out

The sum is exact only where the two solves share a characteristic, i.e.
where the streamline from the WM boundary through a voxel continues to the
pial boundary along the same path. That holds where the cortical normal is
near-straight -- most of a gyral wall -- and degrades at crowns and fundi,
where the two fronts arrive from different directions and T_wm + T_out
overestimates the true normal thickness. It is a first-arrival approximation
of the streamline length DiReCT integrates, not a drop-in equivalent.

Speed choices
-------------
--speed binary marches at speed 1, so both fields are plain Euclidean
distances in mm and the sum is a geometric length. --speed soft marches at
speed clip(p_GM, eps, 1), so the front is slowed where the segmentation is
unsure. That is closer in spirit to DiReCT, which is driven by the
probabilities rather than by a hard mask, but the resulting T is a travel
TIME, not a distance: it is >= the mm distance everywhere and equals it only
where p_GM == 1 along the whole path. Do not compare soft and binary numbers
to each other, or soft numbers to a mm thickness, without that caveat.

How large that caveat is, on IXI522 (`--surface-pv` off, model
posteriors): `--speed binary` gives a mean of 1.847 mm and `--speed soft`
2.393, a ratio of 1.30. It cannot be more than 1/gm_thr = 2x, because the
GM mask IS `p_GM > gm_thr`, so every voxel the front travels through has
speed at least gm_thr. For the same reason `eps` is inert -- 0.01, 0.1 and
0.5 all give 2.393 -- and it is what it looks like, a floor that stops a
degenerate divide, not a parameter with an opinion.

CORRECTION: an earlier version of this note reported soft means of 61.5 /
9.08 / 3.70 / 2.48 at eps 0.01 / 0.1 / 0.3 / 0.5 and concluded that eps
"is the parameter that decides the answer". That was a bug in _march, not
a property of the method: the speed OUTSIDE the marching domain still
reaches skfmm's solution, and leaving the background at the eps floor
slowed the front with cells it never travels through. Every soft number
from before that fix is void.

Verification
------------
Two checks, both reproducible from this file alone.

1. A synthetic phantom -- WM sphere of radius 20 mm, GM shell out to 23 mm,
   1 mm isotropic, so a known uniform thickness of 3.00 mm and no shock
   anywhere. With `--shock none` it recovers mean 2.787 mm (`--phi mask`)
   and 2.965 mm (`--phi prob`). Any Sigma the phantom reports is detector
   noise by construction; that is what drop_exposed was measured on.

2. T_out with no shock seeds is, by definition, the distance from each GM
   voxel to the outer GM boundary, which scipy's Euclidean distance
   transform computes independently. On IXI522, median(T_out - EDT) is
   -0.500 mm for `--phi mask` -- exactly the half voxel between the
   boundary and the nearest CSF voxel CENTRE that the EDT measures to, i.e.
   the solve is right -- and -0.874 mm for `--phi prob`, i.e. 0.37 mm short.

`prob` fails a second, sharper test on real data: it is not monotone in the
number of shock seeds. Adding seeds raised T_out for 117k GM voxels on
IXI522, which a first-arrival solve cannot do -- more seeds can only make
the front arrive sooner -- while `mask` is monotone including in the
maximum.

Feeding the solve genuine partial-volume maps (`--surface-pv`) does NOT
rescue it. On OAS30072's PV maps T_out p99.9 still goes 4.202 -> 4.268 ->
4.382 mm and the max 6.776 -> 7.066 as shock seeds are added, while `mask`
falls 8.634 -> 3.439 -> 2.647 as it must. Nor is the seed magnitude the
cause: seeding at -0.5 rather than -1.0, i.e. on the field's own scale,
moves the mean by 0.003 mm and leaves the tail exactly as non-monotone. The
defect is in how the solve handles a smooth phi carrying interior seeds,
not in the quality of the input.

`mask` is therefore the default, despite losing to `prob` on the phantom --
whose probability ramps are exactly linear, the ideal case for the
sub-voxel initialisation, where real posteriors are saturated. Treat
`--phi prob` as diagnostic only; it warns when combined with a non-empty
Sigma.

Against the pipeline's own numbers
----------------------------------
Eight OASIS cases from /data/disk2/oasis840 (`--surface-pv`, reusing
field_pial_sigma0.65's ?h.white; PV export 26.6 s/case, the eikonal solve
2.7 s), beside the same cases' field_pial_isowhite thickness CSVs. Means:

    eikonal --shock none    2.976      isowhite field       2.810
    eikonal --div-k 2.0     2.877      isowhite travel      3.205
    eikonal --div-k 1.0     2.521      isowhite nn          2.789
                                       isowhite sym_nn      2.834

These are DIFFERENT DEFINITIONS, not competing estimates of one quantity:
the eikonal column is a voxelwise mean over the GM ribbon, the isowhite
columns are unweighted means over parcels of vertex-based metrics, and the
package already documents that its own definitions separate by 0.12-0.70 mm
region-dependently. Read the table as a range check -- the eikonal numbers
land inside the spread of definitions the pipeline already produces -- not
as agreement or disagreement.

The shock threshold remains the dominant free parameter and is NOT
calibrated: k=2.0 -> k=1.0 moves the mean 0.36 mm. What the PV input does
change is how much gets flagged: Sigma_div at k=2.0 is 0.15% of GM on these
PV maps against 1.13% on IXI522's posteriors.

Scan-rescan, from the two run-01/run-02 pairs in that set: 0.006 and 0.027
mm on the eikonal side, against 0.006 and 0.042 mm for isowhite's `field`.

Propagating the surface (--propagate)
-------------------------------------
The unit gradient of T_wm is a flow field -- the direction the first-arrival
front travels -- so white-surface vertices can be integrated along it to the
outer boundary, which is the eikonal counterpart of what the pipeline does
with DiReCT's velocity field. No travel cap is needed: a vertex stops when
it crosses the tissue/CSF interface or enters Sigma, both refined by
bisection on the partial-volume field so the vertex lands on the 0.5
iso-surface sub-voxel.

It has two DIFFERENT defects, and one knob fixes only the first.

1. Runaway streamlines -- vertices that never stop, running the full
   100 mm integration cap. TWO things reduce them, and the stencil matters
   more than the smoothing. With Sigma and the thickness field HELD FIXED
   so only the gradient varies, OAS30072 lh:

       scheme   smooth   runaways   arclen-thick   vs the pipeline's pial
       central   0.0       2400        +0.424        0.904 / 1.471 mm
       central   1.0        417        +0.382        0.698 / 1.290
       upwind    0.0        217        +0.311        0.834 / 1.113
       upwind    1.0         14        +0.333        0.670 / 0.944

   Upwind with NO smoothing beats central WITH smoothing on every column,
   which is why --gradient defaults to upwind. --flow-smooth still helps
   (217 -> 14) and stays at 1.0, but it is no longer load-bearing: set it
   to 0 if you would rather not blur the shock structure, since the
   smoothing also shrinks Sigma.

   Both were selected on the runaway count, a pure failure mode needing no
   external reference. The same changes move the propagated surface closer
   to the pipeline's pial, but that was NOT the criterion -- tuning a
   method to agree with the method it is meant to be an alternative to
   would make the comparison worthless.

2. Overshoot. Excluding the runaways, the arc length a vertex travels
   exceeds the thickness T_wm + T_out at its own column -- by a median
   +0.424 mm with the central stencil, +0.311 upwind. Those two should
   agree if the streamline and the two solves share a characteristic.
   Upwind reduces it by about a quarter but does not remove it, and
   smoothing does not touch it at all. Still unexplained, and the open
   question in this module; do not read the propagated surface as if it
   realised the thickness map.

What it does not do
-------------------
Everything here is voxelwise. `--surface-pv` reads ?h.white in order to
rasterise it, but nothing in this module ever SAMPLES a voxel map at
surface vertices, deliberately: that needs the voxel/RAS frame handling
which lives elsewhere in this package, and getting it wrong silently biases
everything by half a voxel. Compare against the surface pipeline by way of
the thickness map, not the other way round.

Usage
-----
    python eikonal_thickness.py <input_dir> [<output_dir>] [options]

<input_dir> is a directory carrying the tissue probabilities in any of the
three layouts this package produces (see load_tissue_probs), or -- with
`--surface-pv` -- a prep with ?h.white from an earlier pipeline run.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import scipy.special as ss
from scipy import ndimage as ndi

try:
    import skfmm
except ImportError:  # pragma: no cover - dependency is prototype-only
    sys.stderr.write("eikonal_thickness needs scikit-fmm:  pip install scikit-fmm\n")
    raise


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def load_surface_pv(src, reuse_white='field_pial_sigma0.65', nsmooth=50,
                    topology='nighres', crop=True):
    """(p_gm, p_wm, ref_img) as PARTIAL-VOLUME occupancy, rasterised from the
    white and GM surfaces by the pipeline's `surface-pv` route.

    This is a better input than any of the probability layouts load_tissue_probs
    reads, and the reason is the boundary placement. A posterior is saturated:
    it is ~1 well inside the tissue and ~0 well outside, and the voxel straddling
    the boundary carries a value that reflects the model's confidence, not the
    fraction of the voxel that is inside. A PV occupancy carries exactly that
    fraction, so a voxel at 0.5 has the boundary through its centre. Measured on
    OAS30072: 26% of the WM voxels and 52% of the GM voxels are strictly
    fractional, against a posterior volume where the straddling band is
    effectively binary.

    Imports the pipeline lazily -- it needs pymeshlab, nighres and torch, which
    the rest of this module does not. Requires ?h.white from an earlier run of
    the pipeline on this prep (`reuse_white`), and the build parameters must
    match that run's white_build.json; the pipeline refuses a mismatch.
    """
    # This file lives inside the package, so import its siblings from the
    # WORKING TREE rather than from whatever `dldirect` happens to be
    # installed in site-packages -- an installed copy shadows the checkout and
    # the import fails outright (or, worse, silently runs the wrong code).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys.path[:1] != [repo_root]:
        sys.path.insert(0, repo_root)
    stale = [m for m in list(sys.modules)
             if m == 'dldirect' or m.startswith('dldirect.')]
    for m in stale:
        f = getattr(sys.modules[m], '__file__', '') or ''
        if not os.path.abspath(f).startswith(repo_root):
            del sys.modules[m]
    from dldirect import surface_seg, pial_pipeline
    reused = pial_pipeline._load_white_for_reuse(
        reuse_white, src, hemis=('lh', 'rh'), nsmooth=nsmooth,
        topology=topology, crop=crop, verbose=False)
    sd = surface_seg.build_surface_segmentation(
        src, hemis=('lh', 'rh'), nsmooth=nsmooth, crop=crop, topology=topology,
        wm_surfaces=reused, correct_ribbon=True, verbose=False)
    gm = np.clip(np.asarray(sd['gmT'], np.float32), 0, 1)
    wm = np.clip(np.asarray(sd['wmT'], np.float32), 0, 1)
    print('  probabilities: surface-pv partial volume (reusing %s/?h.white)'
          % reuse_white)
    for n, v in (('gmT', gm), ('wmT', wm)):
        frac = int(((v > 0) & (v < 1)).sum())
        print('    %s: %d voxels >0.5, %d strictly fractional (%.0f%%)'
              % (n, int((v > 0.5).sum()), frac,
                 100.0 * frac / max(1, int((v > 0).sum()))))
    return gm, wm, sd['ref_img']


def load_tissue_probs(src, gm_labels=None, wm_labels=None):
    """(p_gm, p_wm, ref_img) from an existing prep or DL+DiReCT output dir.

    Three layouts are recognised, in this order:

    1. prob_gm.nii.gz / prob_wm.nii.gz -- a prep built by prep_from_seg.py
       from a segmenter that emits posteriors (SAMSEG, SynthSeg). These are
       already probabilities; take them as they are.
    2. gmprob.nii.gz / wmprob.nii.gz -- what DiReCT.py writes BEFORE its
       thresholding step. Also already probabilities.
    3. seg_<label>.nii.gz -- DeepSCAN's per-label logits, combined the way
       DiReCT.py combines them (max over labels, then expit, with logit == 0
       held at probability 0 so the background stays background).

    The label lists default to cortex and cerebral white matter only.
    DiReCT.py's own defaults also pull in amygdala and hippocampus; this
    prototype leaves them out, because a wave seeded on the subcortical
    grey/white interface has nothing resembling a pial boundary to stop at.
    """
    p_gm = os.path.join(src, 'prob_gm.nii.gz')
    p_wm = os.path.join(src, 'prob_wm.nii.gz')
    if gm_labels is None and wm_labels is None \
            and os.path.exists(p_gm) and os.path.exists(p_wm):
        gm_img = nib.load(p_gm)
        gm = np.clip(gm_img.get_fdata(dtype=np.float32), 0, 1)
        wm = np.clip(nib.load(p_wm).get_fdata(dtype=np.float32), 0, 1)
        print('  probabilities: prob_gm/prob_wm.nii.gz (posteriors)')
        return gm.astype(np.float32), wm.astype(np.float32), gm_img

    p_gm = os.path.join(src, 'gmprob.nii.gz')
    p_wm = os.path.join(src, 'wmprob.nii.gz')
    if gm_labels is None and wm_labels is None \
            and os.path.exists(p_gm) and os.path.exists(p_wm):
        gm_img = nib.load(p_gm)
        gm = np.clip(gm_img.get_fdata(dtype=np.float32), 0, 1)
        wm = np.clip(nib.load(p_wm).get_fdata(dtype=np.float32), 0, 1)
        print('  probabilities: gmprob/wmprob.nii.gz (DiReCT.py, pre-threshold)')
        return gm.astype(np.float32), wm.astype(np.float32), gm_img

    if gm_labels is None:
        gm_labels = ['Left-Cerebral-Cortex', 'Right-Cerebral-Cortex']
    if wm_labels is None:
        wm_labels = ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter']
        if os.path.exists(os.path.join(src, 'seg_WM-hypointensities.nii.gz')):
            wm_labels = wm_labels + ['WM-hypointensities']

    def _load(label):
        path = os.path.join(src, 'seg_%s.nii.gz' % label)
        if not os.path.exists(path):
            raise SystemExit('no tissue probabilities in %s (looked for '
                             'prob_gm/prob_wm, gmprob/wmprob, and %s)'
                             % (src, os.path.basename(path)))
        return nib.load(path).get_fdata(dtype=np.float32)

    ref_img = nib.load(os.path.join(src, 'seg_%s.nii.gz' % gm_labels[0]))
    gm_logit = np.max(np.stack([_load(l) for l in gm_labels]), axis=0)
    wm_logit = np.max(np.stack([_load(l) for l in wm_labels]), axis=0)
    gm = np.where(gm_logit == 0, 0, ss.expit(gm_logit))
    wm = np.where(wm_logit == 0, 0, ss.expit(wm_logit))
    print('  probabilities: seg_<label>.nii.gz logits, GM %s, WM %s'
          % ('+'.join(gm_labels), '+'.join(wm_labels)))
    return gm.astype(np.float32), wm.astype(np.float32), ref_img


def save_img(arr, path, ref_img):
    """Write a volume the way DiReCT.py's save_img does (reference affine, mm
    units), so these outputs are drop-in comparable with the shipped ones."""
    img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), ref_img.affine)
    img.header['xyzt_units'] = 2  # mm
    nib.save(img, path)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def build_masks(p_gm, p_wm, gm_thr, wm_thr):
    """WM and GM masks, with WM winning any overlap.

    A voxel over threshold in both would otherwise be inside the domain and
    inside the seed at once, which leaves the zero level set undefined there.
    WM wins because the WM/GM interface is the thing being seeded: a voxel
    that is confidently white matter must be on the interior side of it.
    """
    wm = p_wm > wm_thr
    gm = (p_gm > gm_thr) & ~wm
    both = (p_wm > wm_thr) & (p_gm > gm_thr)
    print('  masks: WM %d, GM %d voxels (%d over both thresholds, assigned WM)'
          % (int(wm.sum()), int(gm.sum()), int(both.sum())))
    if not wm.any() or not gm.any():
        raise SystemExit('empty WM or GM mask -- check --gm-thr / --wm-thr')
    return gm, wm


# ---------------------------------------------------------------------------
# the two eikonal solves
# ---------------------------------------------------------------------------

def _march(phi, mask, speed, spacing, soft):
    """One first-arrival solve on the unmasked part of the domain.

    Returns (T, unreachable), where T is non-negative and `unreachable` marks
    voxels the front never reached (a GM island with no route to a seed).
    Both skfmm entry points return a masked array; travel_time additionally
    masks whatever it could not reach.
    """
    phi_m = np.ma.MaskedArray(np.asarray(phi, dtype=np.float64), mask)
    if soft:
        # The speed OUTSIDE the marching domain still reaches the solution --
        # masking phi does not stop skfmm reading those cells' speed. Leaving
        # the background at the eps floor made T_wm over GM read 42.647 mm on
        # IXI522 where the correct answer is 1.097, because the front was
        # being slowed by cells it never travels through. Neutralise them:
        # the domain's own speeds are the only ones that should matter, and
        # inside the domain speed is >= gm_thr by construction, so a soft
        # solve can never exceed 1/gm_thr times the binary one.
        sp = np.asarray(speed, dtype=np.float64).copy()
        sp[mask] = 1.0
        out = skfmm.travel_time(phi_m, sp, dx=spacing)
    else:
        out = skfmm.distance(phi_m, dx=spacing)
    filled = np.abs(np.ma.filled(out, 0.0))
    unreachable = np.ma.getmaskarray(out) & ~mask
    unreachable |= ~np.isfinite(filled)
    filled[~np.isfinite(filled)] = 0.0
    return filled.astype(np.float32), unreachable


def solve_t_wm(p_gm, p_wm, gm, wm, spacing, args):
    """T_wm: outward from the WM/GM interface, through grey matter.

    The domain is GM U WM: the white matter has to stay in it, because the
    zero level set IS its boundary and skfmm needs phi to change sign. Only
    the GM values are used afterwards. Everything outside (CSF, background,
    the other tissues) is masked, so the front cannot walk around the cortex
    through the subarachnoid space.
    """
    domain = gm | wm
    mask = ~domain

    if args.phi == 'prob':
        # Sub-voxel interface: the zero crossing sits where p_WM crosses the
        # threshold, rather than on the half-voxel step of the mask boundary.
        phi = (p_wm.astype(np.float64) - args.wm_thr)
        # Force the sign to agree with the masks in the (rare) voxels where
        # the two disagree -- see build_masks.
        phi[wm] = np.minimum(phi[wm], -1e-3)
        phi[gm] = np.maximum(phi[gm], 1e-3)
    else:
        phi = np.where(wm, -1.0, 1.0)

    speed = None
    if args.speed == 'soft':
        speed = np.clip(p_gm.astype(np.float64), args.eps, 1.0)
        speed[wm] = 1.0  # interior of the seed; its values are discarded

    T, unreach = _march(phi, mask, speed, spacing, args.speed == 'soft')
    n_bad = int((unreach & gm).sum())
    if n_bad:
        print('  T_wm: %d GM voxels unreachable from the WM interface' % n_bad)
    return T, unreach


def solve_t_out(p_gm, gm, wm, sigma, spacing, args):
    """T_out: inward from (pial boundary U Sigma), through grey matter.

    The pial boundary is represented by a one-voxel shell of non-tissue
    immediately outside GM, held at phi < 0; the shock set Sigma is pushed to
    phi < 0 as well, which makes it behave as an additional piece of outer
    boundary half a voxel inside the cortex. White matter is masked out
    entirely here, so a front cannot cut across the gyral stem from one bank
    to the other.
    """
    outside = ~gm & ~wm
    shell = ndi.binary_dilation(gm, ndi.generate_binary_structure(3, 1)) \
        & outside
    domain = gm | shell
    mask = ~domain

    if args.phi == 'prob':
        phi = (p_gm.astype(np.float64) - args.gm_thr)
        phi[gm] = np.maximum(phi[gm], 1e-3)
        phi[shell] = np.minimum(phi[shell], -1e-3)
    else:
        phi = np.where(shell, -1.0, 1.0)
    if sigma is not None and sigma.any():
        phi[sigma] = -1.0

    speed = None
    if args.speed == 'soft':
        speed = np.clip(p_gm.astype(np.float64), args.eps, 1.0)
        speed[shell] = 1.0

    if not (phi[domain] < 0).any():
        raise SystemExit('T_out has no seed: GM has no exposed outer boundary')

    T, unreach = _march(phi, mask, speed, spacing, args.speed == 'soft')

    # Sigma is a seed region with a thickness of its own, and it lies INSIDE
    # the read-out domain (the WM and the shell do not). skfmm returns a
    # signed distance, so a voxel in the middle of a large shock blob comes
    # back with its distance to that blob's own surface -- which is an
    # arrival time measured from the wrong thing, and grows with the size of
    # the blob. Left in, it makes T_out INCREASE as shock seeds are added,
    # which is impossible for a first-arrival solve. A seed has arrival 0.
    if sigma is not None and sigma.any():
        T = T.copy()
        T[sigma] = 0.0

    n_bad = int((unreach & gm).sum())
    if n_bad:
        print('  T_out: %d GM voxels unreachable from the pial boundary/Sigma'
              % n_bad)
    print('  T_out seeds: %d shell voxels + %d Sigma voxels'
          % (int(shell.sum()), 0 if sigma is None else int(sigma.sum())))
    return T, unreach


# ---------------------------------------------------------------------------
# the shock set
# ---------------------------------------------------------------------------

_OFFSETS_26 = [(i, j, k)
               for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
               if (i, j, k) != (0, 0, 0)]


def _shift(a, off):
    """a shifted by `off` voxels, zero-padded (no wraparound)."""
    out = np.zeros_like(a)
    src = tuple(slice(max(0, -o), a.shape[i] - max(0, o))
                for i, o in enumerate(off))
    dst = tuple(slice(max(0, o), a.shape[i] - max(0, -o))
                for i, o in enumerate(off))
    out[dst] = a[src]
    return out


def upwind_gradient(field, valid, spacing):
    """Godunov upwind gradient of an arrival-time field.

    A central difference is the wrong stencil for a first-arrival time. At a
    shock both neighbours along an axis have SMALLER T, so differencing
    across the centre averages two opposing characteristics into a direction
    that belongs to neither -- and the shock set is exactly where this module
    needs the direction to be meaningful.

    The upwind rule differences only towards where the front came FROM. Per
    axis, with the backward and forward differences

        D- = (T[i] - T[i-1]) / h      D+ = (T[i+1] - T[i]) / h

    take the steeper descent, max(D-, -D+, 0), and give it the matching sign.
    At a smooth point that reduces to the usual one-sided upwind difference;
    at a shock it picks ONE of the two arriving characteristics instead of
    their average; at a local minimum it returns zero, which is correct --
    that is a source, and it has no single direction.

    `valid` marks the cells that may be differenced against, so the stencil
    never reaches outside the marching domain.
    """
    g = np.zeros((3,) + tuple(field.shape), np.float32)
    for a in range(3):
        h = float(spacing[a])
        off_m = tuple(1 if k == a else 0 for k in range(3))   # brings i-1 to i
        off_p = tuple(-1 if k == a else 0 for k in range(3))  # brings i+1 to i
        Tm, Tp = _shift(field, off_m), _shift(field, off_p)
        vm, vp = _shift(valid, off_m), _shift(valid, off_p)
        # candidate magnitudes for a gradient pointing along +a and -a.
        # -inf where the neighbour is outside the domain, so that side is
        # never selected and no infinity reaches the arithmetic below.
        cand_pos = np.where(vm, (field - Tm) / h, -np.inf)
        cand_neg = np.where(vp, (field - Tp) / h, -np.inf)
        use_pos = (cand_pos >= cand_neg) & (cand_pos > 0)
        use_neg = (~use_pos) & (cand_neg > 0)
        ga = np.zeros(field.shape, np.float32)
        ga[use_pos] = cand_pos[use_pos]
        ga[use_neg] = -cand_neg[use_neg]
        g[a] = ga
    return g


def characteristic_directions(T, gm, wm, spacing, smooth=0.0,
                              scheme='upwind'):
    """Unit gradient of T_wm: the direction its characteristics travel.

    Two preparations before differencing, both to keep the central difference
    at the edges of the domain from reading a cliff that is not there:

    * the WM side is filled with -T, so the field is monotone and continuous
      across the interface it was seeded on (in `soft` mode T is a travel
      time and is positive on both sides, which would otherwise put a spurious
      minimum exactly on the boundary);
    * everything outside GM U WM is filled with its nearest in-domain value,
      so the outermost GM voxels difference against a plausible continuation
      rather than against zero.
    """
    field = np.where(wm, -T, T).astype(np.float64)
    domain = gm | wm
    if not domain.all():
        _, idx = ndi.distance_transform_edt(~domain, sampling=spacing,
                                            return_indices=True)
        field[~domain] = field[tuple(i[~domain] for i in idx)]

    if smooth > 0:
        # The gradient of a distance field is a badly-conditioned object here:
        # this package measured a 1e-6 grid perturbation flipping 4688 normals
        # by 90 degrees and moving a patch 3.94 mm. Smoothing T before
        # differencing is the cheap mitigation; `smooth` is in mm.
        field = ndi.gaussian_filter(
            field, sigma=[smooth / sp for sp in spacing], mode='nearest')

    if scheme == 'upwind':
        g = upwind_gradient(field, domain, spacing)
    else:
        g = np.stack(np.gradient(field, *spacing)).astype(np.float32)
    mag = np.sqrt((g * g).sum(axis=0))
    unit = np.zeros_like(g)
    ok = mag > 1e-6
    np.divide(g, mag[None], out=unit, where=ok[None])
    return unit, mag


def shock_by_divergence(unit, gm, spacing, k):
    """Sigma_div: GM voxels where the characteristic field converges.

    div(n) is the rate at which neighbouring characteristics approach each
    other. It is strongly negative where two fronts are about to collide --
    i.e. where a sulcus has closed -- and mildly negative anywhere the front
    is concave, which is why the threshold is -k rather than 0.
    """
    d = np.gradient(unit[0], spacing[0], axis=0) \
        + np.gradient(unit[1], spacing[1], axis=1) \
        + np.gradient(unit[2], spacing[2], axis=2)
    return (gm & (d < -k)).astype(bool), d.astype(np.float32)


def shock_by_reversal(unit, gm, k_cos, converging):
    """Sigma_rev: GM voxels with a 26-neighbour whose characteristic points
    the opposite way (cosine < k_cos, default 0).

    Both members of an opposing pair are flagged, since all 26 offsets are
    swept. With --converging the pair must also be closing on each other --
    n_i . d > 0 > n_j . d for the offset d from i to j -- which rejects the
    case of two characteristics that diverge back-to-back, as they do on a
    gyral crown, where the cosine is equally negative but nothing collides.
    """
    flag = np.zeros(gm.shape, dtype=bool)
    for off in _OFFSETS_26:
        valid = gm & _shift(gm, off)
        if not valid.any():
            continue
        cos = sum(unit[c] * _shift(unit[c], off) for c in range(3))
        hit = valid & (cos < k_cos)
        if converging:
            d = np.asarray(off, dtype=np.float32)
            d = d / np.linalg.norm(d)
            along_i = sum(unit[c] * d[c] for c in range(3))
            along_j = sum(_shift(unit[c], off) * d[c] for c in range(3))
            hit &= (along_i > 0) & (along_j < 0)
        flag |= hit
    return flag


def exposed_rim(gm, wm):
    """GM voxels that already touch non-tissue, i.e. that sit on a pial
    boundary the segmentation did resolve."""
    outside = ~gm & ~wm
    return gm & ndi.binary_dilation(outside,
                                    ndi.generate_binary_structure(3, 1))


def drop_exposed(name, sigma, rim):
    """Remove the exposed rim from a shock set.

    Two reasons. It is redundant: those voxels are already T_out seeds via
    the shell, so keeping them changes nothing about the boundary condition.
    And it is where the shock detector is least trustworthy -- the
    characteristic directions there are computed partly from the
    nearest-value fill outside the domain (see characteristic_directions),
    which flattens the field outward and bends the unit gradient inward.
    On the synthetic sphere phantom, which has no shock anywhere, that
    artifact alone flagged 6.2% of GM, all of it on the rim, and pulled the
    recovered thickness from 2.97 mm to 2.66 mm against a true 3.00 mm.
    """
    n = int(sigma.sum())
    out = sigma & ~rim
    print('  %s: dropped %d of %d voxels on the exposed GM rim' %
          (name, n - int(out.sum()), n))
    return out


def report_sigma(name, sigma, gm):
    if sigma is None:
        return
    n = int(sigma.sum())
    if n == 0:
        print('  %s: empty' % name)
        return
    _, n_cc = ndi.label(sigma, ndi.generate_binary_structure(3, 3))
    print('  %s: %d voxels (%.2f%% of GM), %d connected components'
          % (name, n, 100.0 * n / max(1, int(gm.sum())), n_cc))


# ---------------------------------------------------------------------------
# propagating the white surface along the flow
# ---------------------------------------------------------------------------

def get_vox2ras_tkr(img):
    """FreeSurfer-style tkrRAS transform, built from an image's own header.

    Copied verbatim from field_pial_prototype (which copied it from
    dl_wm_surface_parallel_dev) so this module builds the SAME frame the
    pipeline's surfaces live in. Re-deriving it is how a half-voxel -- or a
    5.5 mm -- shift gets in.
    """
    ds = img.header._structarr['pixdim'][1:4]
    ns = img.header._structarr['dim'][1:4] * ds / 2.0
    return np.array([[-ds[0], 0, 0, ns[0]],
                     [0, 0, ds[2], -ns[2]],
                     [0, -ds[1], 0, ns[1]],
                     [0, 0, 0, 1]], dtype=np.float64)


def make_transforms(ref_img):
    A = get_vox2ras_tkr(ref_img)
    Ainv = np.linalg.inv(A)
    return (lambda p: nib.affines.apply_affine(Ainv, p),
            lambda q: nib.affines.apply_affine(A, q))


def _sample_scalar(vol, pos_vox):
    return ndi.map_coordinates(vol, pos_vox.T, order=1, mode='nearest')


def _sample_vec(unit, pos_vox):
    """Trilinear sample of the (3, D, H, W) direction field, renormalised.

    Interpolating unit vectors does not give a unit vector -- near a shock,
    where neighbours point opposite ways, the interpolant can be near zero.
    Renormalise, and report the ones that collapsed rather than dividing by
    a number that means the direction is undefined there.
    """
    v = np.stack([ndi.map_coordinates(unit[k], pos_vox.T, order=1,
                                      mode='nearest') for k in range(3)], 1)
    n = np.linalg.norm(v, axis=1)
    ok = n > 1e-6
    v[ok] /= n[ok, None]
    return v, ok


def propagate_surface(verts_vox, unit, tissue_pv, sigma, spacing, step=0.25,
                      max_steps=400):
    """Integrate vertices along the flow field to the outer boundary.

    This is the eikonal counterpart of what the pipeline does with DiReCT's
    velocity field: march each white-surface vertex along the direction the
    first-arrival front travels, and stop it where the front stops.

    Stopping needs no travel cap, unlike a fixed number of integration
    rounds. The front's own terminal set IS the stopping condition: a vertex
    halts when it crosses the tissue/CSF interface, or when it enters the
    shock set Sigma, where the characteristic it was following ends because
    another front arrived first. Both crossings are refined by bisection on
    the PARTIAL-VOLUME field rather than snapped to a voxel, so the final
    vertex sits on the 0.5 iso-surface sub-voxel.

    Returns (verts_vox, arclen_mm, reason) with reason in
    {0: still moving at max_steps, 1: left the tissue, 2: hit Sigma,
     3: flow direction collapsed}.
    """
    pos = np.asarray(verts_vox, float).copy()
    sp = np.asarray(spacing, float)
    sig_f = sigma.astype(np.float32)
    n = pos.shape[0]
    arclen = np.zeros(n)
    reason = np.zeros(n, np.int8)
    alive = np.ones(n, bool)

    def outside(x):
        return _sample_scalar(tissue_pv, x) < 0.5

    def in_shock(x):
        return _sample_scalar(sig_f, x) > 0.5

    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        d, ok = _sample_vec(unit, pos[idx])
        dead = idx[~ok]
        if dead.size:
            alive[dead] = False
            reason[dead] = 3
        idx = idx[ok]
        if not idx.size:
            continue
        nxt = pos[idx] + step * d[ok] / sp[None]

        stop_out = outside(nxt)
        stop_sig = in_shock(nxt) & ~stop_out
        moved = np.full(idx.size, step)

        for mask, code, fn in ((stop_out, 1, lambda x: _sample_scalar(tissue_pv, x) - 0.5),
                               (stop_sig, 2, lambda x: 0.5 - _sample_scalar(sig_f, x))):
            if not mask.any():
                continue
            lo = pos[idx[mask]]          # still valid
            hi = nxt[mask]               # already past
            for _ in range(8):           # bisection on the PV field
                mid = 0.5 * (lo + hi)
                past = fn(mid) < 0
                hi = np.where(past[:, None], mid, hi)
                lo = np.where(past[:, None], lo, mid)
            fin = 0.5 * (lo + hi)
            moved[mask] = np.linalg.norm((fin - pos[idx[mask]]) * sp[None], axis=1)
            nxt[mask] = fin
            reason[idx[mask]] = code

        pos[idx] = nxt
        arclen[idx] += moved
        alive[idx[stop_out | stop_sig]] = False

    return pos, arclen, reason


def propagate_and_write(out_dir, surf_dir, ref_img, unit, tissue_pv, sigma,
                        spacing, hemis=('lh', 'rh'), step=0.25, prefix='',
                        compare_dir=None):
    fsio = nib.freesurfer.io
    tovox, totkr = make_transforms(ref_img)
    written = []
    for h in hemis:
        wp = os.path.join(surf_dir, '%s.white' % h)
        if not os.path.exists(wp):
            print('  %s: no %s, skipping' % (h, wp))
            continue
        w, f = fsio.read_geometry(wp)
        pos, arclen, reason = propagate_surface(
            tovox(np.asarray(w, float)), unit, tissue_pv, sigma, spacing,
            step=step)
        out = totkr(pos)
        disp = np.linalg.norm(out - np.asarray(w, float), axis=1)
        counts = {c: int((reason == c).sum()) for c in (0, 1, 2, 3)}
        print('  %s: %d vertices | stopped at tissue %d (%.1f%%), at Sigma %d '
              '(%.1f%%), flow collapsed %d, still moving at cap %d'
              % (h, len(w), counts[1], 100.0 * counts[1] / len(w), counts[2],
                 100.0 * counts[2] / len(w), counts[3], counts[0]))
        print('     straight-line displacement  mean %.3f  median %.3f  '
              'p95 %.3f mm;  arc length mean %.3f mm'
              % (disp.mean(), np.median(disp), np.percentile(disp, 95),
                 arclen.mean()))
        if compare_dir:
            cp = os.path.join(compare_dir, '%s.pial' % h)
            if os.path.exists(cp):
                other, _ = fsio.read_geometry(cp)
                if other.shape == out.shape:
                    gap = np.linalg.norm(out - np.asarray(other, float), axis=1)
                    print('     vs %s/%s.pial (same vertices): mean %.3f  '
                          'median %.3f  p95 %.3f mm'
                          % (os.path.basename(compare_dir), h, gap.mean(),
                             np.median(gap), np.percentile(gap, 95)))
        path = os.path.join(out_dir, '%s%s.pial_eikonal' % (prefix, h))
        fsio.write_geometry(path, out.astype(np.float32), f, create_stamp=None,
                            volume_info=None)
        written.append(path)
        print('     wrote %s' % path)
    return written


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def summarise(name, arr, gm):
    v = arr[gm]
    v = v[np.isfinite(v)]
    if v.size == 0:
        print('  %-10s empty' % name)
        return
    print('  %-10s mean %.3f  median %.3f  p95 %.3f  p99.9 %.3f  max %.3f'
          % (name, v.mean(), np.median(v), np.percentile(v, 95),
             np.percentile(v, 99.9), v.max()))


def apply_gm_source(p_gm, p_wm, ref_img, src, source, gm_labels, wm_labels):
    """Optionally swap the GM channel for the segmentation model's own output.

    The PV maps from `--surface-pv` are a rasterisation of a MESH, so their
    outer boundary is wherever the ribbon was meshed -- a hard geometric
    surface, and a binary decision already taken. The model's posterior is
    not: it falls off gradually where the model is unsure there is cortex.
    Taking gmT from the model therefore makes both the outer boundary AND
    (under --speed soft) the front's speed follow the model's confidence
    rather than a committed surface, while the WM channel stays on the PV
    rasterisation so the seed is still the shared, sub-voxel white surface.

    Mixing sources is only valid because both live on the prep's own grid;
    that is asserted, not assumed.
    """
    if source != 'model':
        return p_gm, p_wm
    m_gm, m_wm, m_ref = load_tissue_probs(src, gm_labels, wm_labels)
    if tuple(m_ref.shape[:3]) != tuple(ref_img.shape[:3]):
        raise SystemExit('--gm-source model: the model output is %s but the '
                         'solve grid is %s' % (m_ref.shape[:3],
                                               ref_img.shape[:3]))
    print('  gmT: taken from the model output, not the PV rasterisation '
          '(%d voxels >0.5, was %d)'
          % (int((m_gm > 0.5).sum()), int((p_gm > 0.5).sum())))
    return m_gm.astype(np.float32), p_wm


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('Usage')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_dir',
                    help='prep or DL+DiReCT output dir carrying the tissue '
                         'probabilities')
    ap.add_argument('output_dir', nargs='?', default=None,
                    help='where to write (default: input_dir)')
    ap.add_argument('--prefix', default='eikonal_',
                    help='filename prefix for the outputs (default eikonal_)')

    ap.add_argument('--gm-thr', type=float, default=0.5,
                    help='GM mask threshold on p_GM (default 0.5)')
    ap.add_argument('--wm-thr', type=float, default=0.5,
                    help='WM mask threshold on p_WM (default 0.5)')
    ap.add_argument('--speed', choices=('binary', 'soft'), default='binary',
                    help='binary: speed 1, T is a distance in mm. '
                         'soft: speed clip(p_GM, eps, 1), T is a travel time '
                         '(default binary)')
    ap.add_argument('--eps', type=float, default=1e-2,
                    help='floor on the soft speed, so the front cannot stall '
                         '(default 0.01)')
    ap.add_argument('--phi', choices=('mask', 'prob'), default='mask',
                    help='where the zero level sets sit: on the mask boundary, '
                         'or sub-voxel at the probability threshold (default '
                         'mask; prob is measurably wrong on real data -- see '
                         'the module docstring)')

    ap.add_argument('--shock', choices=('div', 'rev', 'both', 'none'),
                    default='div',
                    help='which shock set seeds T_out (default div). Both '
                         'variants are always computed and written.')
    ap.add_argument('--div-k', type=float, default=1.0,
                    help='Sigma_div threshold: flag div(n) < -k, per mm '
                         '(default 1.0)')
    ap.add_argument('--cos-thr', type=float, default=0.0,
                    help='Sigma_rev threshold on the neighbour cosine '
                         '(default 0.0)')
    ap.add_argument('--keep-exposed-shocks', action='store_true',
                    help='keep shock voxels that already touch non-tissue; '
                         'they are redundant as T_out seeds and are where the '
                         'detector is least reliable (see drop_exposed)')
    ap.add_argument('--converging', action='store_true',
                    help='Sigma_rev additionally requires the opposing pair to '
                         'be closing on each other')

    ap.add_argument('--gradient', choices=('upwind', 'central'),
                    default='upwind',
                    help='stencil for differencing T_wm into the flow field '
                         'and the divergence. Default upwind (Godunov); '
                         "'central' is np.gradient, which is what this "
                         'module used before and is the wrong stencil at a '
                         'shock. See upwind_gradient.')
    ap.add_argument('--gm-source', choices=('same', 'model'), default='same',
                    help="'model' replaces the GM channel with the "
                         "segmentation model's own posterior, keeping the WM "
                         'channel as loaded. Only meaningful with '
                         '--surface-pv, where the two then differ. See '
                         'apply_gm_source.')
    ap.add_argument('--propagate', action='store_true',
                    help='integrate the white surface along the flow field '
                         '(the unit gradient of T_wm) to the outer boundary, '
                         'and write ?h.pial_eikonal. This is the eikonal '
                         "counterpart of the pipeline's velocity-field "
                         'propagation. See propagate_surface.')
    ap.add_argument('--surf-dir', default=None,
                    help='--propagate: directory holding ?h.white (default: '
                         'the --reuse-white directory)')
    ap.add_argument('--compare-pial', default=None,
                    help='--propagate: a run directory whose ?h.pial shares '
                         'these vertices, reported as a paired distance '
                         '(e.g. field_pial_isowhite)')
    ap.add_argument('--step', type=float, default=0.25,
                    help='--propagate: integration step in mm (default '
                         '%(default)s)')
    ap.add_argument('--flow-smooth', type=float, default=1.0,
                    help='smooth T_wm by this sigma in mm before differencing '
                         'it into the flow field (default %(default)s, 0 = '
                         'off). The gradient of a distance field is badly '
                         'conditioned; see the module docstring for the sweep '
                         'that set this.')
    ap.add_argument('--surface-pv', action='store_true',
                    help="build the WM/GM maps as partial-volume occupancy from "
                         "the pipeline's surfaces instead of reading "
                         "probabilities off disk. Needs ?h.white from an "
                         "earlier run (--reuse-white) and the pipeline's own "
                         "environment. See load_surface_pv.")
    ap.add_argument('--reuse-white', default='field_pial_sigma0.65',
                    help='--surface-pv only: the run directory (inside '
                         'input_dir, or an absolute path) holding ?h.white '
                         '(default %(default)s)')
    ap.add_argument('--gm-labels', nargs='+', default=None,
                    help='override the GM label list (logit layout only)')
    ap.add_argument('--wm-labels', nargs='+', default=None,
                    help='override the WM label list (logit layout only)')
    args = ap.parse_args(argv)

    src = args.input_dir
    dst = args.output_dir or src
    if not os.path.isdir(dst):
        os.makedirs(dst)

    print('input:  %s' % src)
    if args.surface_pv:
        p_gm, p_wm, ref_img = load_surface_pv(src, args.reuse_white)
    else:
        p_gm, p_wm, ref_img = load_tissue_probs(src, args.gm_labels,
                                                args.wm_labels)
    p_gm, p_wm = apply_gm_source(p_gm, p_wm, ref_img, src, args.gm_source,
                                 args.gm_labels, args.wm_labels)
    spacing = tuple(float(z) for z in ref_img.header.get_zooms()[:3])
    print('  grid %s, spacing %s mm' % (p_gm.shape, spacing))

    gm, wm = build_masks(p_gm, p_wm, args.gm_thr, args.wm_thr)

    print('T_wm (outward from the WM/GM interface, speed=%s, phi=%s)'
          % (args.speed, args.phi))
    t_wm, _ = solve_t_wm(p_gm, p_wm, gm, wm, spacing, args)

    print('shock set')
    unit, _ = characteristic_directions(t_wm, gm, wm, spacing,
                                        smooth=args.flow_smooth,
                                        scheme=args.gradient)
    sigma_div, div = shock_by_divergence(unit, gm, spacing, args.div_k)
    sigma_rev = shock_by_reversal(unit, gm, args.cos_thr, args.converging)
    if not args.keep_exposed_shocks:
        rim = exposed_rim(gm, wm)
        sigma_div = drop_exposed('Sigma_div', sigma_div, rim)
        sigma_rev = drop_exposed('Sigma_rev', sigma_rev, rim)
    report_sigma('Sigma_div', sigma_div, gm)
    report_sigma('Sigma_rev', sigma_rev, gm)
    overlap = int((sigma_div & sigma_rev).sum())
    print('  overlap: %d voxels (%.0f%% of div, %.0f%% of rev)'
          % (overlap,
             100.0 * overlap / max(1, int(sigma_div.sum())),
             100.0 * overlap / max(1, int(sigma_rev.sum()))))

    sigma = {'div': sigma_div,
             'rev': sigma_rev,
             'both': sigma_div | sigma_rev,
             'none': None}[args.shock]

    if args.phi == 'prob' and sigma is not None and sigma.any():
        print('  WARNING: --phi prob with shock seeds is not monotone in the '
              'seed set (see the module docstring). Use --phi mask.')
    print('T_out (inward from the pial boundary U Sigma=%s)' % args.shock)
    t_out, _ = solve_t_out(p_gm, gm, wm, sigma, spacing, args)

    # The sum is the streamline length only where the two solves agree on the
    # characteristic through the voxel -- see the module docstring.
    thickness = np.where(gm, t_wm + t_out, 0.0).astype(np.float32)

    print('statistics over %d GM voxels (%s)'
          % (int(gm.sum()), 'mm' if args.speed == 'binary' else 'travel time'))
    summarise('T_wm', t_wm, gm)
    summarise('T_out', t_out, gm)
    summarise('thickness', thickness, gm)

    if args.propagate:
        surf_dir = args.surf_dir
        if surf_dir is None:
            surf_dir = (args.reuse_white if os.path.isdir(args.reuse_white)
                        else os.path.join(src, args.reuse_white))
        cmp_dir = args.compare_pial
        if cmp_dir and not os.path.isdir(cmp_dir):
            cmp_dir = os.path.join(src, cmp_dir)
        print('propagating the white surface along the flow (step %g mm%s)'
              % (args.step,
                 '' if not args.flow_smooth
                 else ', T smoothed %g mm' % args.flow_smooth))
        propagate_and_write(dst, surf_dir, ref_img, unit,
                            np.clip(p_gm + p_wm, 0, 1), sigma
                            if sigma is not None else np.zeros_like(gm),
                            spacing, hemis=('lh', 'rh'), step=args.step,
                            compare_dir=cmp_dir)

    ref_thick = os.path.join(src, 'T1w_thickmap.nii.gz')
    if os.path.exists(ref_thick):
        ref = nib.load(ref_thick).get_fdata(dtype=np.float32)
        if ref.shape == gm.shape:
            both_pos = gm & (ref > 0)
            print('  DiReCT T1w_thickmap.nii.gz over the same %d voxels where '
                  'it is non-zero:' % int(both_pos.sum()))
            summarise('DiReCT', ref, both_pos)
            summarise('eikonal', thickness, both_pos)

    p = os.path.join(dst, args.prefix)
    save_img(thickness, p + 'thickmap.nii.gz', ref_img)
    save_img(t_wm * gm, p + 'T_wm.nii.gz', ref_img)
    save_img(t_out * gm, p + 'T_out.nii.gz', ref_img)
    save_img(sigma_div.astype(np.float32), p + 'sigma_div.nii.gz', ref_img)
    save_img(sigma_rev.astype(np.float32), p + 'sigma_rev.nii.gz', ref_img)
    save_img(np.where(gm, div, 0.0), p + 'div.nii.gz', ref_img)
    save_img(gm.astype(np.float32) * 2 + wm.astype(np.float32) * 3,
             p + 'seg.nii.gz', ref_img)
    print('wrote %sthickmap/T_wm/T_out/sigma_div/sigma_rev/div/seg.nii.gz to %s'
          % (args.prefix, dst))
    return 0


if __name__ == '__main__':
    sys.exit(main())
