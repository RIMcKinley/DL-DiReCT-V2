#!/usr/bin/env python
"""The field-propagated pial surface, current default configuration only.

A minimal restatement of what field_pial_prototype.py does with its shipped
defaults, with none of the options that exist there for experiments. Every
parameter below is fixed at the validated value and stated once; there are no
alternative branches to choose between. If you want to vary something, use
field_pial_prototype.py -- that is what it is for.

The method, in three stages:

  1. PREPARE. Load the GM/WM probabilities, build DiReCT's seg/gmT/wmT, then
     reconcile the WM label with the white surface itself (partial-volume
     rasterisation, supersample 3) so the solve and the surface share one WM
     boundary.

  2. SOLVE. ANTs' DiReCT (KellyKapowski) on the GPU, with ONE deviation: the
     velocity field is smoothed by a direction-gated Gaussian instead of an
     isotropic one. Across a sulcus the two banks carry near-antiparallel
     velocity and an isotropic kernel averages them into a common translation;
     the gate weights each neighbour by relu(cos) against the centre voxel's own
     velocity direction, so opposing banks stop cancelling. The divisor is the
     PLAIN Gaussian weight sum, so disagreement attenuates rather than being
     renormalised away.

  3. PROPAGATE. Carry the white surface along that field, 20 rounds at half the
     per-round step (so the total equals the solve's 10 integration points),
     with a light Taubin relaxation between rounds and the medial wall pinned.

Written 2026-09-11. The numbers it produces on bert (lh/rh): fundus CSF arrival
3.5/4.1%, mean displacement 2.401/2.402mm, self-intersections 173/36,
flipped_face_pct 0.0067/0.0015.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from . import direct_cuda as _direct_cuda
from .direct_cuda import (gaussian_smooth_3d, gaussian_gradient_3d, extract_wm_contours,
                          _make_identity_grid, warp_image, compose_fields, invert_field)
from .field_pial_prototype import (load_gm_wm_probability, build_seg_maps, make_transforms,
                                   check_frame_alignment, rasterize_mesh, rasterize_mesh_pv,
                                   reconcile_seg_with_surface, _mesh_adjacency,
                                   _smoothed_normals, build_constrained_white,
                                   build_no_push_mask, build_pin_mask, _pin_weights,
                                   evaluate_surface)
from .surface_frames import volume_info_from_image

# ---------------------------------------------------------------------------
# The configuration. These are the validated values; nothing here is a knob.
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 45        # DiReCT outer iterations
INTEGRATION_POINTS = 10    # inner integration steps per iteration
GRADIENT_STEP = 0.025      # mm, the Euler step of the descent
GRADIENT_GATE = 1e-3       # gradient magnitudes at or below this do not propagate
THICKNESS_PRIOR = 10.0     # mm; ANTs' cap. Never binds here (max observed 6.1mm)
SMOOTH_SIGMA = 1.0         # voxels, gradient + hit/total accumulation
VELOCITY_SIGMA = 0.6       # voxels, ANTs' -b. See the note below.
# 0.6, not ANTs' 1.0: this is the shipped operating point, validated on
# 2653 OASIS-3 sessions alongside segmentation='logits' (reproducibility
# 0.398%% global / 0.960%% ROI-average on same-session re-scans, against
# 0.489/1.274 for the released DL+DiReCT; CDR separation d = -1.33).
# The older comment here warned that below 1.0 "the mesh tangles steeply",
# which still holds as a statement about mesh self-intersection -- the
# trade was accepted on the reproducibility and atrophy benchmarks, not
# refuted. Raise it back to 1.0 if mesh quality is what you are after.
GATE_TRUNCATE = 2.0        # kernel radius = 2 voxels
FIELD_EPS = 1e-3           # a velocity below this has no usable direction
GATE_BLEND_BETA = 0.5      # the nu/field direction blend; see gated_velocity_smooth
ROUNDS = 20                # propagation rounds
STEP_SCALE = INTEGRATION_POINTS / ROUNDS   # keeps the total deformation fixed
RELAX_ITERS = 2            # Taubin iterations between rounds
RELAX_ITERS_FINAL = 1      # odd, so the last round ends on an unpaired shrink
RELAX_LAMBDA = 0.51        # matched to pymeshlab's filter; do not change
SUBSTEPS = 16              # field re-reads per round; 1 is a single linear step.
# Chosen on CONVERGENCE, not on a metric plateau: the propagated vertices move
# 0.0134, 0.0065, 0.0032, 0.0016, 0.0008 mm as K doubles from 2 to 32 -- clean
# first-order behaviour -- so by 16 the integration has converged to well under
# a thousandth of a voxel and is no longer a source of error. Cost is flat
# (0.13-0.18 s a hemisphere at any K, against 183 s for a case), so there is no
# reason to stop short. 32 buys nothing measurable and doubles the time.
PIN_FEATHER = 2            # mesh rings over which the medial-wall pin ramps off
WM_SUPERSAMPLE = 3         # partial-volume rasterisation of the white surface
INVERT_MAX_ITER = 20       # ANTs' cap on the inversion's fixed-point iterations
INVERT_CHECK_EVERY = 4     # host syncs per that many iterations; see below


# invert_field's convergence test -- `if max_error <= 0.1 or mean <= 0.001` --
# is a host synchronisation on every fixed-point iteration, ~9800 per solve and
# 16.8% of the function's time, which is 44.7% of the solve. MEASURED AND
# REJECTED: evaluating the test on the GPU (a sticky scalar flag freezing the
# update, so the answer is bit-identical to breaking) and syncing only every 4th
# iteration came out 10.3% SLOWER, 4/4 alternating repetitions. Freezing costs
# whole iterations -- the mean is 10.84 and checking every 4th runs to ~12.7 --
# and an iteration costs more than the sync it saves. Freezing with no sync at
# all (always 20) was 3.3% slower again. Do not retry without a way to stop on
# the exact iteration without asking the host.
#
# The first version of that experiment also reported -29.5%, which was
# torch.compile warm-up being paid by whichever arm ran first. Benchmark the
# solve by alternating arms and taking medians; a single A-then-B is worthless
# here.

def wm_normal_field(seg, device, wmT=None, spacing=(1.0, 1.0, 1.0)):
    """nu: the unit gradient of the WM signed distance, one vector per voxel.

    Points OUT of white matter. DiReCT's velocity runs GM->WM, so the outward
    direction the blend wants is -nu.

    THE BOUNDARY. With `wmT` -- the WM map the solve actually deforms -- the
    signed distance is taken from the 0.5 level set of that map rather than
    from the `seg == 3` binarisation. Two defects are fixed at once:

      seg == 3 is not what the solve deforms. wmT carries a partial-volume
      band from rasterize_mesh_pv of the white surfaces (5.1% of voxels
      strictly between 0 and 1); the two boundaries disagree in 0.73% of WM
      and give a nu more than 20 deg apart in 7% of GM.

      A mask boundary sits on a voxel STEP, so its zero level set is half a
      voxel inside the interface. distance_transform_edt measures to voxel
      centres, so an adjacent voxel reads 1.0 rather than 0.5.

    skfmm.distance solves the same signed-distance problem from a level set
    with sub-voxel accuracy. Its sign follows phi, and phi = 0.5 - wmT is
    negative inside white matter, which is the convention the binary branch
    has (edt(~wm) - edt(wm) is negative inside). Do NOT route this through
    eikonal_thickness._march: that returns np.abs(T), which would make the
    gradient reverse across the interface -- the bug fixed in d042362.
    """
    from scipy.ndimage import distance_transform_edt
    spacing = tuple(float(z) for z in spacing)
    if wmT is None:
        wmb = (seg == 3)
        sdt = (distance_transform_edt(~wmb, sampling=spacing)
               - distance_transform_edt(wmb, sampling=spacing))
    else:
        import skfmm
        phi = 0.5 - np.asarray(wmT, np.float64)
        # keep the sign consistent with the labels where the two disagree
        phi[seg == 3] = np.minimum(phi[seg == 3], -1e-3)
        phi[seg == 2] = np.maximum(phi[seg == 2], 1e-3)
        sdt = np.asarray(skfmm.distance(phi, dx=spacing), np.float64)
    grad = np.stack(np.gradient(sdt, *spacing), axis=-1)
    nu = grad / np.maximum(np.linalg.norm(grad, axis=-1), 1e-9)[..., None]
    return torch.from_numpy(nu.transpose(3, 0, 1, 2)[None].astype(np.float32)).to(device)


# Two smoothing regimes are supported, and BOTH are kept deliberately.
#
#   SHIPPED      blend_beta=0.5, reorient_alpha=None
#                The gate's reference is an equal mix of -nu and the current
#                velocity, so the operator adapts to the field it is smoothing.
#                Nonlinear and solution-dependent: measured S(2*v1 - 3*v2) vs
#                2*S(v1) - 3*S(v2) gives a relative error of 3.5e-01, and the
#                response to an identical perturbation differs by 4.0e-02
#                between two background fields.
#
#   VARIATIONAL  blend_beta=1.0, reorient_alpha=0.5
#                The reference is -nu alone, so it does not depend on the
#                velocity at all. The smoother is then LINEAR (same test:
#                3.5e-07) and solution-INDEPENDENT (2.98e-08) -- a fixed
#                anisotropic diffusion keyed to WM geometry, not an upwinding
#                or limiter scheme, because there is nothing for its stencil to
#                adapt to. All the nonlinearity sits in reorient_velocity,
#                which is a direction-only map: non-additive, and positively
#                homogeneous to 3.5e-05 (float32 rounding in the normalise)
#                away from the FIELD_EPS guard.
#
#                So the solve reads as linear smoothing step + nonlinear shrink
#                toward fixed geometry: proximal-gradient structure, which is
#                the form a variational argument needs. The two numbers only
#                give that property TOGETHER -- beta=1.0 alone is the gate at
#                its plateau, alpha=0.5 with beta=0.5 leaves the smoother
#                nonlinear -- so they are named as one configuration here
#                rather than left as two flags to remember.
#
# Geometry wins go to VARIATIONAL (12/12 hemispheres on self-intersections,
# transit, obliquity and boundary placement). Reproducibility is the open
# question; until that is settled neither is "the" default and SHIPPED stays
# the one you get by not asking.
VARIATIONAL = dict(blend_beta=1.0, reorient_alpha=0.5)


def lagrangian_nu(nu, inverse, identity, eps=0.5):
    """nu carried back to each voxel's ORIGIN instead of read where it sits.

    The Eulerian nu -- the normalised gradient of the WM signed distance --
    is degenerate exactly where the gate matters most. On the medial axis of a
    sulcal gap two banks are equidistant, the distance function creases, and a
    centred difference averages two opposing unit vectors: |grad| collapses
    toward 0 and the direction is decided by rounding. Measured on one case,
    16.8% of GM voxels have |grad| < 0.8.

    The degeneracy is an artefact of demanding ONE vector per voxel. A voxel in
    the middle of a sulcal gap has two answers because two banks own it -- so
    index the geometry by where the flow CAME FROM rather than by position.
    `inverse` is the solve's own map back toward WM, so

        nu_lag(x) = nu(x + inverse(x))

    reads the WM normal at the origin of whatever is sitting at x. Voxels
    arriving at the gap midline from opposite banks pull back to opposite
    faces of the WM surface and carry opposite normals, which is precisely the
    disagreement the gate needs in order to reject the far bank -- and it is
    available where the local gradient has none to give.

    This keeps the geometry FIXED in the sense that matters: what is read is
    still the segmentation's WM normal, never the evolving flow direction. The
    velocity enters only by choosing the sample point, so the reference remains
    a material property of the tissue rather than a function of the field.

    Where the pull-back leaves the volume, grid_sample's zero padding returns a
    null vector with no direction; those voxels keep the Eulerian nu rather
    than being handed noise.
    """
    lag = warp_image(nu, inverse, identity)
    mag = lag.norm(dim=1, keepdim=True)
    return torch.where(mag > eps, lag / mag.clamp(min=1e-9), nu)


def reorient_velocity(vol, nu, alpha):
    """Rotate the velocity toward the WM interface normal, preserving speed.

        v' = |v| * normalise( alpha * (-nu) + (1 - alpha) * vhat )

    This is NOT the gate's blend. gated_velocity_smooth uses the same
    combination as a REFERENCE direction -- a test each neighbour is scored
    against -- and leaves the vectors it averages untouched; the field it
    produces still points wherever the data pointed it, only less
    contaminated by the opposing bank. Here the field itself is turned, before
    every smoothing step, so the rotation compounds over the solve's
    iterations rather than acting once as a filter.

    Read as a regulariser, this is a proximal step on a penalty against
    deviation from the normal direction: each iteration takes the data term's
    increment, then pulls the direction back toward -nu by a fixed fraction,
    the way a proximal gradient method alternates a data step with a shrink.
    alpha is the step size of that shrink, not a mixing weight -- at alpha=1
    the field is projected ONTO -nu every iteration and the data term can only
    set the speed.

    Speed is preserved exactly, so a voxel with no velocity stays at zero
    rather than being handed -nu as a direction: the reorientation cannot
    create flow, only turn flow that the data term already produced.
    """
    if alpha is None or alpha <= 0:
        return vol
    mag = (vol ** 2).sum(dim=1, keepdim=True).sqrt()
    vhat = vol / mag.clamp(min=max(FIELD_EPS, 1e-12))
    u = alpha * (-nu) + (1.0 - alpha) * vhat
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
    # Below FIELD_EPS vhat is noise, so u is essentially -nu; multiplying by
    # mag keeps such a voxel at (near) zero either way. Guard it anyway so the
    # dead WM contour is bit-exactly untouched.
    return torch.where(mag > FIELD_EPS, mag * u, vol)


def travel_time_normal_field(seg, gm_prob, device, spacing=(1.0, 1.0, 1.0), eps=1e-3,
                             wm_prob=None, wmT=None):
    """nu from a GM-speed travel time out of white matter, not a Euclidean
    distance transform.

    wm_normal_field takes the gradient of edt(~wm) - edt(wm), which measures
    straight-line distance through anything. A sulcal gap is therefore as easy
    to cross as tissue: the two banks share a nearest WM voxel, the medial axis
    of the gap sits inside it, and nu there is the average of two opposing
    directions. Measured on OASIS, ~9.3% of GM is genuinely medial-axis under
    that field.

    Here the front marches OUT of white matter through tissue only, at a speed
    given by the GM posterior:

        |grad T| = 1 / F,   F = clip(p_gm, eps, 1),   domain = GM u WM

    Two distinct effects, measured separately on OAS30001:

      the DOMAIN restriction contributes ~11.3 deg of the ~13.3 deg total
      change in nu -- the front cannot cross a LABELLED sulcal gap at all, so
      the two banks get genuinely different directions;

      the SPEED field contributes the remaining ~3.2 deg, and it acts where the
      domain cannot: on BURIED csf, where a collapsed sulcus is unlabelled and
      sits inside the GM mask. There the posterior dips, the front slows, and
      the gap is recovered. Those voxels sit 2.45 mm from WM (mid-ribbon, where
      a collapsed sulcus belongs) against 1.00 mm for GM/WM partial volume, and
      nu turns 8.3 deg there against 0.36 deg on the exposed rim.

    The eps floor never binds on real data (min p_gm inside the domain is 0.021,
    no voxel reaches 1e-3); it exists so the march cannot divide by zero.

    Verified over 16 hemispheres against the Euclidean field, same solve
    otherwise: self-intersecting faces fall in 16/16 for both smoothing
    configurations, and with VARIATIONAL the fold faces -- the genuinely
    pathological class, as opposed to two banks legitimately touching -- fall
    2.3x, 16/16, while transit falls 3.0x against blend.

    The corpus callosum is deliberately NOT blocked: the front seeds on the
    WM/GM interface and never travels through white matter, so its speed is
    never read. Measured, it changes nothing (19.1 vs 19.6 deg).
    """
    import skfmm
    gm = seg == 2
    wm = seg == 3
    mask = ~(gm | wm)
    if wm_prob is None:
        # F = p_gm. Brakes wherever the GM posterior dips, which is BOTH buried
        # CSF and GM/WM partial volume -- and the latter is ~10x more numerous
        # (38778 vs 3925 voxels), so most of the braking is at the inner
        # boundary rather than at the buried sulci this is meant to recover.
        speed = np.clip(np.asarray(gm_prob, np.float64), eps, 1.0)
    else:
        # F = p_gm + p_wm. The mass a voxel assigns to NEITHER tissue. This is
        # ~1 at GM/WM partial volume, where the posterior merely moves between
        # the two tissues, and dips only where the model believes the voxel is
        # neither -- i.e. csf, buried or not. Targets the intended mechanism
        # rather than catching it incidentally.
        speed = np.clip(np.asarray(gm_prob, np.float64)
                        + np.asarray(wm_prob, np.float64), eps, 1.0)
    speed[wm] = 1.0
    # skfmm reads speeds OUTSIDE the marching domain too; leaving them at the
    # eps floor inflates T by ~40x. eikonal_thickness._march documents this and
    # neutralises the background, which is why it is used here rather than
    # calling skfmm directly.
    from .eikonal_thickness import _march
    spacing = tuple(float(z) for z in spacing)
    # THE SEED IS THE SOLVE'S OWN WM BOUNDARY. seg == 3 is a binarisation; what
    # the solve deforms is wmT, which carries a real partial-volume band from
    # rasterize_mesh_pv of the white surfaces (5.1% of voxels strictly between
    # 0 and 1, median 0.481). The two boundaries disagree in 0.73% of WM and
    # give a nu more than 20 deg apart in 7% of GM, so the choice is not a
    # rounding difference.
    #
    # phi = 0.5 - wmT puts the zero crossing where wmT crosses 0.5, sub-voxel,
    # rather than on the half-voxel step of a mask boundary -- the same trick
    # solve_t_wm uses, and the reason a binary-seeded march reads a median T of
    # 1.396 against the Euclidean 1.732, about half a voxel short.
    #
    # The sign is written out rather than copied from solve_t_wm: wmT is HIGH
    # inside white matter, so phi must be 0.5 - wmT to be negative there.
    if wmT is None:
        phi = np.where(wm, -1.0, 1.0)
    else:
        phi = 0.5 - np.asarray(wmT, np.float64)
        phi[wm] = np.minimum(phi[wm], -1e-3)
        phi[gm] = np.maximum(phi[gm], 1e-3)
    T, unreached = _march(phi, mask, speed, spacing, True)
    # SIGN IT. _march returns |T| -- the distance from the zero level set in
    # BOTH directions -- so T has a V-shaped minimum at the WM/GM interface and
    # its gradient REVERSES across it. Unsigned, nu points INTO white matter on
    # the WM side: measured cos against the Euclidean nu is -0.92 at the WM
    # contour and -0.99 in the WM interior, against +0.96 in GM.
    #
    # That is not cosmetic. The gate then sees the contour's reference opposing
    # every GM neighbour's, rejects them all, and the contour's velocity
    # collapses by a factor of 500 (|v| 0.092 -> 0.00018). GM velocity is
    # untouched, so travel and nn look normal while the FIELD metric -- which
    # integrates outward FROM the contour -- reads 0.34 mm instead of 2.07.
    # The Euclidean sdt is signed by construction and never had this.
    T = np.where(wm, -T, T)
    T = np.where(unreached | mask, np.nan, T)
    finite = np.isfinite(T)
    T = np.where(finite, T, np.nanmax(T[finite]) * 1.5)
    grad = np.stack(np.gradient(T, *spacing), axis=-1)
    mag = np.linalg.norm(grad, axis=-1)
    # A zero gradient (flat fill outside the domain, or a plateau) would
    # normalise to a null vector, which the gate would score as zero
    # agreement with everything. Fall back to the Euclidean direction there
    # so every voxel still carries a unit direction, as wm_normal_field does.
    dead = mag <= 1e-9
    if dead.any():
        from scipy.ndimage import distance_transform_edt
        sdt = (distance_transform_edt(~wm, sampling=spacing)
               - distance_transform_edt(wm, sampling=spacing))
        g2 = np.stack(np.gradient(sdt, *spacing), axis=-1)
        grad = np.where(dead[..., None], g2, grad)
        mag = np.linalg.norm(grad, axis=-1)
    nu = grad / np.maximum(mag, 1e-9)[..., None]
    return torch.from_numpy(nu.transpose(3, 0, 1, 2)[None].astype(np.float32)).to(device)


def gated_velocity_smooth(vol, sigma, device, nu=None, beta=GATE_BLEND_BETA):
    """Gaussian smoothing of the velocity field that refuses to average across a
    direction reversal.

    out(i) = sum_j w_j * relu(cos(v_j, v_i)) * v_j / sum_j w_j

    w is the separable Gaussian weight. The numerator drops neighbours in the
    opposing half-space; the divisor keeps them, so a voxel whose neighbourhood
    disagrees is attenuated rather than renormalised back to full magnitude.

    A velocity below FIELD_EPS has no direction -- normalising it returns noise,
    which under relu scores about half weight on average rather than being
    ignored. Such a centre is left unsmoothed and such a neighbour carries no
    vote. This leaves the WM contour (which gets no increment of its own, since
    the speed term is masked to GM) holding zero for the whole solve; the
    propagation's trilinear stencil still reads live corners around it.

    BLEND. With `nu` supplied, the direction each voxel is compared against is
    not its own velocity but an equal mix of the flow and the WM interface
    geometry:

        u = normalise( beta * (-nu) + (1 - beta) * vhat )

    used for the centre AND the neighbours. Two consequences. A voxel whose
    velocity is zero still has a direction (-nu), so the WM contour -- 31% of
    the active region, and dead for the whole solve under the field reference --
    is smoothed rather than skipped; the FIELD_EPS guard is therefore not
    applied. And a neighbour on the far bank of a sulcus is rejected on
    geometry even where the flow has not yet separated the two.

    beta=0.5 is the validated value: 36/36 hemispheres better on fundus CSF
    arrival (3.39 -> 7.79%), fundus travel (+0.437mm), slide and
    self-intersections; 0/36 on crown arrival. beta=0 is the plain field
    reference, beta=1 is the pure geometric one.
    """
    r = max(1, int(GATE_TRUNCATE * sigma + 0.5))
    coords = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    # The tap weights are constants of the kernel. Reading them off the DEVICE,
    # as float(g[i]*g[j]*g[k]) did, is a host synchronisation per tap: 125 per
    # call, 5625 per solve. Multiply them here in float32, exactly as the device
    # expression did, so the weights are the same bits.
    gh = g.cpu().numpy()

    mag = (vol ** 2).sum(dim=1, keepdim=True).sqrt()
    ok = mag > FIELD_EPS
    if nu is None:
        ref = vol / mag.clamp(min=FIELD_EPS)
    else:
        fhat = vol / mag.clamp(min=max(FIELD_EPS, 1e-12))
        ref = beta * (-nu) + (1.0 - beta) * torch.where(ok, fhat, torch.zeros_like(fhat))
        ref = ref / ref.norm(dim=1, keepdim=True).clamp(min=1e-9)
        ok = torch.ones_like(ok)          # every voxel now has a direction

    D, H, W = vol.shape[2:]
    padded = F.pad(vol, (r,) * 6, mode='replicate')
    padded_ref = F.pad(ref, (r,) * 6, mode='replicate')
    padded_ok = F.pad(ok.to(vol.dtype), (r,) * 6, mode='replicate')
    # Fold the neighbour's liveness into the field once instead of reading it
    # back on every tap. ok is exactly 0.0 or 1.0, so this is bit-exact.
    padded = padded * padded_ok

    acc = torch.zeros_like(vol)
    wsum = 0.0
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                w = float(gh[dz + r] * gh[dy + r] * gh[dx + r])
                if w < 1e-6:
                    continue
                sl = (slice(None), slice(None), slice(r + dz, r + dz + D),
                      slice(r + dy, r + dy + H), slice(r + dx, r + dx + W))
                dw = (padded_ref[sl] * ref).sum(dim=1, keepdim=True).clamp(min=0.0)
                acc = acc + w * dw * padded[sl]
                wsum += w
    out = acc / max(wsum, 1e-6)
    return torch.where(ok, out, vol)

    # Measured -16.6% and bit-identical. torch.compile on top of this adds
    # nothing (-15.9% vs -16.6%): the loop is memory-bandwidth bound, not
    # launch bound, so the only remaining lever is fewer taps, which would
    # change the answer.


def solve_velocity_field_t(seg, gm_prob, wm_prob, ref_img, verbose=True, device=None,
                           compute_thickness=True, blend_beta=GATE_BLEND_BETA,
                           velocity_sigma=VELOCITY_SIGMA, smoothing='gated',
                           reorient_alpha=None, nu_source='eulerian',
                           nu_mode='euclidean', gm_posterior=None, nu=None,
                           crop_safe=False):
    """The solve itself, returning the field as a TORCH TENSOR on its device.

    Returns (velocity, thickness, device) with velocity [1, 3, D, H, W] and
    thickness [1, 1, D, H, W]. Nothing is copied to the host, so a caller that
    propagates on the GPU never moves the field across the bus.
    solve_velocity_field() below is the numpy-returning wrapper.

    compute_thickness=False skips the thickness map entirely and returns None in
    its place. That removes two warp_image calls per integration point (20 per
    iteration) and two Gaussian smooths per iteration.

    It also removes the THICKNESS_PRIOR cap, which is derived from the same
    hit/total accumulation: where the running thickness exceeds the prior the
    velocity is scaled down by (prior/thickness)^2. On the data this has been
    run on the cap never binds (max observed thickness 6.1 mm against a 10.0 mm
    prior) and the field is bit-identical either way -- but that is a property
    of the data, not a guarantee. If you need the pial surface from a subject
    where cortex might exceed the prior, leave this on.
    """
    device = torch.device(device) if device is not None else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # BATCHING. Every tensor below carries a leading batch dimension that used
    # to be pinned at 1. Pass seg/gm_prob/wm_prob as SEQUENCES of equally
    # shaped volumes and the whole solve runs them together in one set of
    # kernels; pass single volumes and nothing changes -- B is 1 and the
    # arithmetic is identical, which is what keeps existing callers working.
    #
    # The solve was already batch-agnostic everywhere it matters:
    # gated_velocity_smooth slices with slice(None) on batch and channel and
    # reduces over dim=1; warp_image indexes channels, not batch, and its
    # identity_grid at [1, D, H, W, 3] BROADCASTS against a [B, ...]
    # displacement; grid_sample and F.pad are natively batched. Only the
    # literal allocations here needed changing.
    #
    # The batch must share one grid. Volumes of different shapes cannot be
    # stacked, and silently padding them would move the anatomy.
    def _stack(a):
        if isinstance(a, (list, tuple)):
            shapes = {np.asarray(x).shape for x in a}
            if len(shapes) != 1:
                raise ValueError('batched solve needs one grid, got %s' % sorted(shapes))
            return np.stack([np.asarray(x, np.float32) for x in a])
        return np.asarray(a, np.float32)[None]

    seg_b, gm_b, wm_b = _stack(seg), _stack(gm_prob), _stack(wm_prob)
    if not (seg_b.shape == gm_b.shape == wm_b.shape):
        raise ValueError('seg/gm_prob/wm_prob disagree: %s %s %s'
                         % (seg_b.shape, gm_b.shape, wm_b.shape))
    B, D, H, W = seg_b.shape
    _batched = isinstance(seg, (list, tuple))

    # CROP-SAFE SOLVE. The solve iterates the whole grid to do work that touches
    # the active region only -- 5.1% of a 256^3 conform on OAS30001. Cropping to
    # the active bounding box is 4.7x fewer voxels and, on its own, 6.6x faster
    # (superlinear: there is a bandwidth effect on top of the work saved).
    #
    # The cropped grid also falls under INVERT_GRAPH_MAX_VOXELS, so it picks up
    # the CUDA graph, which is worth -25.6% at 2.45M and +8.1% (i.e. a loss) at
    # 11.53M -- the size gate is what lets the crop have it and the full grid not.
    #
    # It is NOT safe with the stock convergence test, which means the residual
    # over the whole volume and so converges sooner the more empty space there
    # is -- cropping alone shifts thickness by -0.0075 mm, as large as the entire
    # model-ensemble uncertainty at the global mean. The two therefore ship as
    # ONE switch: crop_safe crops AND drops invert_field's volume-dependent mean
    # term, keeping only max_error <= tol, which is scale-free. Measured
    # crop-vs-full shift: -7.52e-03 mm with the stock test, +1.35e-04 with
    # max-only. The mean term was the whole problem.
    # See direct_cuda.INVERT_MEAN_OVER_SUPPORT for the deviation this accepts.
    #
    # Pass an int to set the margin in voxels (default 4). Velocity and thickness
    # are returned on the FULL grid, so no caller has to know this happened.
    if crop_safe:
        margin = 4 if crop_safe is True else int(crop_safe)
        _s = torch.from_numpy(seg_b).reshape(B, 1, D, H, W)
        _act = ((_s == 2).float() + extract_wm_contours(_s)).clamp(max=1.0)
        _act = (_act.numpy()[:, 0] > 0).any(axis=0)
        if _act.any():
            _i = np.array(np.nonzero(_act))
            _lo = np.maximum(_i.min(1) - margin, 0)
            _hi = np.minimum(_i.max(1) + 1 + margin, np.array([D, H, W]))
            sl = tuple(slice(int(a), int(b)) for a, b in zip(_lo, _hi))
            cut = lambda a: a[(slice(None),) + sl]
            pick = lambda a: [x for x in a] if _batched else a[0]
            gp = gm_posterior
            if gp is not None:
                _g = [np.asarray(x)[sl] for x in
                      (gp if isinstance(gp, (list, tuple)) else [gp])]
                gp = _g if isinstance(gp, (list, tuple)) else _g[0]
            nu_c = (nu[(slice(None), slice(None)) + sl].contiguous()
                    if nu is not None else None)
            was = _direct_cuda.INVERT_MAX_ONLY
            _direct_cuda.INVERT_MAX_ONLY = True
            try:
                v, th, dv = solve_velocity_field_t(
                    pick(cut(seg_b)), pick(cut(gm_b)), pick(cut(wm_b)), ref_img,
                    device=device, compute_thickness=compute_thickness,
                    blend_beta=blend_beta, smoothing=smoothing,
                    reorient_alpha=reorient_alpha, nu_source=nu_source,
                    nu_mode=nu_mode, gm_posterior=gp, nu=nu_c, crop_safe=False,
                    verbose=verbose)
            finally:
                _direct_cuda.INVERT_MAX_ONLY = was
            V = torch.zeros(B, 3, D, H, W, device=v.device, dtype=v.dtype)
            V[(slice(None), slice(None)) + sl] = v
            T = None
            if th is not None:
                T = torch.zeros(B, 1, D, H, W, device=th.device, dtype=th.dtype)
                T[(slice(None), slice(None)) + sl] = th
            return V, T, dv
    t = lambda a: torch.from_numpy(a).to(device).reshape(B, 1, D, H, W)
    seg_t, gm_t, wm_t = t(seg_b), t(gm_b), t(wm_b)

    gm_mask = (seg_t == 2).float()                 # the increment lands here ONLY
    wm_contour = extract_wm_contours(seg_t)
    active = (gm_mask + wm_contour).clamp(max=1.0)
    identity = _make_identity_grid((D, H, W), device)
    # PRECOMPUTED nu. Building it is host work -- distance_transform_edt and
    # np.gradient, or skfmm for travel -- and on a 2.5M-voxel grid it costs
    # 9.3 s against a 20.6 s GPU solve, i.e. 45%. In EUCLIDEAN mode nu is a
    # function of (seg, wmT) alone, so every batch item sharing a WM source
    # shares a nu: an n x n grid needs n builds, not n^2. Pass it here as a
    # [B, 3, D, H, W] tensor, or as [1, 3, D, H, W] to be copied across the
    # whole batch (see the .repeat below -- expand is NOT safe). Travel-time nu also
    # consumes gm_posterior and does vary per cell, so it cannot be shared
    # across differing GM sources.
    nu_t = None
    if nu is not None:
        nu_t = nu if nu.shape[0] in (1, B) else None
        if nu_t is None:
            raise ValueError('nu has batch %d, need 1 or %d' % (nu.shape[0], B))
        if nu_t.shape[0] == 1 and B > 1:
            # .repeat, NOT .expand. expand gives stride 0 on the batch axis so
            # every item aliases one buffer, and the solve then produces items
            # that agree with each other but do NOT match the same volume solved
            # alone -- measured max|diff| 2.6e-2 on a two-item phantom, with the
            # items bit-identical to each other. A real copy costs
            # B * 3 * D*H*W * 4 bytes (1.1 GB at B=36 on a 2.5M-voxel grid),
            # which is affordable and correct.
            nu_t = nu_t.repeat(B, 1, 1, 1, 1)
    elif blend_beta is not None:
        if nu_mode == 'travel':
            if gm_posterior is None:
                raise ValueError(
                    "nu_mode='travel' needs gm_posterior, the RAW GM probability. "
                    "`gm_prob` here is gmT, which build_seg_maps has already "
                    "saturated (83% of GM at 1.0) and which shows no dip at buried "
                    "CSF -- passing it would silently reduce the speed field to a "
                    "flat-speed geodesic. prepare() returns it as 'gm_raw'.")
            # SEEDED FROM THE SOLVE'S OWN WM BOUNDARY. seg == 3 is a
            # binarisation: the two disagree in 0.73% of WM, give a nu more
            # than 20 deg apart in 7% of GM, and the binary seed puts the zero
            # level set half a voxel out -- median T 1.597 against a Euclidean
            # 1.732, which is impossible for a speed <= 1. phi = 0.5 - wmT puts
            # it on the real interface, sub-voxel, and T comes out at 1.780.
            #
            # Measured over 16 hemispheres the binary seed scores slightly
            # better (self-int 0.522 vs 0.666, transit 0.1152 vs 0.1574), but
            # those margins are 0.14 and 0.04 percentage points, and this
            # project has repeatedly been misled by ranking on small
            # differences in exactly these counts. A seed that is provably in
            # the wrong place is not worth 0.14 pp.
            # nu is built per batch item and stacked: both builders run
            # scikit-fmm / distance transforms on the host and have no batch
            # dimension of their own. They return [1, 3, D, H, W], so the cat
            # gives [B, 3, D, H, W] in the same order as seg_b.
            gp = (gm_posterior if isinstance(gm_posterior, (list, tuple))
                  else [gm_posterior] * B)
            if len(gp) != B:
                raise ValueError('gm_posterior has %d entries, need %d' % (len(gp), B))
            nu_t = torch.cat([
                travel_time_normal_field(seg_b[k], gp[k], device,
                                         ref_img.header.get_zooms()[:3],
                                         wmT=wm_b[k])
                for k in range(B)], dim=0)
        else:
            nu_t = torch.cat([
                wm_normal_field(seg_b[k], device, wmT=wm_b[k],
                                spacing=ref_img.header.get_zooms()[:3])
                for k in range(B)], dim=0)

    velocity = torch.zeros(B, 3, D, H, W, device=device)
    integrated = torch.zeros(B, 3, D, H, W, device=device)
    thickness_img = torch.zeros(B, 1, D, H, W, device=device)
    cortical_thickness = torch.zeros(B, 1, D, H, W, device=device)

    for iteration in range(MAX_ITERATIONS):
        increment = torch.zeros(B, 3, D, H, W, device=device)
        inverse = torch.zeros(B, 3, D, H, W, device=device)
        hit = torch.zeros(B, 1, D, H, W, device=device)
        total = torch.zeros(B, 1, D, H, W, device=device)

        for pt in range(1, INTEGRATION_POINTS + 1):
            inverse = compose_fields(velocity * active, inverse, identity)
            warped_wm = warp_image(wm_t, inverse, identity)
            if compute_thickness:
                warped_contour = warp_image(wm_contour, inverse, identity)
                warped_thick = warp_image(thickness_img, inverse, identity)

            grad = gaussian_gradient_3d(warped_wm, SMOOTH_SIGMA, device)
            gmag = (grad * grad).sum(dim=1, keepdim=True).sqrt()
            direction = grad / (gmag + 1e-8) * (gmag > GRADIENT_GATE).float()

            speed = -(warped_wm - gm_t) * gm_t * GRADIENT_STEP * gm_mask
            speed = torch.where(torch.isfinite(speed), speed, torch.zeros_like(speed))
            increment = increment + direction * speed

            if compute_thickness:
                if pt == 1:
                    thickness_img = integrated.norm(dim=1, keepdim=True) * wm_contour
                    hit = wm_contour.clone()
                    total = thickness_img.clone()
                else:
                    hit = hit + warped_contour * gm_mask
                    total = total + warped_thick * gm_mask

            inverse = inverse * active
            velocity = velocity * active
            if pt == 1:
                integrated.zero_()
            integrated = invert_field(inverse, identity, initial=integrated)
            inverse = invert_field(integrated, identity, initial=inverse)

        velocity = velocity + increment

        if compute_thickness:
            sh = gaussian_smooth_3d(hit, SMOOTH_SIGMA, device, zero_boundary=False)
            st = gaussian_smooth_3d(total, SMOOTH_SIGMA, device, zero_boundary=False)
            has = sh > 0.001
            vals = torch.where(has, st / sh.clamp(min=0.001), torch.zeros_like(sh)).clamp(min=0)
            over = has & (vals > THICKNESS_PRIOR) & (gm_mask > 0)
            if over.any():
                frac = THICKNESS_PRIOR / vals.clamp(min=1e-8)
                velocity = velocity * torch.where(over, frac * frac, torch.ones_like(frac))
            cortical_thickness = vals * gm_mask

        # The geometry the gate and the reorientation are measured against.
        # 'lagrangian' reads it at each voxel's origin, which gives the two
        # banks of a sulcus different normals where the SDT has none.
        nu_ref = nu_t
        if nu_t is not None and nu_source == 'lagrangian':
            nu_ref = lagrangian_nu(nu_t, inverse, identity)

        if reorient_alpha:
            # BEFORE the smoothing, every iteration: see reorient_velocity.
            if nu_ref is None:
                nu_t = torch.cat([wm_normal_field(seg_b[k], device)
                                  for k in range(B)], dim=0)
                nu_ref = (lagrangian_nu(nu_t, inverse, identity)
                          if nu_source == 'lagrangian' else nu_t)
            velocity = reorient_velocity(velocity, nu_ref, reorient_alpha)

        if smoothing == 'plain':
            # The ungated Gaussian the original DiReCT applies: no relu(cos)
            # rejection of opposing neighbours, no nu blend. For a baseline arm;
            # the gate is the default for the reasons gated_velocity_smooth
            # documents.
            velocity = gaussian_smooth_3d(velocity, velocity_sigma, device,
                                          zero_boundary=False)
        else:
            velocity = gated_velocity_smooth(velocity, velocity_sigma, device,
                                         nu=nu_ref, beta=blend_beta)
        velocity = velocity * active          # MUST precede the save; see below
        if verbose and (iteration + 1) % 10 == 0:
            if compute_thickness:
                print('  iteration %d/%d, mean thickness %.3f mm'
                      % (iteration + 1, MAX_ITERATIONS,
                         float(cortical_thickness[gm_mask > 0].mean())))
            else:
                print('  iteration %d/%d' % (iteration + 1, MAX_ITERATIONS))

    return velocity, (cortical_thickness if compute_thickness else None), device


def velocity_to_numpy(velocity):
    """[B, 3, D, H, W] tensor -> [D, H, W, 3] array, or a LIST of them when B > 1.

    Kept single-volume for B == 1 so every existing caller is untouched.
    and the NIfTI export both want it."""
    a = velocity.detach().cpu().numpy().transpose(0, 2, 3, 4, 1).astype(np.float32)
    return a[0] if a.shape[0] == 1 else [a[k] for k in range(a.shape[0])]


def solve_velocity_field(seg, gm_prob, wm_prob, ref_img, out_prefix=None, verbose=True,
                         compute_thickness=True, blend_beta=GATE_BLEND_BETA,
                         velocity_sigma=VELOCITY_SIGMA):
    """DiReCT with the gated velocity smoothing. Returns (velocity, thickness).

    velocity is [D, H, W, 3] in voxels, components in voxel-index order (d,h,w),
    and is the PER-INTEGRATION-POINT field: the solve composes it
    INTEGRATION_POINTS times, which is why the propagation below applies it
    ROUNDS times at STEP_SCALE.
    """
    velocity, cortical_thickness, _ = solve_velocity_field_t(
        seg, gm_prob, wm_prob, ref_img, verbose=verbose,
        compute_thickness=compute_thickness, blend_beta=blend_beta,
        velocity_sigma=velocity_sigma)
    vel = velocity_to_numpy(velocity)
    if out_prefix:
        # AFTER the active-region mask. Saving before it exported a 12% smoothing
        # halo into CSF, which the propagation then rode.
        img = nib.Nifti1Image(vel, ref_img.affine)
        img.header['xyzt_units'] = 10
        nib.save(img, out_prefix + 'Velocity.nii.gz')
    if cortical_thickness is None:
        return vel, None
    # squeeze() alone would also drop the batch axis at B == 1, which is what
    # we want there, but at B > 1 it must be kept: squeeze the CHANNEL only.
    ct = cortical_thickness.detach()[:, 0].cpu().numpy()
    return vel, (ct[0] if ct.shape[0] == 1 else [ct[k] for k in range(ct.shape[0])])


def propagate_pial(white_verts, faces, velocity, seg, tovox, totkr, pin_mask=None,
                   return_path=False):
    """Carry the white surface along the velocity field.

    DiReCT's velocity points GM->WM, so the outward direction is its negative.
    The field is sampled AT the vertex (the old out-of-WM offset was a
    workaround for the dead WM shell and made a vertex step on a field half a
    millimetre from where it is).

    return_path also returns the TRAJECTORY, [ROUNDS+1, n, 3] in tkrRAS: the
    path each vertex took, which IS its cortical column. ribbon_labels.
    stamp_columns turns that into a ribbon parcellation without re-integrating
    anything, and it is the real path including relaxation, not a pure-field
    reconstruction of it. Costs ~38 MB for a 150k-vertex hemisphere.
    """
    mesh, Wm, deg = _mesh_adjacency(white_verts, faces)
    from scipy.ndimage import distance_transform_edt
    wmb = (seg == 3)
    sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)
    grad = np.stack(np.gradient(sdt), axis=-1)
    gu = grad / np.maximum(np.linalg.norm(grad, axis=-1), 1e-9)[..., None]
    cache = (Wm, deg, sdt, gu)

    pin_w = _pin_weights(pin_mask, Wm, PIN_FEATHER)[:, None] \
        if (pin_mask is not None and pin_mask.any()) else None
    start = np.asarray(white_verts).copy()
    cur = white_verts
    path = [np.asarray(cur, np.float32).copy()] if return_path else None
    for rnd in range(ROUNDS):
        # SUB-STEPPED, matching propagate_pial_torch: the round's displacement
        # is unchanged, integrated in SUBSTEPS pieces with the field re-read
        # between them. A single step linearises the trajectory over the whole
        # round and overshoots where the field varies fast along it -- which is
        # where the folds are. See the note in propagate_pial_torch.
        for _sub in range(SUBSTEPS):
            pos = tovox(cur)
            v = np.stack([map_coordinates(velocity[..., k], pos.T, order=1, mode='nearest')
                          for k in range(3)], axis=1)
            cur = cur + (totkr(pos - v) - totkr(pos)) * (STEP_SCALE / SUBSTEPS)
        iters = RELAX_ITERS if rnd < ROUNDS - 1 else RELAX_ITERS_FINAL
        cur = build_constrained_white(cur, faces, seg, tovox, totkr,
                                      floor=-np.inf, iters=iters, lam=RELAX_LAMBDA,
                                      cache=cache)
        if pin_w is not None:
            cur = pin_w * start + (1.0 - pin_w) * cur
        if path is not None:
            path.append(np.asarray(cur, np.float32).copy())
    if path is not None:
        return cur, np.stack(path, 0)
    return cur


def prepare(prep_dir, surf_dir=None, hemis=('lh', 'rh'), surfaces=None, tissue=None):
    """seg/gmT/wmT reconciled against the white surfaces, plus the transforms.

    `surfaces` optionally supplies {hemi: (verts, faces)} already in the cropped
    tkrRAS frame, in place of reading ?h.white from `surf_dir`. BOTH hemispheres
    are still required: the WM label is reconciled against the union of the two
    surfaces, so a one-hemisphere call would demote the other hemisphere's WM.
    """
    import pandas as pd
    # `tissue` is (gm_prob, wm_prob, ref_img) already in memory -- what
    # field_pial_prototype.gm_wm_probability_from_logits returns straight from
    # the model, so the 94 logit volumes never touch disk. Verified identical
    # to the on-disk collapse (max|diff| 0).
    if tissue is not None:
        gm_prob, wm_prob, ref_img = tissue
    else:
        gm_prob, wm_prob, ref_img = load_gm_wm_probability(prep_dir)
    seg, gmT, wmT = build_seg_maps(gm_prob, wm_prob)
    tovox, totkr = make_transforms(ref_img)
    shape = tuple(ref_img.shape[:3])

    given = dict(surfaces) if surfaces else None
    if given is not None and set(given) != {'lh', 'rh'}:
        sys.exit('surfaces= needs both hemispheres (got %s); the WM label is '
                 'reconciled against their union' % sorted(given))
    if given is None and not surf_dir:
        sys.exit('give either surf_dir or surfaces=')
    surfaces = {}
    partial = np.zeros(shape, np.float32)
    crisp = np.zeros(shape, bool)
    for h in ('lh', 'rh'):
        if given is not None:
            v, f = given[h]
            v = np.asarray(v, np.float64)
            f = np.asarray(f)
        else:
            path = os.path.join(surf_dir, '%s.white' % h)
            if not os.path.exists(path):
                sys.exit('missing %s -- both white surfaces are needed to define WM' % path)
            v, f, vinfo = nib.freesurfer.io.read_geometry(path, read_metadata=True)
            if vinfo and 'volume' in vinfo and \
                    tuple(int(x) for x in vinfo['volume']) != shape:
                sys.exit('%s.white was built on a %s grid, reference is %s: the tkrRAS frames '
                         'differ by half a voxel per odd axis. Rebuild with preparedata.py '
                         '--space cropped.' % (h, list(vinfo['volume']), list(shape)))
        surfaces[h] = (v, f)
        crisp |= rasterize_mesh(tovox(v), f, shape)
        partial += rasterize_mesh_pv(tovox(v), f, shape, WM_SUPERSAMPLE)
    before = int((seg == 3).sum())
    seg, gmT, wmT, dem, pro = reconcile_seg_with_surface(
        seg, gmT, wmT, np.clip(partial, 0.0, 1.0), label_mask=crisp)
    print('WM from surface: %d -> %d voxels (%d demoted, %d promoted)'
          % (before, int((seg == 3).sum()), dem, pro))
    for h in hemis:
        dist, frac, inb = check_frame_alignment(surfaces[h][0], seg, tovox)
        print('frame check %s: mean |distance to WM boundary| %.3f mm '
              '(expect <1.5; %.1f%% sample WM, %.1f%% in bounds)'
              % (h, dist, 100 * frac, 100 * inb))
        if dist > 1.5 or inb < 0.99:
            print('WARNING: frame alignment looks wrong for %s' % h, file=sys.stderr)
    # gm_raw is the UNTRANSFORMED posterior. gmT is 83% saturated at 1.0 and
    # shows no dip at buried CSF, so it cannot drive the travel-time speed
    # field; see travel_time_normal_field.
    return dict(seg=seg, gmT=gmT, wmT=wmT, gm_raw=gm_prob, ref_img=ref_img,
                tovox=tovox, totkr=totkr,
                surfaces={h: surfaces[h] for h in hemis}, prep_dir=prep_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True,
                   help='a --space cropped prep (seg_<Label>.nii.gz, softmax_seg.nii.gz, '
                        'label_def.csv)')
    p.add_argument('--surf-dir', required=True, help='directory holding ?h.white')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--write-thickness', action='store_true',
                   help="also write this solve's thickness map and segmentation")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd
    d = prepare(args.prep_dir, args.surf_dir, tuple(args.hemi))
    seg, tovox, totkr = d['seg'], d['tovox'], d['totkr']

    print('solving the velocity field (%d iterations)...' % MAX_ITERATIONS)
    velocity, thickness = solve_velocity_field(
        seg, d['gmT'], d['wmT'], d['ref_img'], os.path.join(args.out_dir, 'pial_'))

    soft = os.path.join(args.prep_dir, 'softmax_seg.nii.gz')
    ldef = os.path.join(args.prep_dir, 'label_def.csv')
    soft_seg = id_map = None
    if os.path.exists(soft) and os.path.exists(ldef):
        soft_seg = np.asarray(nib.load(soft).dataobj)
        id_map = {r.LABEL: int(r.ID) for _, r in pd.read_csv(ldef).iterrows()}
    else:
        print('WARNING: no softmax_seg.nii.gz / label_def.csv; the medial wall will not '
              'be pinned and will be dragged outward.', file=sys.stderr)

    vinfo = volume_info_from_image(d['ref_img'], args.prep_dir)
    for hemi, (white, faces) in d['surfaces'].items():
        pin = None
        if soft_seg is not None:
            no_push = build_no_push_mask(white, faces, seg, soft_seg, id_map, tovox, rings=0)
            pin = build_pin_mask(no_push, white, faces, seg, soft_seg, id_map, tovox,
                                 scope='medial-wall', rings=0)
            print('%s: %d/%d vertices with no cortex to move into, %d pinned'
                  % (hemi, no_push.sum(), len(no_push), pin.sum()))
        pial = propagate_pial(white, faces, velocity, seg, tovox, totkr, pin_mask=pin)
        out = os.path.join(args.out_dir, '%s.pial' % hemi)
        nib.freesurfer.io.write_geometry(out, pial, faces, create_stamp=None,
                                         volume_info=vinfo)
        m = evaluate_surface(white, pial, faces, seg, tovox,
                             no_push=pin if (pin is not None and pin.any()) else None)
        print('%s -> %s   displacement %.3f mm, crossed_csf %d, self-intersections %d, '
              'flipped %.4f%%' % (hemi, out, m['mean_displacement_mm'],
                                  m['crossed_csf_count'], m['self_intersections'],
                                  m['flipped_face_pct']))

    if args.write_thickness and np.isfinite(thickness).all() and (thickness > 0).any():
        for name, arr in (('T1w_thickmap.nii.gz', thickness.astype(np.float32)),
                          ('seg.nii.gz', seg.astype(np.uint8))):
            img = nib.Nifti1Image(arr, d['ref_img'].affine)
            img.header['xyzt_units'] = 2
            nib.save(img, os.path.join(args.out_dir, name))
        print('thickness: mean %.3f mm over non-zero voxels'
              % thickness[thickness > 0].mean())


if __name__ == '__main__':
    main()
