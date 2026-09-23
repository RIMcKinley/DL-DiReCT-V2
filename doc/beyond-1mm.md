# Running beyond 1mm isotropic

Until this branch, every DL+DiReCT-V2 path began with `conform.py` resampling the
input to 1mm isotropic LIA, and everything downstream was written against that
guarantee. Not loudly — almost nothing said "1mm" anywhere. The assumption is
that **a voxel is a millimetre**, and it is invisible precisely because at 1mm
isotropic it is true.

This document records where that assumption lived, what was changed, and what
was measured on 0.7mm data.

## The shape of the bug

Two units get conflated:

* a **displacement** measured in voxel indices, and
* a **distance** measured in millimetres.

The whole pipeline is full of parameters documented in millimetres — the DiReCT
solver's `gradient_step` (0.025mm) and `thickness_prior` (10mm), the pial
propagation's signed-distance `floor` (-0.5mm) and `floor_after` (0.5mm), the
sulcal sheet's `max_ray_mm` (4mm), the no-cortex `radius` — and those were
compared against, or applied to, quantities that were actually in voxels.

At 1mm isotropic both readings agree and every number is right. At 0.7mm the two
differ by a factor of 1/0.7 = 1.43, in whichever direction the particular site
happens to point. Nothing errors; the pipeline produces a complete, plausible set
of results that are simply scaled wrong.

Measured on the sample subject (`T1NAT__PT030_OpenNeuro_ds000031__sub-01__ses-07`,
0.7mm isotropic, GT segmentation), mean cortical thickness over cortex voxels:

| solver configuration | mean | median | p95 |
|---|---|---|---|
| before: `voxel_size=None`, i.e. voxel displacements read as mm | 3.418 mm | 3.390 | 5.175 |
| `voxel_size` set, `sigma_units='voxel'` (ANTs literal) | 2.651 mm | 2.623 | 3.915 |
| `voxel_size` set, `sigma_units='mm'` (new default) | 2.415 mm | 2.384 | 3.689 |

The first row is the old behaviour: a whole-brain cortex 3.4mm thick, which is
not an implausible-looking number, which is the problem.

## What changed

### 1. The CUDA solve knows its voxel size

`kelly_kapowski_cuda` already had an unused `voxel_size` parameter; nothing ever
passed it. `DiReCT.py` and `field_pial_prototype.solve_velocity_field` now both
read it from the reference image's header. Displacement fields stay in voxels
throughout — that is what `grid_sample` wants — and the voxel size is what
relates them to the millimetre-valued step size, thickness prior and output.

Passing `voxel_size=(1,1,1)` on 1mm data is **bit-identical** to the old
`voxel_size=None` path (verified: max |difference| exactly 0.0 on a phantom), so
nothing that already worked changes.

### 2. Smoothing sigmas are physical by default

ANTs uses inconsistent units for its own smoothing parameters: the gradient sigma
is in mm, while the hit/total and velocity-field variances are in voxels² with
`SetUseImageSpacing(false)`. At 1mm that inconsistency cannot be observed. Off
1mm it means the regulariser's width is set by the acquisition rather than by the
anatomy.

`sigma_units='mm'` (the new default) converts every sigma per axis before it
reaches a kernel. `sigma_units='voxel'` restores ANTs' literal behaviour for a
like-for-like comparison against it. The two are identical at 1mm isotropic.

The gradient sigma is additionally floored at 0.6 voxel per axis. A sub-voxel
Gaussian derivative kernel degenerates towards zero and the solve silently
returns an all-zero thickness map; on a 1×1×5mm acquisition a 1mm sigma is 0.2
voxel on the slice axis. The floor makes the differentiation kernel physically
anisotropic on that axis — nothing can recover detail the acquisition did not
sample — but the solve survives and says so.

`direction_masked_smooth_3d` and the `NormalGate` neighbourhood filter enumerate
an explicit (dz, dy, dx) box, so they take a per-axis radius now; a single radius
would make the physical neighbourhood the shape of the voxels.

### 3. The pial pipeline's distance field is in millimetres

Every site that did

```python
sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)   # VOXELS
gu  = grad / |grad|                                                 # unit in voxel-index space
```

now calls `wm_distance_field(seg, zooms)`, which returns three things:

* `sdt` — signed distance in **millimetres** (`sampling=zooms`),
* `step` — the **voxel** displacement whose effect on `sdt` is +1mm,
* `normal` — the unit outward normal in **physical** space.

Splitting `step` from `normal` matters. A displacement applied in voxel
coordinates to change `sdt` by a known number of millimetres must be the normal
divided by the voxel size. A *direction* compared against another direction —
the `NormalGate`'s dot product between neighbouring interface normals — must be
the physical unit normal, or angles are measured in a sheared space and two
genuinely opposed sulcal banks can come out with a positive dot product, which is
exactly the case that gate exists to catch. At 1mm isotropic the two arrays
coincide, which is how one served both roles.

Consequently `floor`, `floor_after`, `escape_clearance`, `damp_wm_floor`, the
in-WM sampling offset (0.25) and `detect_sulcal_csf_sheet`'s `max_ray_mm` /
`step_mm` all now mean millimetres on any grid. `build_no_push_mask`'s `radius`
became millimetres too, rounded to whole voxels off the finest axis (all
`binary_dilation` offers).

### 4. White-surface smoothing holds a physical scale

Taubin smoothing is defined on the mesh: each step averages a vertex with its
graph neighbours. Finer voxels give marching cubes smaller triangles, so a fixed
step count covers a smaller physical distance and the white surface comes out
systematically rougher for no reason but the acquisition. The diffusion length of
n averaging steps grows as √n × edge length, so `-ns auto` (now the default)
scales n as 1/edge², anchored to the 1mm/50-step operating point everything here
was tuned at. At 0.7mm that is 102 steps. An explicit integer still passes
through unchanged.

## What was NOT changed, and why

* **`invert_field`'s stopping thresholds** (max residual ≤ 0.1, mean ≤ 0.001) are
  in voxels and are ANTs' own. They are the only part of the solve that is not
  scale-covariant: rescaling every physical length of a problem by k reproduces k
  times the thickness to a relative 1e-3 rather than exactly, and the error grows
  with k because the same voxel tolerance buys less physical accuracy on a
  coarser grid. Left alone as an ANTs-compatibility matter.
* **`extract_stats.py`'s `nearest_distances > sqrt(3)`** looks like a millimetre
  threshold but is not: the KD-tree is built on voxel indices and √3 is one
  voxel's diagonal, i.e. an adjacency test. Voxel units are the right units
  there.
* **`extract_wm_contours`** marks a 1-voxel-thick band, so the seeding band is
  physically thinner at finer resolution. The thickness estimate normalises
  total/hit, so this largely cancels; it has not been isolated.
* **`conform.py`** still resamples to 1mm by default. The shipped DeepSCAN models
  are 2D slice networks trained at 1mm, and running them at native resolution is
  a model question, not a plumbing one. The route that avoids the conform is
  `seg2surf.sh`, which starts from a segmentation that already exists.

## Verification

**Scale covariance.** The strongest available check does not need ground truth:
the solver should be exactly covariant under rescaling. Take one array, solve it
with voxel size v and physical parameters (σ, step, prior), then again with
(k·v, k·σ, k·step, k·prior); the answer must be exactly k times the first.

| k | measured mean | expected | max relative error |
|---|---|---|---|
| 0.70 | 2.2303 | 2.2306 | 1.0e-3 |
| 0.50 | 1.5934 | 1.5933 | 6.6e-4 |
| 2.00 | 6.3737 | 6.3731 | 1.5e-3 |

The residue is `invert_field`'s voxel-unit tolerances, as above.

**1mm regression.** On a phantom, `voxel_size=None` and
`voxel_size=(1,1,1), sigma_units='mm'` agree to max |difference| 0.0.

Both are reproducible: `python dldirect/scripts/check_voxel_size.py`.

**End to end.** See `doc/beyond-1mm-results.md` for the 0.7mm-vs-1mm run on the
sample subject. The load-bearing result there is that both the white and the
pial surface sit at the minimum of their distance to the boundary they belong on
at 0.7mm (0.43mm and 0.72mm respectively, 96.9% / 83.7% of vertices within 1mm),
with the distance rising monotonically as the surface is displaced.

That check needed a new function. `check_surface_frame`'s tissue-fraction
statistic is not monotonic for the PIAL on an imported ground-truth segmentation
-- shifting it 8mm RAISES the score -- so it reports FAIL for a correct pial and
PASS for a misplaced one. `surface_frames.boundary_distance` measures the
distance directly and is monotonic; see its docstring for the measured shift
response.

## Caveats

* One subject, one segmentation. Nothing here establishes that 0.7mm thickness is
  *more accurate* than 1mm thickness — only that the pipeline now measures the
  same physical quantity at both, instead of a voxel-scaled one at one of them.
* The 0.7mm and 1mm arms of the comparison are not a controlled resolution
  experiment either: the 1mm arm's segmentation is a nearest-neighbour downsample
  of the 0.7mm one, so it differs in cortical detail as well as in grid. Neither
  arm is ground truth for the other.
* Anisotropic (thick-slice) data is handled by construction throughout, but has
  not been run. The gradient-sigma floor is the part most likely to need
  attention there.
