# 0.7mm results on the sample subject

Companion to `doc/beyond-1mm.md`, which describes the changes. This records what
was actually run and measured.

**Subject.** `T1NAT__PT030_OpenNeuro_ds000031__sub-01__ses-07`, 0.7mm isotropic,
256×320×320, oblique affine (~2° off axis). Ground-truth segmentation
`__seg_GT.nii.gz` and the model prediction `.nii.gz`, both 94-class index maps in
irrepunet's `seg_label_map._full()` scheme, with the T1 `__image.nii.gz`.

The label scheme was auto-detected. Both candidate 94-class schemes place every
cortical parcel identically and are separated only by the ordering of the 26
subcortical/midline classes, so the laterality test had to be extended to those
(and to requiring midline structures to actually be midline) before it
discriminated: 99.5% for `v0-fs-sorted` against 88.0% for `v0-model-order`.
Under the wrong one, class 8 would have been Left-Hippocampus rather than
Brain-Stem.

## Arms

| arm | grid | how |
|---|---|---|
| native | 191×206×270 @ 0.7mm | `seg2surf.sh` (LIA reorientation only; exact) |
| 0.7mm | 191×206×270 @ 0.7mm | `--voxel-size 0.7` (identity here — see below) |
| 1.0mm | 134×144×189 @ 1.0mm | `--voxel-size 1.0` |

The 0.7 and 1.0 arms keep the input's direction cosines and differ in resolution
alone, so they are a controlled pair. Verified by mapping structure centroids
back to world coordinates: the 0.7mm arm reproduces the input exactly (0.00mm,
+0.0% volume for thalamus, cerebral WM and corpus callosum); the 1.0mm arm is
within 0.31mm, with volumes −0.2% (cerebral WM), −0.9% (thalamus) and −9.4%
(corpus callosum, which is thin and loses most to a nearest-neighbour
downsample).

## The solver

Same 0.7mm inputs, three unit conventions, mean over cortex voxels:

| configuration | mean | median | p95 |
|---|---|---|---|
| before this branch (`voxel_size=None`) | 3.418 mm | 3.390 | 5.175 |
| `voxel_size` set, `sigma_units='voxel'` | 2.651 mm | 2.623 | 3.915 |
| `voxel_size` set, `sigma_units='mm'` | 2.415 mm | 2.384 | 3.689 |

The first row is what the shipped code produced on this data. 1/0.7 = 1.43, and
3.418/2.415 = 1.42.

Stock `DiReCT.py --cuda` end to end: 2.540 mm at 1.0mm (15.8s), 2.415 mm at
0.7mm (75.3s).

## The surfaces

`field_pial_prototype` with its own solve (`--write-thickness --skip-naive`,
sigma 1.0, dip 0.95, WM from the white surface at `--wm-pv 3`).

| | 1.0mm | 0.7mm |
|---|---|---|
| vertices (lh / rh) | 147,882 / 148,424 | 303,964 / 303,596 |
| Taubin steps (`-ns auto`) | 50 | 102 |
| frame check, mean \|dist to WM boundary\| | 0.456 / 0.457 mm | 0.439 / 0.441 mm |
| no-cortex vertices | 4.9% / 4.6% | 4.5% / 4.2% |
| transit % | 1.82 / 2.08 | 1.50 / 1.55 |
| crossed CSF (count) | 2415 / 2260 | 2942 / 2895 |
| end in WM, free vertices | 604 / 499 | 1048 / 901 |
| normal reversal % | 0.035 / 0.038 | 0.073 / 0.053 |
| self-intersecting faces % | 0.484 / 0.568 | 0.802 / 0.865 |
| mean displacement | 3.129 / 3.167 mm | 2.850 / 2.887 mm |
| thickness map mean (non-zero) | 2.629 mm | 2.528 mm |
| `extract_stats` mean thickness | 2.480 mm | 2.394 mm |

Mesh-quality metrics (normal reversal, self-intersections) are worse at 0.7mm.
No controlled experiment isolates why, and there are several differences between
the arms at once (vertex count, triangle size, the segmentation's own detail,
the Taubin step count), so no cause is claimed here.

### Surface placement

The pipeline's usual placement check, `surface_frames.check_surface_frame`,
reports FAIL for the pial in **both** arms (53.7%/54.3% at 0.7mm, 48.0%/48.6% at
1.0mm, against a 62% threshold). That verdict is not usable on this input, and
the shift test shows why — the statistic is not monotonic in displacement for the
pial here:

| lh, tissue fraction % | 0mm | 1mm | 2mm | 3mm | 5mm | 8mm | 12mm |
|---|---|---|---|---|---|---|---|
| white (0.7mm arm) | **96.2** | 95.7 | 94.7 | 90.2 | 82.9 | 80.2 | 79.7 |
| pial (0.7mm arm) | 53.7 | 60.4 | 67.2 | 72.5 | 77.4 | **78.5** | 78.1 |

The white surface behaves exactly as that function documents. The pial's optimum
is at +8mm, so a correct pial fails and a badly misplaced one passes. The
threshold was calibrated against a DeepSCAN segmentation, whose labelled volume
is surrounded by unlabelled CSF and skull; an imported ground-truth parcellation
stops at the pial boundary, so burying the surface in tissue raises the score.

`surface_frames.boundary_distance` was added to measure the thing directly — the
distance to the boundary the surface belongs on (WM/non-WM for white,
brain/background for pial). It is monotonic, and both surfaces sit at its
minimum in both arms:

| arm | hemi | surface | 0mm | 1mm | 2mm | 3mm | 5mm | 8mm | 12mm | within 1mm @0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.7mm | lh | white | **0.431** | 0.837 | 1.225 | 1.527 | 1.927 | 2.229 | 2.436 | 96.9% |
| 0.7mm | lh | pial | **0.724** | 0.970 | 1.277 | 1.617 | 2.261 | 2.877 | 3.458 | 83.7% |
| 0.7mm | rh | white | **0.437** | 0.826 | 1.204 | 1.504 | 1.978 | 2.572 | 3.352 | 96.9% |
| 0.7mm | rh | pial | **0.758** | 1.021 | 1.362 | 1.736 | 2.421 | 3.060 | 3.715 | 82.5% |
| 1.0mm | lh | white | **0.447** | 0.965 | 1.371 | 1.681 | 2.092 | 2.392 | 2.604 | 96.3% |
| 1.0mm | lh | pial | **0.775** | 1.052 | 1.382 | 1.730 | 2.390 | 3.041 | 3.637 | 80.2% |
| 1.0mm | rh | white | **0.453** | 0.954 | 1.352 | 1.663 | 2.149 | 2.750 | 3.537 | 96.3% |
| 1.0mm | rh | pial | **0.807** | 1.121 | 1.490 | 1.877 | 2.586 | 3.270 | 3.980 | 79.0% |

This is the load-bearing result: the 0.7mm surfaces are placed correctly, not
merely produced.

### Agreement between arms

Symmetric nearest-vertex distance in world coordinates. A weak metric on folded
surfaces (see `field_pial_prototype`'s module docstring: a 12mm-misregistered
pial still scored 1.8mm), reported for scale only.

| hemi | surface | mean | median | p95 | max |
|---|---|---|---|---|---|
| lh | white | 0.381 | 0.370 | 0.642 | 2.166 |
| lh | pial | 0.407 | 0.383 | 0.733 | 5.592 |
| rh | white | 0.366 | 0.356 | 0.616 | 2.487 |
| rh | pial | 0.399 | 0.376 | 0.718 | 3.942 |

### Per-parcel thickness

70 Desikan-Killiany parcels, `extract_stats.py`:

* correlation across parcels r = 0.9916
* 0.7mm mean 2.451 mm, 1.0mm mean 2.537 mm
* difference (0.7 − 1.0): mean −0.087, median −0.081, sd 0.072, |max| 0.297 mm
* lh-MeanThickness 2.377 vs 2.463; rh-MeanThickness 2.411 vs 2.497

Largest differences: rh-temporalpole (−0.297), rh-fusiform (−0.281),
rh-parsorbitalis (−0.206), lh/rh-rostralanteriorcingulate (−0.205 / −0.204).

The 0.7mm arm is systematically thinner. The two arms differ in resolution AND
in the segmentation the 1mm arm was downsampled from, so this comparison does not
attribute that offset to either, and neither arm is ground truth for the other.

## Both segmentations, native grid, through `seg2surf.sh`

The GT and the model prediction were each run end to end on the native 0.7mm
grid with the shipped driver, no hand steps:

```
seg2surf.sh -s PT030_GT   ..._seg_GT.nii.gz ..._image.nii.gz out_gt
seg2surf.sh -s PT030_pred ....nii.gz        ..._image.nii.gz out_pred
```

Both auto-detected `v0-fs-sorted` (99.5% / 99.6% laterality), cropped to
191×206×270 and 190×204×270, used 102 Taubin steps, and completed.

| run | hemi | surface | mean dist | median | within 1mm | vertices |
|---|---|---|---|---|---|---|
| GT | lh | white | 0.431 | 0.269 | 96.9% | 303,964 |
| GT | lh | pial | 0.724 | 0.436 | 83.7% | 303,964 |
| GT | rh | white | 0.437 | 0.268 | 96.9% | 303,596 |
| GT | rh | pial | 0.758 | 0.452 | 82.5% | 303,596 |
| pred | lh | white | 0.379 | 0.199 | 96.7% | 300,986 |
| pred | lh | pial | 0.695 | 0.374 | 85.7% | 300,986 |
| pred | rh | white | 0.380 | 0.200 | 96.8% | 303,566 |
| pred | rh | pial | 0.706 | 0.369 | 85.2% | 303,566 |

Per-parcel thickness, GT vs prediction over 70 parcels: r = 0.9592, mean
difference +0.025 mm, |max| 0.409 mm. lh/rh MeanThickness 2.377/2.410 (GT)
against 2.433/2.361 (prediction).

The native run reproduces the `--voxel-size 0.7` arm to within rounding
(`extract_stats` mean thickness 2.3938 vs 2.3939; crossed-CSF counts 2939/2891
vs 2942/2895), which is expected: at the input's own voxel size the resampling
step is the identity, so the two differ only in going through the reorientation
path.

## Caveats

One subject, one segmentation, no repeat runs. The 0.7-vs-1.0 comparison is not a
resolution experiment in the controlled sense: the 1mm arm's parcellation is a
nearest-neighbour downsample of the 0.7mm one, so cortical detail differs along
with the grid.
