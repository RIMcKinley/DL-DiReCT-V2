# The mesh CRF on 30 subjects

Everything in `mesh_crf` had been chosen on one subject. This is the first
cohort measurement: 30 OASIS-3 subjects, one session each, taken every 34th
subject of the 1042 that have both our `field_pial_sigma0.65` surfaces and a
FreeSurfer recon on the NAS — so they are spread across the cohort and no
subject appears twice. 60 hemispheres. Run with `mesh_crf_cohort.py`; the
per-hemisphere rows are in `data/mesh-crf-30-subjects.csv` and the subject
list in `data/mesh-crf-30-subjects.txt`.

## The first thing to know: subjects vary far more than settings

Unary Dice against FreeSurfer's aparc runs from **0.829 to 0.921**, median
0.899. The largest difference between any two CRF settings is **0.0005**.

A single-subject comparison cannot see past that, and several earlier
conclusions in this line of work were drawn from one. Judge by sign
consistency over the 60 hemispheres, not by a mean.

## The CRF moves boundaries into the fundi, on every hemisphere

Paired against the unary, median delta and the count of hemispheres improved:

| setting | Dice | gap in our z | gap in FreeSurfer's sulc | strays |
|---|---|---|---|---|
| theta 0.5, beta 1 | +0.0004 (45/60) | +0.0200 (**60/60**) | +0.165 (**60/60**) | 0 median, 26 worse |
| theta 1, beta 4   | +0.0003 (42/60) | +0.0243 (**60/60**) | +0.175 (**60/60**) | +3 median, 40 worse |
| theta 2, beta 8   | +0.0005 (46/60) | +0.0213 (**60/60**) | +0.160 (58/60)      | **-7 median, 1 worse** |

`gap_z` is measured in the same z-score the edge weights are built from, so a
CRF driven hard enough raises it by construction. `gap_sulc` is measured in
FreeSurfer's own `?h.sulc`, which the energy cannot pay for, and it moves
with it on essentially every hemisphere. That is the result: the effect is
real and it replicates.

Against the anchor, in medians: the unary sits at z 0.092 / sulc 1.447 and
FreeSurfer's own aparc at z 0.117 / sulc 1.671, so the CRF closes about 85%
of the z deficit and 75% of the sulc deficit. Hemispheres whose boundaries
are at least as sulcal as FreeSurfer's go from 13% to 60% (z) and 18% to 40%
(sulc).

**theta 2, beta 8 is the setting to use.** Same fundus gain as the others,
and it REMOVES stray components — median -7, one hemisphere of 60 worse —
where theta 1 / beta 4 makes 40 of 60 worse. Dice should not decide this;
it is inert across all three.

### What this does NOT show

At theta 2 the edge weights are nearly uniform (median 0.981, none below
0.5), so the best-behaved cell is the one where the sulcal gating
contributes least, and what is running there is close to plain Potts. The
boundaries move to the fundi; whether the SULCAL WEIGHTING is what moves
them needs the flat-weight control at matched beta, crossing both places
sulcality enters the energy (the edge weights and the per-border prior).

## The gated vote holds up

Sampling only non-null voxels (`column_unary`, `vote_gate`) was also chosen
on one subject. Over 30:

| | median | range |
|---|---|---|
| our null fraction | 5.78% | 4.78 – 6.64% |
| FreeSurfer unknown fraction | 6.04% | 5.03 – 8.17% |
| Dice, our null vs FS unknown | **0.956** | 0.886 – 0.974 |

## A case to look at

`OAS30063_ses-d0160_run-01` is the floor in both hemispheres (0.829 / 0.833)
and also carries the most stray components (27 / 16). That pattern is
case-level, not method-level, and it should go through the segmentation QC
channels rather than be treated as a CRF result.
