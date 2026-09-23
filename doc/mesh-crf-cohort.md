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

## The gating is what moves them (control run)

The obvious objection to the table above is that at theta 2 the edge weights
are nearly uniform (median 0.981, none below 0.5), so the best-behaved cell
is the one where the sulcal gating contributes least -- what runs there is
close to plain Potts. The control settles it: 60 hemispheres, four arms per
cell, crossing GATED vs FLAT edge weights and the SULCAL vs UNIFORM per-border
prior. The flat arm uses mean(w_gated), not 1.0, so total pairwise mass
matches and the only difference is WHERE the smoothness sits.

Gated minus flat, at matched beta and matched G:

| cell | G | gap in z | gap in FreeSurfer sulc |
|---|---|---|---|
| theta 0.5, beta 1  | sulcal  | +0.0117 (**60/60**) | +0.0256 (56/60) |
| theta 0.5, beta 1  | uniform | +0.0231 (**60/60**) | +0.0451 (53/60) |
| theta 0.25, beta 4 | sulcal  | +0.1253 (**60/60**) | +0.5892 (**60/60**) |
| theta 0.25, beta 4 | uniform | +0.0485 (**60/60**) | +0.0708 (59/60) |
| theta 2, beta 8    | sulcal  | +0.0135 (**60/60**) | +0.0283 (52/60) |
| theta 2, beta 8    | uniform | +0.0250 (**60/60**) | +0.0494 (58/60) |

The gating wins the z gap in 360 of 360 comparisons and FreeSurfer's sulc in
52-60 of 60 in every cell. Flat weights alone do move borders sulcally
(+0.008 z, +0.12-0.13 sulc -- plain Potts shortens borders, and short borders
tend to land in fundi), but the gating adds a third again on top at theta 2
and quadruples it at theta 0.25. Rows in `data/mesh-crf-30-control.csv`.

Note theta 0.25 / beta 4 with the sulcal prior: the largest fundus gain of
any arm (+0.59 in sulc) and +240 stray components. That combination is not
usable.

## The per-border prior is neutral, and the aggregate gap cannot score it

The uniform prior gives a LARGER aggregate gap than the sulcal one (at
theta 2 / beta 8, +0.185 vs +0.160 in sulc), which looks like G costing
fundus depth. It is not that simple, and the aggregate gap is the wrong
measure: G exists so that a pair of parcels whose true border runs over a
gyral crown is NOT pushed into a fundus, so part of that "loss" is correct
behaviour.

Scored where G claims to act -- per parcel pair, against the median depth of
FREESURFER's own border for that same pair, 45 hemispheres:

| G | r with FS | MAE | bias | on FS-sulcal borders | on FS-crown borders |
|---|---|---|---|---|---|
| sulcal  | 0.765 | 0.146 | +0.002 | -0.034 | **+0.040** |
| uniform | 0.765 | 0.144 | +0.015 | -0.028 | **+0.053** |

G does what it is supposed to -- it cuts the over-deepening of borders that
FreeSurfer puts on a crown, and that pulls the overall bias from +0.015 to
+0.002 -- but it buys no agreement: same correlation, same error, better in
19/45 and 18/45 hemispheres respectively. Coin flip. Keep it or drop it; the
evidence does not decide.

The larger signal in that table belongs to neither arm. Our per-pair border
depths are COMPRESSED toward the mean relative to FreeSurfer's: too shallow
where FreeSurfer's border is deep (-0.034), too deep where it is shallow
(+0.040). The parcellation under-differentiates which borders are sulcal, by
more than any setting here moves. `--g-scale` (0.6) is the knob that controls
how far G may separate the two classes, and it has never been swept.

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
