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

Both arms are too shallow where FreeSurfer's border is deep (-0.034) and too
deep where it is shallow (+0.040). An earlier version of this document read
that sign pattern as our per-pair depths being COMPRESSED toward the mean.
That was an inference, and the direct measurement contradicts it: the ratio
of our per-pair SD to FreeSurfer's is 1.063 median and above 1.0 in 79% of
hemispheres, i.e. slightly MORE dispersed, not less. What is real is the
bias split, not a compression.

## The g-scale sweep

`g = clip(median_z / scale, 0, 1)`, so a small scale saturates every pair at
g = 1 (all borders loosened at fundi, i.e. plain Potts) and a large one
spreads them down, making more borders pay full price as a gyral border
should. Swept at theta 2 / beta 8 over all 60 hemispheres, off the cached
unaries -- no GPU. Medians, ~75 parcel pairs per hemisphere:

| g-scale | r | MAE | bias | on FS-sulcal | on FS-crown | Dice | strays |
|---|---|---|---|---|---|---|---|
| 0.0 (no prior) | 0.766 | 0.1438 | +0.0146 | -0.028 | +0.054 | 0.8988 | 3 |
| 0.6 (current)  | 0.766 | 0.1448 | +0.0042 | -0.035 | +0.042 | 0.8987 | 3 |
| 1.5            | 0.767 | 0.1416 | -0.0009 | -0.041 | +0.042 | 0.8988 | 3 |
| **2.0**        | 0.769 | **0.1404** | -0.0023 | -0.043 | +0.041 | 0.8987 | 3.5 |
| 3.0            | 0.761 | 0.1426 | -0.0030 | -0.046 | +0.039 | 0.8988 | 3.5 |

Paired against 0.6, g = 2.0 cuts MAE in 45/60 hemispheres and the crown bias
in 47/60; g = 3.0 in 49/60 on both, with the correlation starting to fall.
Correlation is flat across the whole range (31-35/60 -- noise). Dice and
stray counts do not move at all.

**Use 1.5 to 2.0** if unbiased per-pair border depth is the goal: the bias
crosses zero there and MAE is at its minimum. But this is a 3% change in MAE,
and the sulcal/crown trade never separates -- pushing crown borders shallower
drags the sulcal ones shallower too. The prior RE-CENTRES the error; it does
not sharpen the distinction between the two kinds of border. Whatever limits
agreement with FreeSurfer at r = 0.77 is not reachable from this knob.

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

---

# Part two: the solver, the coupling, and where the unary comes from

Everything above tunes the smoothness term. This part measures the three
things that turned out to matter more, and records four ideas that failed.

## ICM is a poor optimiser of this energy

Annealing from ICM's own solution finds **3.92% lower energy on 10 of 10
hemispheres** (range 3.50-4.48%). The gap ICM leaves is larger than the gap it
closes from the unary. The reason is structural: shifting a border segment
needs a coordinated move whose intermediate states are uphill, and ICM only
takes moves that are downhill immediately.

The scale of the block is measurable. At beta 8, the pairwise term opposes a
single-vertex flip at a border by ~32 units, against a unary margin of ~0.4 --
a factor of 80. The unary at borders IS soft (margin 0.400 there against 2.046
cortex-wide); it is simply outvoted.

What the lower energy buys is modest and should not be oversold: median
agreement with FreeSurfer moved **-0.0005** (better in 4/10) and strays 3 -> 2
(better in 6/10, worse in 1/10). Rows in `data/mesh-crf-annealing-10.csv`.

Two practical findings came with it, both in `mesh_crf.banded_anneal`:

* **The interior cannot move.** Of the vertices a full anneal relabelled, 100%
  lay within 3 hops of a border (median 0, p99 1, max 2) and 12% outside their
  parcel's main component. Restricting to that set touches 28% of the mesh,
  runs 3.3x faster, and reached a LOWER energy than annealing everything.
* **ICM is not needed first.** Annealing from the unary argmax reaches the same
  energy to within 0.13% and saves 30% of the runtime.

## The sulcal weight and the model's uncertainty are unrelated fields

The CRF can only act where the unary is soft AND the weights are loose. Those
two conditions turn out to be nearly independent:

| inside the active band | median over 6 hemispheres |
|---|---|
| AUC, hull-depth z predicting where the unary is soft | 0.487 |
| AUC, curvature predicting it | 0.512 |
| lift of (soft AND sulcal) over chance | 0.95 |

Cortex-wide the lift looks like 1.24, but that is an artefact of the frozen
interior; conditioned on being near a border, depth carries no information
about whether the model is undecided there. This is not a defect -- the weight
encodes "a border is cheap here because this is a fundus", and was never meant
to track uncertainty, which already enters through the unary margin on the
other side of the same inequality. Adding softness to the weight as well would
count it twice.

But it does explain the day's flat sweeps: every knob on the smoothness term
adjusts the 32 by a few percent, on the wrong side of a factor of 80.

## beta, swept below 1 for the first time

| beta vs 8 | median change in the fundus gap | better in |
|---|---|---|
| 0.125 | -0.0770 | 0/6 |
| 0.25 | -0.0327 | 2/6 |
| 0.5 | -0.0255 | 2/6 |
| 1.0 | +0.0504 | 4/6 |
| 2.0 | +0.0036 | 3/6 |

reach/margin tracks beta linearly (1.0 at beta 0.125, 67 at beta 8), so beta is
the direct lever on that factor of 80 -- and pulling it down to parity does NOT
let the unary place borders better: beta below 0.5 is worst on the gap in all
six and worst on fragmentation where fragmentation exists. beta 1 is the best
cell (4/6, and strays 2 vs 3.5) but the per-hemisphere winner is inconsistent
(1, 8, 1, 8, 1, 2) and agreement moves 0.002 across a 64-fold change.
`data/mesh-crf-beta-sweep.csv`.

NOTE a confound in that table: a fixed temperature schedule anneals a beta
0.125 energy far hotter than a beta 8 one, since the cost gaps scale with beta.
`banded_anneal` now scales temperature by the arm's own median cost gap.

## Where the unary comes from matters more than any of it

`column_unary` walks white -> pial and samples along the chord. That chord
leaves the vertex's own bank for **2.6-2.9% of cortical vertices** at depth
0.5, rising to 6-10% near the pial. Integrating the DiReCT field instead does
NOT fix it (2.72% vs 2.64%, worse toward the pial): where the banks merge in
the segmentation, no path can stay on one side. The field's gated smoothing
refuses to AVERAGE across banks, which is not the same as a trajectory staying
on one.

`mesh_crf.ribbon_unary` removes the failure mode instead of mitigating it: read
the k nearest ribbon voxels, which sit a median 0.60 mm away, so there is no
path to leave the bank. Four hemispheres, same CRF:

| | ribbon vs column |
|---|---|
| fundus gap (FreeSurfer sulc, independent) | better **4/4** (+0.29, +0.14, +0.09, +0.06) |
| agreement with aparc | worse 3/4, all <= 0.003 |
| null vs FreeSurfer unknown | worse 4/4, by ~0.02 |
| stray components | worse 4/4 (20/4, 6/3, 1/0, 1/0) |
| runtime | **faster 4/4** (31 s vs 48 s median) |

`data/mesh-crf-unary-comparison.csv`. The fragmentation is dominated by one
hemisphere; the CRF clears ribbon islands well on three of four (79-89%) and
badly on one (35%).

### The averaging radius, and what the stray count is really counting

Swept k/sigma from 4/0.5 to 40/3.0 with the null region held fixed
(`data/mesh-crf-unary-radius-sweep.csv`). The fundus gap falls monotonically as
the radius grows -- 2.102 -> 1.930 (lh), 2.196 -> 2.024 (rh) -- while
post-inference stray components do not move (21 -> 19 lh, flat at 6 rh). Wider
averaging suppresses islands BEFORE inference (37 -> 28) but those are the ones
the CRF removes anyway. **k=4, sigma=0.5 is the default**: best gap on both
hemispheres, same strays, half the voxels read.

Inspecting the strays directly changes how the metric should be read. Only 1 of
7 components is enclosed by a single other label (and that one by Unknown, the
medial wall); the rest straddle a border between two parcels, median size 44
vertices, max 301. The same parcels fragment under both unaries -- bankssts and
pericalcarine lead both lists -- and FreeSurfer's own parcellation has 8 strays
on this subject. Most of the count is awkward DK geometry, not labelling noise,
so **it is a poor tiebreaker** and is over-weighted elsewhere in this document.

The null rule had to be geometric, not probabilistic: vertices that the column
nulls and a 3 mm range rule does not sit at median 1.89 mm from the nearest
ribbon voxel, against 0.59 mm for correctly labelled ones, and a 1.5 mm cutoff
separates them almost perfectly (1582 of 2165 caught, 3 of 145899 lost).
Cortical MASS does not separate them at all.

## Cost, and why nearly all of it is avoidable

Measured per case, both hemispheres, on top of a ~69 s pipeline:

| stage | s |
|---|---|
| writing 94 logit volumes | 21.4 |
| sampling them back | 33.4 |
| mesh adjacency | 2.8 |
| rasterize pial + hull EDT | 6.6 |
| geodesic z-score | 0.4 |
| edge weights + prior + inference | 12.1 |
| ribbon propagation | 5.7 |

Two thirds is moving logits through 389 MB of gzipped NIfTI. Computed inside
the pipeline while the model's output is still in memory -- ~190 MB holds the
posterior at every ribbon voxel -- the realistic cost is **~25 s/case**, and the
mesh adjacency is already built by `propagate_pial`.

## Failed ideas, recorded so they are not retried

* **A finer depth signal.** z at 10 geodesic smoothing iterations loses to z at
  50 on 4/4 hemispheres, and moves FEWER vertices. Coarser is better here.
* **Curvature as the weight.** Worst single signal (0/4), and in the band its
  AUC for softness is 0.512 -- no information. Combining it with z by max or
  mean wins on one hemisphere each and loses elsewhere; the per-hemisphere
  winner is different every time and the whole spread is ~2%.
* **Field-trajectory sampling.** See above: no better than the chord.
* **Lowering beta to balance the terms.** See above: 0/6.

One robust oddity, unexplained: the fundus gap measured in CURVATURE is
negative in 24 of 24 arm/hemisphere rows (median -0.049). These borders sit
reliably deep and reliably LESS concave than average -- i.e. on sulcal walls,
not on curvature ridges. "Moves boundaries into deep regions" is the claim the
evidence supports; "moves boundaries to the fundus" is not.
