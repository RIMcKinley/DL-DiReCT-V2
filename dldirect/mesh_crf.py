"""A contrast-sensitive CRF over the white surface, loosened at sulcal fundi.

STATUS: research prototype. Nothing imports it; run it directly.

    E(L) = sum_i  U_i(L_i)  +  beta * sum_(i,j)  w_ij * [L_i != L_j]

U is a per-vertex cost over the 68 Desikan-Killiany labels. w comes from
hull_depth.crf_edge_weights, which collapses on a sulcal fundus, so the
smoothness term pulls neighbours together within a gyral bank and lets them
disagree across a fundus.

WHAT THIS CAN AND CANNOT DO
---------------------------
The unary is built by VOTING the existing voxel parcellation into each
vertex's neighbourhood, because nothing on disk carries per-parcel
probabilities: both aparc.atlas+aseg and softmax_seg are integer label maps,
and the model's own logits are tissue classes, not parcels.

So the CRF can only clean up what the nearest-voxel propagation got noisy --
a boundary that the atlas put in the wrong place is invisible to this energy,
because no term knows better. Expect boundaries that snap to fundi and agree
slightly better with an independent parcellation; do not expect a parcel to
move where the atlas was confidently wrong.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 1. unary
# ---------------------------------------------------------------------------

def soft_unary(verts_vox, parc, labels, radius_vox=2.0, sigma=1.0, eps=1e-3):
    """(-log p) per vertex per label, from distance-weighted voxel votes.

    A hard nearest-voxel lookup gives each vertex one label and no measure of
    how close the call was -- exactly the information a CRF needs at a
    boundary. Voting a small neighbourhood instead, with a Gaussian weight in
    distance, recovers it: a vertex well inside a parcel votes almost purely,
    one straddling a border splits.
    """
    r = int(np.ceil(radius_vox))
    off = np.array([(dz, dy, dx)
                    for dz in range(-r, r + 1)
                    for dy in range(-r, r + 1)
                    for dx in range(-r, r + 1)
                    if dz * dz + dy * dy + dx * dx <= radius_vox ** 2])
    wts = np.exp(-(off ** 2).sum(1) / (2.0 * sigma ** 2))
    idx = {int(l): k for k, l in enumerate(labels)}
    base = np.rint(np.asarray(verts_vox)).astype(int)
    votes = np.zeros((len(base), len(labels)), np.float32)
    shape = np.array(parc.shape)
    for o, wt in zip(off, wts):
        q = np.clip(base + o, 0, shape - 1)
        lab = parc[q[:, 0], q[:, 1], q[:, 2]]
        for l, k in idx.items():
            votes[lab == l, k] += wt
    p = votes / np.maximum(votes.sum(1, keepdims=True), 1e-9)
    return -np.log(p + eps).astype(np.float32), votes


def model_unary(verts_vox, logit_dir, names, sigma=1.0, eps=1e-6,
                temperature=1.0, topk=None):
    """(-log softmax) per vertex per label, from the SEGMENTATION MODEL's own
    per-parcel logits.

    The pipeline discards these: DeepSCAN computes a logit for every class,
    softmax_seg is the argmax of them, but SAVE_LOGITS_FILTER keeps only nine
    tissue and subcortical labels. Re-running with the filter open writes all
    68 parcels, and their softmax is a genuine probabilistic unary -- it
    carries the model's own uncertainty at a boundary, which the voxel-vote
    unary can only approximate from hard labels.

    Sampled trilinearly at the vertex rather than nearest-voxel, so the unary
    varies smoothly along the surface.
    """
    from scipy.ndimage import map_coordinates
    pos = np.asarray(verts_vox).T
    L = []
    for n in names:
        f = os.path.join(logit_dir, 'seg_%s.nii.gz' % n)
        if not os.path.exists(f):
            raise SystemExit('missing logit volume %s -- re-run the model with '
                             'SAVE_LOGITS_FILTER = None' % f)
        vol = np.asarray(nib.load(f).dataobj, dtype=np.float32)
        L.append(map_coordinates(vol, pos, order=1, mode='nearest'))
    lg = np.stack(L, axis=1)                       # [N, n_labels]
    # Temperature: T < 1 sharpens the softmax, T > 1 flattens it. This model's
    # softmax over 34 labels leaks ~35% of its mass to parcels that are not
    # even adjacent to the winner, while still ranking the winner 5:1 ahead of
    # the runner-up -- diffuse calibration rather than genuine local doubt.
    lg = lg / float(temperature)
    lg = lg - lg.max(1, keepdims=True)             # softmax, stably
    p = np.exp(lg)
    p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
    if topk:
        # Zero everything outside the top-k and renormalise. This is a HARD
        # restriction of the label set per vertex: the excluded labels get a
        # large finite cost, so the pairwise term chooses among the survivors
        # instead of ranging over all 34.
        cut = np.partition(p, -topk, axis=1)[:, -topk][:, None]
        p = np.where(p >= cut, p, 0.0)
        p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
    return -np.log(p + eps).astype(np.float32), p.astype(np.float32)


META_CLASSES = ('Left-Cerebral-Cortex', 'Right-Cerebral-Cortex',
                'left-hemisphere', 'right-hemishpere', 'brain')


def column_unary(white_tkr, pial_tkr, logit_dir, names, tovox,
                 depths=tuple(round(0.1 * k, 2) for k in range(1, 10)),
                 eps=1e-6, include_null=True, vote_gate=True):
    """Unary from the model's logits sampled along each vertex's CORTICAL COLUMN.

    Sampling at the white vertex itself is wrong, and measurably so. A white
    vertex sits ON the WM/GM interface, so the model's argmax there is
    routinely not a cortical parcel at all: at one site examined, 100% of 691
    vertices had a global argmax of `brain` or `left-hemisphere` -- whole-brain
    meta-classes in the 99-class output. Restricting the softmax to the 34
    cortical parcels then FORCES a label the model never chose, and invented a
    parahippocampal patch inside fusiform that appears nowhere in the model's
    own voxelwise segmentation.

    ONLY NON-NULL SAMPLES VOTE (vote_gate, the default). A sample counts
    only if its OWN argmax over the 94 real classes is a parcel of this
    hemisphere; the vertex distribution is the mean of the voting samples,
    and null wins only where NO sample voted. Averaging the whole column
    instead lets a few white-matter or CSF samples drag a vertex to null:
    over the nine depths below, the plain mean called 20.8% of the hemisphere
    null against FreeSurfer's 6.4%, and the midpoint alone 13.9%. Gating the
    vote gives 5.9% (lh) / 5.6% (rh), 99.7% and 98.0% of it inside
    FreeSurfer's own unknown region -- Dice against that region 0.953 / 0.910,
    against 0.625 / 0.634 for the midpoint.

    Measured on OAS30001_ses-d0129_run-01, scored on FS-labelled cortical
    vertices, mean Dice over the 34 parcels:

        midpoint (0.5,), plain mean    0.8755 lh / 0.8847 rh
        gated vote, depths 0.1..0.9    0.9020 lh / 0.9168 rh
        null column dropped entirely   0.9022 lh / 0.9180 rh

    i.e. the gate recovers what forcing a parcel everywhere would buy, while
    keeping a null label that still wins on the medial wall.

    Instead, walk from the white vertex to its corresponding pial vertex --
    the propagation preserves vertex identity, so the column is known exactly
    -- and take a weighted average of the cortical-parcel probabilities along
    it. Each sample is weighted by the mass the FULL softmax puts on cortical
    parcels there, so a point still inside white matter contributes almost
    nothing and a point in the middle of the ribbon dominates.

    A vertex whose whole column is non-cortical ends up with a near-flat
    distribution rather than a confident wrong answer, which is the honest
    outcome.
    """
    from scipy.ndimage import map_coordinates
    import glob
    files = sorted(glob.glob(os.path.join(logit_dir, 'seg_*.nii.gz')))
    if not files:
        raise SystemExit('no logit volumes in %s' % logit_dir)
    allnames = [os.path.basename(f)[4:-7] for f in files]
    # DeepSCAN's own derivation drops the last `label_num_ignore` classes
    # BEFORE the argmax (see DeepSCAN_Anatomy_Newnet_apply: logit_sm =
    # logit[:-NUM_IGNORE_LABELS]). They are parents, not alternatives:
    # 'brain' and the two hemispheres are true of every brain voxel, and
    # Left/Right-Cerebral-Cortex is the parent of all 34 parcels. Leaving them
    # in the softmax put ~86% of the mass on them and starved every real
    # class, which is what forced a parahippocampal patch into fusiform.
    keep = [k for k, n in enumerate(allnames) if n not in META_CLASSES]
    files = [files[k] for k in keep]
    allnames = [allnames[k] for k in keep]
    cidx = [allnames.index(n) for n in names if n in allnames]
    if len(cidx) != len(names):
        missing = [n for n in names if n not in allnames]
        raise SystemExit('logits missing for %s' % missing[:3])
    w = np.asarray(white_tkr, float); p = np.asarray(pial_tkr, float)
    pos = [tovox(w + t * (p - w)).T for t in depths]
    n = len(w)
    L = np.empty((len(depths), n, len(files)), np.float32)
    for k, f in enumerate(files):
        vol = np.asarray(nib.load(f).dataobj, dtype=np.float32)
        for di, q in enumerate(pos):
            L[di, :, k] = map_coordinates(vol, q, order=1, mode='nearest')
    L -= L.max(2, keepdims=True)
    P = np.exp(L)
    P /= np.maximum(P.sum(2, keepdims=True), 1e-12)     # softmax over the 94 real classes
    Pc = P[:, :, cidx]                                  # this hemisphere's parcels
    wt = Pc.sum(2)                                      # how cortical each sample is
    # a sample votes iff the model's own call there is a parcel of this
    # hemisphere -- not merely that some cortical mass exists
    voter = np.isin(P.argmax(2), np.asarray(cidx))      # (D, N)

    if not include_null:
        num = (Pc * wt[:, :, None]).sum(0)
        prob = num / np.maximum(wt.sum(0), 1e-12)[:, None]
        prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
        return (-np.log(prob + eps)).astype(np.float32), prob.astype(np.float32), wt

    # With a NULL label the column is averaged UNIFORMLY, not weighted towards
    # whatever cortex it happens to contain: null has to be able to win. Its
    # probability is everything the model did not put on a parcel of this
    # hemisphere -- white matter, hippocampus, ventricle, the other
    # hemisphere. Forcing those vertices into the nearest cortical parcel is
    # what produced a parahippocampal patch inside fusiform: over all 94
    # classes, parahippocampal won at 38 of 691 vertices there, but won the
    # cortex-only restriction at 213.
    if vote_gate:
        nvote = voter.sum(0)
        prob = ((Pc * voter[:, :, None]).sum(0)
                / np.maximum(nvote, 1)[:, None])
        prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
        # null is not a probability here but a verdict: no sample voted
        prob = np.concatenate([prob, (nvote == 0).astype(prob.dtype)[:, None]], 1)
        prob[nvote == 0, :-1] = 0.0
        prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
        return (-np.log(prob + eps)).astype(np.float32), prob.astype(np.float32), wt

    prob = Pc.mean(0)
    null = np.maximum(1.0 - prob.sum(1), 0.0)
    prob = np.concatenate([prob, null[:, None]], axis=1)
    prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
    return (-np.log(prob + eps)).astype(np.float32), prob.astype(np.float32), wt


# ---------------------------------------------------------------------------
# 2. inference
# ---------------------------------------------------------------------------

def border_sulcality_prior(ids, edges, z, labels, min_edges=30, scale=0.6):
    """g[a, b] in [0, 1]: how sulcally-defined the border between two parcels is.

    Not every atlas border follows a sulcus. Measured on this data, the DK
    borders span both extremes -- lateralorbitofrontal/parstriangularis sits
    at median sulc +13.1 while parsorbitalis/parstriangularis sits at -7.5 --
    so loosening the smoothness prior at every fundus is right for roughly
    half of them and wrong for the rest.

    g is taken from the median edge-z along each border in a labelling that
    is NOT the reference, so no information leaks from the thing being
    scored. A cohort-level prior would be better; this is the one-subject
    stand-in.

    Borders with fewer than `min_edges` get g = 1 (behave as before), since
    their median is not worth trusting either way.
    """
    idx = {int(l): k for k, l in enumerate(labels)}
    n = len(labels)
    a, b = ids[edges[:, 0]], ids[edges[:, 1]]
    ze = np.maximum(z[edges[:, 0]], z[edges[:, 1]])
    sel = (a != b) & np.isin(a, labels) & np.isin(b, labels)
    from collections import defaultdict
    acc = defaultdict(list)
    for x, y, v in zip(a[sel], b[sel], ze[sel]):
        acc[(idx[int(x)], idx[int(y)])].append(v)
    G = np.ones((n, n), np.float32)
    for (i, j), v in acc.items():
        if len(v) >= min_edges:
            g = float(np.clip(np.median(v) / scale, 0.0, 1.0))
            G[i, j] = G[j, i] = g
    np.fill_diagonal(G, 0.0)              # no cost for agreeing
    return G


def icm_pairwise(unary, edges, w, G, beta=1.0, n_iter=30, verbose=False):
    """ICM with a label-pair-dependent pairwise term.

        cost(i,j,a,b) = [a != b] * (1 - G[a,b] * (1 - w_ij))

    G = 1 recovers the plain weighted Potts; G = 0 for a pair makes that
    border ignore the sulcal weighting entirely and pay full price, which is
    what a gyral border should do.
    """
    n, L = unary.shape
    u = 1.0 - np.asarray(w, float)                  # how much a fundus loosens
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    U = sp.csr_matrix((np.concatenate([u, u]), (rows, cols)), shape=(n, n))
    Acnt = sp.csr_matrix((np.ones(2 * len(edges)), (rows, cols)), shape=(n, n))
    ncount = np.asarray(Acnt.sum(1)).ravel()
    lab = unary.argmin(1)
    best, best_e = lab.copy(), np.inf
    for it in range(n_iter):
        onehot = sp.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, L))
        C = np.asarray((Acnt @ onehot).todense())   # neighbours carrying m
        S = np.asarray((U @ onehot).todense())      # loosening toward m
        smooth = (ncount[:, None] - C) - (S @ G.T)
        cost = unary + beta * smooth
        new = cost.argmin(1)
        e = float(unary[np.arange(n), new].sum()
                  + beta * 0.5 * smooth[np.arange(n), new].sum())
        if e < best_e:
            best_e, best = e, new.copy()
        changed = int((new != lab).sum())
        lab = new
        if changed == 0:
            break
    return best, best_e


def ribbon_unary(white_tkr, parc, lo, hi, post, names, allnames, totkr,
                 k=8, sigma=1.0, null_d=1.5, gated=True, eps=1e-6):
    """Unary from the nearest RIBBON VOXELS, distance-weighted.

    The alternative to column_unary, and measurably better where it matters
    most. column_unary walks white -> pial and samples along the way, and that
    chord can leave the vertex's own bank: measured on four hemispheres, 2.6-2.9%
    of cortical vertices have their depth-0.5 sample taken in a DIFFERENT parcel
    than the vertex sits in, rising to 6-10% near the pial. Integrating the
    DiReCT field instead of the chord does NOT fix it (2.72% vs 2.64%, and worse
    toward the pial) -- where the banks merge in the segmentation, no path can
    stay on one side.

    Reading the nearest ribbon voxels instead removes the failure mode rather
    than mitigating it: the nearest one sits a median 0.60 mm away, so there is
    no path to leave the bank at all. It is also defined on every vertex, where
    the column question is well posed for only ~45% of them (a white vertex
    frequently sits on a voxel the parcellation calls white matter).

    NULL AND AVERAGING ARE SEPARATE RADII, deliberately. null_d is a property of
    the VERTEX (no ribbon voxel near it at all -> medial wall); sigma/k set how
    much the posterior is smoothed. Conflating them, as a single cutoff does,
    makes the null region move whenever the smoothing is retuned.

    null_d = 1.5 mm from measurement, not taste: vertices that column_unary nulls
    and a 3 mm rule does not sit at median 1.89 mm from the nearest ribbon voxel,
    while correctly labelled vertices sit at 0.59 mm (p95 0.86). A 1.5 mm cutoff
    nulls 1582 of those 2165 and 3 of 145899 correctly labelled ones. Cortical
    MASS does not separate them at all (0 of 2165 below 0.10), so the rule has to
    be geometric.

    Measured against column_unary on four hemispheres, same CRF: the independent
    fundus gap (FreeSurfer's sulc) is better 4/4, agreement with aparc is within
    0.003, null-vs-unknown Dice is ~0.02 worse, stray components are worse 4/4
    (dominated by one hemisphere: 20 vs 4, then 6 vs 3, 1 vs 0, 1 vs 0), and it
    runs in 31 s against 48 s.

    `post` is the softmax over the real classes AT THE RIBBON VOXELS -- the
    voxels of `parc` in (lo, hi) in C order -- with `allnames` its class names.
    Passing it in rather than reading volumes is the point: computed inside the
    pipeline while the model's output is still in memory, this costs seconds,
    and ~190 MB holds every ribbon voxel's posterior.
    """
    from scipy import spatial
    gm = (parc > lo) & (parc < hi)
    coords = np.array(np.nonzero(gm)).T
    if len(coords) != len(post):
        raise ValueError('post has %d rows for %d ribbon voxels'
                         % (len(post), len(coords)))
    cidx = [allnames.index(n) for n in names]
    Pc = post[:, cidx]
    voter = np.isin(post.argmax(1), np.asarray(cidx))
    tree = spatial.cKDTree(totkr(coords.astype(float)))
    V = np.asarray(white_tkr, float)
    dn, _jn = tree.query(V)                       # nearest: the NULL test
    d, j = tree.query(V, k=k)
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    if gated:
        # a voxel votes only if its own argmax is a parcel of this hemisphere.
        # NOTE this is close to inert when the ribbon is taken from the
        # parcellation itself, since such a voxel is cortical by construction;
        # it earns its keep only if the ribbon comes from the tissue labels.
        w = w * voter[j]
    num = (Pc[j] * w[:, :, None]).sum(1)
    den = w.sum(1)
    prob = num / np.maximum(den, 1e-12)[:, None]
    prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
    isnull = (dn > null_d) | (den <= 0)
    prob = np.concatenate([prob, isnull.astype(prob.dtype)[:, None]], axis=1)
    prob[isnull, :-1] = 0.0
    prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
    return (-np.log(prob + eps)).astype(np.float32), prob.astype(np.float32)


def _graph_colouring(A):
    """Greedy colouring; vertices of one colour share no edge, so they can be
    resampled simultaneously and exactly."""
    deg = np.asarray(A.sum(1)).ravel()
    colour = np.full(A.shape[0], -1, np.int32)
    rows = A.tolil().rows
    for v in np.argsort(-deg):
        used = {colour[j] for j in rows[v] if colour[j] >= 0}
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    return [np.where(colour == c)[0] for c in range(colour.max() + 1)]


def banded_anneal(unary, edges, w, G, Wm, beta=1.0, init=None, band=3,
                  sweeps=120, t0=1.5, t1=0.02, seed=0):
    """Annealing restricted to the borders and the pockets. Use this, not icm.

    ICM IS A POOR OPTIMISER OF THIS ENERGY, measured: annealing from its own
    solution finds 3.92% lower energy on 10 of 10 hemispheres (range 3.50-4.48%),
    and the gap it leaves is LARGER than the gap it closes from the unary. The
    reason is structural -- shifting a border segment needs a coordinated move
    whose intermediate states are uphill, and ICM only takes moves that are
    downhill now. At beta 8 the pairwise term opposes a single-vertex flip at a
    border by ~32 units against a unary margin of ~0.4, so essentially every
    single-vertex move is blocked.

    What the lower energy buys downstream is modest and should not be oversold:
    over those 10 hemispheres the median agreement with FreeSurfer moved -0.0005
    (better in 4/10) and stray components 3 -> 2 (better in 6/10, worse in 1).

    BANDED, because the interior cannot move: of the vertices a full anneal
    relabelled, 100% lay within 3 hops of a border (median 0, p99 1, max 2) and
    12% outside their parcel's main component. Restricting to that set touches
    28% of the mesh, runs 3.3x faster, and reached a LOWER energy than annealing
    everything (-7676 vs -7431 on the hemisphere tested).

    Init defaults to the unary argmax: ICM first is not needed -- it reaches the
    same energy to within 0.13% and costs 30% of the runtime.

    Temperature is scaled by the arm's own median cost gap, so a change of beta
    does not silently change how hot the anneal is.
    """
    unary = np.asarray(unary, np.float64)
    n, L = unary.shape
    u = 1.0 - np.asarray(w, float)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    U = sp.csr_matrix((np.concatenate([u, u]), (rows, cols)), shape=(n, n))
    A = sp.csr_matrix((np.ones(2 * len(edges)), (rows, cols)), shape=(n, n))
    ncount = np.asarray(A.sum(1)).ravel()
    G = np.asarray(G, np.float64)
    lab = unary.argmin(1) if init is None else np.asarray(init).copy()

    def cost(state, sel):
        onehot = sp.csr_matrix((np.ones(n), (np.arange(n), state)), shape=(n, L))
        C = np.asarray((A[sel] @ onehot).todense())
        Sm = np.asarray((U[sel] @ onehot).todense())
        return unary[sel] + beta * ((ncount[sel][:, None] - C) - (Sm @ G.T))

    from scipy.sparse.csgraph import dijkstra, connected_components
    bnd = np.unique(edges[lab[edges[:, 0]] != lab[edges[:, 1]]].ravel())
    hop = dijkstra(A, directed=False, indices=bnd, unweighted=True, min_only=True)
    in_main = np.ones(n, bool)
    for p in np.unique(lab):
        m = lab == p
        if m.sum() < 2:
            continue
        k, cc = connected_components(Wm[m][:, m], directed=False)
        if k > 1:
            idx = np.where(m)[0]
            in_main[idx[cc != np.bincount(cc).argmax()]] = False
    active = (hop <= band) | (~in_main)
    classes = [c[active[c]] for c in _graph_colouring(A)]
    classes = [c for c in classes if len(c)]

    gaps = np.sort(cost(lab, np.arange(n)), axis=1)
    scale = max(float(np.median(gaps[:, 1] - gaps[:, 0])), 1e-6)
    rng = np.random.default_rng(seed)
    for s in range(sweeps):
        T = scale * t0 * (t1 / t0) ** (s / max(sweeps - 1, 1))
        for c in classes:
            q = cost(lab, c)
            q -= q.min(1, keepdims=True)
            p = np.exp(-q / T)
            p /= p.sum(1, keepdims=True)
            lab[c] = (np.cumsum(p, 1) < rng.random(len(c))[:, None]).sum(1).clip(0, L - 1)
    for _ in range(15):                            # quench to a local minimum
        changed = 0
        for c in classes:
            new = cost(lab, c).argmin(1)
            changed += int((new != lab[c]).sum())
            lab[c] = new
        if changed == 0:
            break
    return lab, int(active.sum())


def n_stray(ids, Wm, labels):
    """Connected components beyond the first, summed over parcels."""
    from scipy.sparse.csgraph import connected_components
    tot = 0
    for pid in np.unique(ids):
        if pid <= 1000:
            continue
        m = ids == pid
        if m.sum() < 2:
            continue
        n, _cc = connected_components(Wm[m][:, m], directed=False)
        tot += n - 1
    return tot


def energy(unary, A, deg_w, lab, beta):
    n = len(lab)
    u = unary[np.arange(n), lab].sum()
    onehot = sp.csr_matrix((np.ones(n), (np.arange(n), lab)),
                           shape=unary.shape)
    agree = np.asarray((A @ onehot)[np.arange(n), lab]).ravel()
    return float(u + beta * 0.5 * (deg_w - agree).sum())


def icm(unary, edges, w, beta=1.0, n_iter=30, init=None, verbose=True):
    """Iterated conditional modes, all vertices updated per sweep.

    The smoothness cost of giving vertex i label l is beta * (W_i - S_i(l)),
    with W_i the total incident edge weight and S_i(l) the weight of the
    neighbours already carrying l -- one sparse product per sweep rather than
    a loop over vertices.

    Synchronous updates can oscillate between two configurations, so the
    energy is recomputed each sweep and the best configuration seen is what
    is returned, never merely the last one.
    """
    n, L = unary.shape
    A = sp.csr_matrix((np.concatenate([w, w]),
                       (np.concatenate([edges[:, 0], edges[:, 1]]),
                        np.concatenate([edges[:, 1], edges[:, 0]]))),
                      shape=(n, n))
    deg_w = np.asarray(A.sum(1)).ravel()
    lab = unary.argmin(1) if init is None else np.asarray(init).copy()
    best, best_e = lab.copy(), energy(unary, A, deg_w, lab, beta)
    if verbose:
        print('    sweep  0: energy %.1f' % best_e)
    for it in range(1, n_iter + 1):
        onehot = sp.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, L))
        agree = np.asarray((A @ onehot).todense()) if sp.issparse(A @ onehot) \
            else (A @ onehot)
        cost = unary + beta * (deg_w[:, None] - agree)
        new = np.asarray(cost).argmin(1)
        changed = int((new != lab).sum())
        lab = new
        e = energy(unary, A, deg_w, lab, beta)
        if e < best_e:
            best_e, best = e, lab.copy()
        if verbose and (it <= 3 or it % 10 == 0 or changed == 0):
            print('    sweep %2d: energy %.1f, %d vertices changed' % (it, e, changed))
        if changed == 0:
            break
    return best, best_e


# ---------------------------------------------------------------------------
# 3. evaluation against an independent parcellation
# ---------------------------------------------------------------------------

def fs_vertex_map(case, fs_dir, hemi, our_verts):
    """For each of our white vertices, the nearest FreeSurfer white vertex.

    Routed through scanner RAS exactly as compare_surfaces documents; the same
    conversion was checked by our white matching FreeSurfer's to a median
    0.42 mm on this subject. Anything of FreeSurfer's that lives per-vertex --
    the annotation, ?h.sulc, ?h.curv -- rides this one correspondence, so they
    cannot disagree about which vertex is which.
    """
    from scipy import spatial
    from dldirect.compare_surfaces import tkr_to_world
    fsio = nib.freesurfer.io
    fsv, _ = fsio.read_geometry(os.path.join(fs_dir, 'surf', '%s.white' % hemi))
    ref = nib.load(os.path.join(case, 'mri', 'aparc.atlas+aseg.nii.gz'))
    fsref = nib.load(os.path.join(fs_dir, 'mri', 'orig.mgz'))
    M = np.linalg.inv(tkr_to_world(
        ref, np.loadtxt(os.path.join(case, 'mri', 'conform_vox2ras.txt')))) \
        @ tkr_to_world(fsref)
    fsv = nib.affines.apply_affine(M, np.asarray(fsv, float))
    return spatial.cKDTree(fsv).query(np.asarray(our_verts, float))[1]


def fs_morph_on_our_mesh(case, fs_dir, hemi, our_verts, what='sulc', j=None):
    """FreeSurfer's own ?h.sulc (or ?h.curv) sampled on our vertices.

    This is the INDEPENDENT fundus signal. hull_depth's z-score is what the
    CRF's edge weights are built from, so a CRF driven hard enough will raise
    the boundary/interior gap measured in z by construction; the same gap
    measured in FreeSurfer's sulc is not something the energy can pay for.
    """
    fsio = nib.freesurfer.io
    v = fsio.read_morph_data(os.path.join(fs_dir, 'surf', '%s.%s' % (hemi, what)))
    if j is None:
        j = fs_vertex_map(case, fs_dir, hemi, our_verts)
    return np.asarray(v, float)[j]


def freesurfer_labels_on_our_mesh(case, fs_dir, hemi, our_verts, labels, names,
                                  j=None):
    """FreeSurfer's ?h.aparc.annot resampled onto our vertices."""
    fsio = nib.freesurfer.io
    lab, _ctab, fsnames = fsio.read_annot(
        os.path.join(fs_dir, 'label', '%s.aparc.annot' % hemi))
    fsnames = [n.decode() if isinstance(n, bytes) else n for n in fsnames]
    if j is None:
        j = fs_vertex_map(case, fs_dir, hemi, our_verts)
    # FreeSurfer annot index -> our integer parcel id, matched by NAME
    short = {n.replace('%s-' % hemi, ''): int(l) for n, l in zip(names, labels)}
    lut = np.full(len(fsnames), -1, np.int64)
    for k, n in enumerate(fsnames):
        if n in short:
            lut[k] = short[n]
    out = lut[np.clip(lab[j], 0, len(fsnames) - 1)]
    return out


def boundary_sulcality(ids, edges, z, keep):
    """How well a labelling's own borders sit in sulcal fundi.

    Dice cannot see this: moving a border a few vertices along a bank costs
    almost no overlap, but it is exactly what the Potts weighting is for.
    Returns (median z on boundary edges, median z elsewhere, the ratio, and
    the boundary edge count).
    """
    a, b = ids[edges[:, 0]], ids[edges[:, 1]]
    ok = keep[edges[:, 0]] & keep[edges[:, 1]]
    ze = np.maximum(z[edges[:, 0]], z[edges[:, 1]])
    on = ok & (a != b)
    off = ok & (a == b)
    if not on.any():
        return np.nan, np.nan, np.nan, 0
    mb, mi = float(np.median(ze[on])), float(np.median(ze[off]))
    return mb, mi, mb - mi, int(on.sum())


def dice_per_label(a, b, labels):
    out = {}
    for l in labels:
        A, B = (a == l), (b == l)
        d = A.sum() + B.sum()
        if d:
            out[int(l)] = 2.0 * (A & B).sum() / d
    return out


# ---------------------------------------------------------------------------

def write_annot_like_fs(path, our_ids, hemi, fs_dir, names, labels):
    """Write our integer parcel ids as a FreeSurfer .annot.

    The colour table and name order are taken from FreeSurfer's own
    ?h.aparc.annot so the parcels render in the standard DK colours and can
    be compared against it by eye without a mental colour map. Vertices whose
    label has no FreeSurfer counterpart fall in index 0 (unknown).
    """
    fsio = nib.freesurfer.io
    _lab, ctab, fsnames = fsio.read_annot(
        os.path.join(fs_dir, 'label', '%s.aparc.annot' % hemi))
    fsnames_s = [n.decode() if isinstance(n, bytes) else n for n in fsnames]
    short = {int(l): n.replace('%s-' % hemi, '') for n, l in zip(names, labels)}
    idx_of = {n: k for k, n in enumerate(fsnames_s)}
    out = np.zeros(len(our_ids), np.int32)
    for k, v in enumerate(our_ids):
        nm = short.get(int(v))
        if nm is not None and nm in idx_of:
            out[k] = idx_of[nm]
    fsio.write_annot(path, out, ctab, fsnames, fill_ctab=False)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHAT THIS')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case')
    ap.add_argument('--run', default='field_pial_sigma0.65')
    ap.add_argument('--fs-dir', required=True, help='recon-all dir, for scoring')
    ap.add_argument('--hemi', nargs='+', default=['lh'])
    ap.add_argument('--beta', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument('--theta', type=float, default=0.5)
    ap.add_argument('--uniform-g', action='store_true',
                    help='control: pairwise path with G = 1 everywhere')
    ap.add_argument('--per-border', action='store_true',
                    help='make the pairwise term depend on WHICH two parcels '
                         'meet, so only sulcally-defined borders loosen at a '
                         'fundus. See border_sulcality_prior.')
    ap.add_argument('--g-scale', type=float, default=0.6,
                    help='border z at which a pair counts as fully sulcal')
    ap.add_argument('--floor', type=float, default=0.05,
                    help='minimum edge weight; lower lets a fundus switch the '
                         'smoothness term off almost entirely')
    ap.add_argument('--radius', type=float, default=5.0, help='hull ball radius')
    ap.add_argument('--zsigma-iters', type=int, default=50)
    ap.add_argument('--flat', action='store_true',
                    help='ablation: ignore sulci, all edge weights = 1')
    ap.add_argument('--out', default=None)
    ap.add_argument('--column', action='store_true',
                    help='sample the logits along the white->pial column over '
                         'all 94 non-meta classes (see column_unary)')
    ap.add_argument('--plain-mean', action='store_true',
                    help='average the whole column instead of letting only '
                         'non-null samples vote (the pre-gate behaviour)')
    ap.add_argument('--no-null', action='store_true',
                    help='--column: force every vertex into a cortical parcel')
    ap.add_argument('--temperature', type=float, default=1.0,
                    help='divide the logits by this before the softmax; '
                         '<1 sharpens, >1 flattens')
    ap.add_argument('--topk', type=int, default=None,
                    help='keep only the top-k labels per vertex and renormalise')
    ap.add_argument('--logit-dir', default=None,
                    help="directory of per-parcel seg_<name>.nii.gz logits from "
                         'a model run with SAVE_LOGITS_FILTER = None; uses the '
                         "model's own softmax as the unary instead of voxel votes")
    ap.add_argument('--write-beta', type=float, default=1.0,
                    help='which beta to write annotations for')
    args = ap.parse_args(argv)

    from dldirect.hull_depth import (rasterize, hull_depth_field, sample,
                                     geodesic_zscore, crf_edge_weights)
    from dldirect.field_pial_prototype import make_transforms, _mesh_adjacency
    from dldirect import regional_stats as rs
    fsio = nib.freesurfer.io

    ref = nib.load(os.path.join(args.case, 'mri', 'aparc.atlas+aseg.nii.gz'))
    parc = np.asarray(ref.dataobj).astype(np.int32)
    spacing = tuple(float(z) for z in ref.header.get_zooms()[:3])
    shape = tuple(ref.shape[:3])
    tovox, _ = make_transforms(ref)
    lut, valid_names, _c = rs.get_labels()
    valid = set(valid_names)

    for h in args.hemi:
        # get_labels already drops unknown / corpuscallosum / medial wall --
        # exactly the labels the model writes no logit for
        names = [k for k in valid_names
                 if k.startswith('%s-' % h) and lut[k] > 1000]
        labels = np.array([lut[k] for k in names])
        w, wf = fsio.read_geometry(os.path.join(args.case, args.run, '%s.white' % h))
        p, pf = fsio.read_geometry(os.path.join(args.case, args.run, '%s.pial' % h))
        wv = tovox(np.asarray(w, float))

        if args.logit_dir and args.column:
            pial, _pf = fsio.read_geometry(os.path.join(args.case, args.run,
                                                        '%s.pial' % h))
            unary, prob, _wt = column_unary(w, pial, args.logit_dir, names, tovox,
                                            include_null=not args.no_null,
                                            vote_gate=not args.plain_mean)
            if not args.no_null:
                # null is a label like any other: the smoothness term can
                # absorb a small null island into its surroundings, while a
                # coherent region such as the medial wall holds together
                labels = np.concatenate([labels, [0]])
                names = names + ['%s-NULL' % h]
            _v, votes = soft_unary(wv, parc, labels[labels > 1000])
            cort = votes.sum(1) > 0
            conf = prob.max(1)
            print('   unary: COLUMN (94 classes, meta dropped%s%s) | max-prob '
                  'median %.3f | null %d (%.1f%%)'
                  % ('' if args.no_null else ', +null',
                     '' if args.plain_mean else ', gated vote', np.median(conf),
                     int((labels[prob.argmax(1)] == 0).sum()),
                     100.0 * (labels[prob.argmax(1)] == 0).mean()))
        elif args.logit_dir:
            unary, prob = model_unary(wv, args.logit_dir, names,
                                      temperature=args.temperature,
                                      topk=args.topk)
            _v, votes = soft_unary(wv, parc, labels[labels > 1000])
            cort = votes.sum(1) > 0
            conf = prob.max(1)
            print('   unary: MODEL logits | max-prob median %.3f, '
                  '%.1f%% of vertices below 0.9 (ambiguous)'
                  % (np.median(conf), 100.0 * (conf < 0.9).mean()))
        else:
            unary, votes = soft_unary(wv, parc, labels)
            cort = votes.sum(1) > 0
            conf = (votes / np.maximum(votes.sum(1, keepdims=True), 1e-9)).max(1)
            print('   unary: voxel VOTES | max-prob median %.3f, '
                  '%.1f%% of vertices below 0.9 (ambiguous)'
                  % (np.median(conf), 100.0 * (conf < 0.9).mean()))
        print('%s: %d vertices, %d cortical, %d labels' % (h, len(w), cort.sum(), len(labels)))

        _m, Wm, _d = _mesh_adjacency(np.asarray(w, float), np.asarray(wf))
        if args.flat:
            z = np.zeros(len(w), np.float32)
        else:
            mask = rasterize(tovox(np.asarray(p, float)), pf, shape)
            dep, _hull = hull_depth_field(mask, args.radius, spacing)
            d = sample(dep, wv).astype(np.float64)
            z, _m1, _sd = geodesic_zscore(d, w, wf, args.zsigma_iters,
                                          adjacency=(Wm, _d))
        edges, ew = crf_edge_weights(z, w, wf, theta=args.theta,
                                     floor=args.floor)
        print('   edge weights: median %.3f, %.1f%% below 0.5%s'
              % (np.median(ew), 100.0 * (ew < 0.5).mean(),
                 '  (FLAT ablation)' if args.flat else ''))

        ref_lab = freesurfer_labels_on_our_mesh(args.case, args.fs_dir, h,
                                                w, labels, names)
        base = labels[unary.argmin(1)]
        keep = cort & (ref_lab > 0)
        d0 = dice_per_label(base[keep], ref_lab[keep], labels)
        print('   BEFORE (argmin unary): mean Dice vs FreeSurfer %.4f over %d parcels'
              % (np.mean(list(d0.values())), len(d0)))
        G = None
        if args.uniform_g:
            # CONTROL: the pairwise path with every pair fully sulcal, which is
            # algebraically the weighted Potts. Run the uniform arm through
            # THIS code rather than icm(), so a per-border comparison differs
            # only in G and not in the inference as well.
            G = np.ones((len(labels), len(labels)), np.float32)
            np.fill_diagonal(G, 0.0)
            print('   per-border prior: DISABLED (G = 1, control)')
        elif args.per_border:
            G = border_sulcality_prior(base, edges, np.asarray(z, float), labels,
                                       scale=args.g_scale)
            off = G[np.triu_indices_from(G, 1)]
            print('   per-border prior: %d pairs, g median %.2f, %.0f%% of '
                  'pairs fully sulcal (g=1), %.0f%% below 0.5'
                  % (len(off), np.median(off), 100 * (off >= 0.999).mean(),
                     100 * (off < 0.5).mean()))
        for beta in args.beta:
            if G is None:
                lab_i, e = icm(unary, edges, ew, beta=beta, verbose=False)
            else:
                lab_i, e = icm_pairwise(unary, edges, ew, G, beta=beta)
            out = labels[lab_i]
            dd = dice_per_label(out[keep], ref_lab[keep], labels)
            ch = 100.0 * (out[keep] != base[keep]).mean()
            mb, mi, gap, nb = boundary_sulcality(out, edges, np.asarray(z, float), keep)
            frg = n_stray(out, Wm, labels)
            print('   beta %-4.1f -> Dice %.4f (%+.4f) | border z %+.2f vs '
                  'interior %+.2f (gap %+.2f) | %d stray comps | %.1f%% moved'
                  % (beta, np.mean(list(dd.values())),
                     np.mean(list(dd.values())) - np.mean(list(d0.values())),
                     mb, mi, gap, frg, ch))
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                tag = ('%g' % beta).replace('.', 'p')
                write_annot_like_fs(os.path.join(args.out, '%s.crf_b%s.annot' % (h, tag)),
                                    out, h, args.fs_dir, names, labels)
                ch = (out != base).astype(np.float32)
                fsio.write_morph_data(os.path.join(args.out, '%s.changed_b%s' % (h, tag)),
                                      ch, fnum=len(wf))
                # where each labelling disagrees with the reference, so a
                # change can be read as a fix or a break rather than just a move
                err = (out != ref_lab).astype(np.float32); err[~keep] = 0
                fsio.write_morph_data(os.path.join(args.out, '%s.err_b%s' % (h, tag)),
                                      err, fnum=len(wf))
                moved = ch.astype(bool) & keep
                if moved.any():
                    fixed = int(((base != ref_lab) & (out == ref_lab))[moved].sum())
                    broke = int(((base == ref_lab) & (out != ref_lab))[moved].sum())
                    print('     beta %g: %d moved -> %d fixed, %d broken, %d neutral'
                          % (beta, moved.sum(), fixed, broke,
                             moved.sum() - fixed - broke))
                if beta == args.beta[0]:
                    write_annot_like_fs(os.path.join(args.out, '%s.base.annot' % h),
                                        base, h, args.fs_dir, names, labels)
                    write_annot_like_fs(os.path.join(args.out, '%s.fsref.annot' % h),
                                        ref_lab, h, args.fs_dir, names, labels)
                    e0 = (base != ref_lab).astype(np.float32); e0[~keep] = 0
                    fsio.write_morph_data(os.path.join(args.out, '%s.err_base' % h),
                                          e0, fnum=len(wf))
    return 0


if __name__ == '__main__':
    sys.exit(main())
