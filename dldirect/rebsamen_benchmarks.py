"""The two OASIS-3 benchmarks from Rebsamen et al. (2020), HBM 41(17):4804.

Reproduces, for any number of pipelines side by side:

  REPRODUCIBILITY (the paper's Table 2 / Figure 5). Re-scans acquired in the
  SAME session, scored by the average absolute change relative to the
  within-session mean,

      eps_mu = 100/N * sum_i [ 1/n_i * sum_t |m_it - mu_i| / mu_i ]

  with N sessions, n_i re-scans in session i, m_it the measurement and mu_i
  the within-session mean. Reported for global mean thickness and as the
  average over the 68 Desikan-Killiany ROIs.

  ATROPHY BY CDR (Table 3 / Figure 6). Subjects with more than one scan at
  least one year apart who did not change CDR group over that interval.
  The annual atrophy rate is the within-subject least-squares slope of
  thickness on age, in mm/year, grouped as CDR 0 (healthy), CDR 0.5
  (questionable) and CDR >= 1 (dementia), and compared between pipelines by
  a paired t-test at alpha = 0.05.

WHAT IS NOT THE PAPER'S

The paper ran on its own cohort and its own pipeline versions; this runs on
whatever trees are passed to it. Sample sizes therefore will not match
exactly -- the paper had 761 re-scan sessions and 368/31/7 subjects in the
three CDR groups. Treat the numbers as this repository's benchmark on this
data, not as a replication of the published figures.

TWO CHOICES THE PAPER LEAVES OPEN, made explicit here

* A session with more than one re-scan contributes ONE value per session to
  the atrophy analysis (the within-session mean), so a session scanned twice
  does not get twice the weight in a subject's slope.
* CDR is parsed out of OASIS-3's free-text DIAGNOSIS_SUBTYPE ("CDR-0.5").
  Non-AD dementias carry a CDR too and are kept, since the grouping in the
  paper is by CDR and not by aetiology. --ad-only drops them.

Usage
-----
    python rebsamen_benchmarks.py --metadata <metadata.csv> --out <dir> \\
        --tree  "label,<prep_root>,<out_name>,<metric>"  [--tree ...] \\
        --table "FreeSurfer,<collected.csv>"             [--table ...]
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np

N_ROI = 68            # Desikan-Killiany, both hemispheres
GLOBAL_COLS = ('lh-MeanThickness', 'rh-MeanThickness')


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def read_row_csv(path):
    """(columns, values) from a one-data-row result-thick CSV."""
    with open(path) as fh:
        rd = list(csv.reader(fh))
    cols = rd[0][1:]
    vals = np.array([float(x) if x not in ('', 'nan') else np.nan
                     for x in rd[1][1:]])
    return cols, vals


def load_tree(root, out_name, metric, sessions):
    """{session: values} from per-subject result-thick-<metric>.csv files."""
    out, cols = {}, None
    for s in sessions:
        p = os.path.join(root, s, out_name, 'result-thick-%s.csv' % metric)
        if not os.path.exists(p):
            continue
        c, v = read_row_csv(p)
        if cols is None:
            cols = c
        elif c != cols:
            raise ValueError('column order differs at %s' % p)
        out[s] = v
    return cols, out


def load_table(path):
    """{session: values} from one collected table (freesurfer_stats.py)."""
    with open(path) as fh:
        rd = list(csv.reader(fh))
    cols = rd[0][1:]
    out = {r[0]: np.array([float(x) if x not in ('', 'nan') else np.nan
                           for x in r[1:]]) for r in rd[1:]}
    return cols, out


_CDR = re.compile(r'CDR-([0-9.]+)')


def load_metadata(path):
    """{session: (participant, age_years, cdr_float, subtype)}.

    OASIS-3's metadata is semicolon-separated and puts CDR inside the
    free-text DIAGNOSIS_SUBTYPE rather than in a column of its own.
    """
    out = {}
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter=';')
        for r in rd:
            m = _CDR.search(r.get('DIAGNOSIS_SUBTYPE') or '')
            try:
                age = float(r.get('AGE') or 'nan')
            except ValueError:
                age = np.nan
            out[r['SUBJECT_ID']] = (r.get('SOURCE_SUBJECT'), age,
                                    float(m.group(1)) if m else np.nan,
                                    (r.get('DIAGNOSIS') or '').strip(),
                                    1.0 if (r.get('SEX') or '').strip().upper().startswith('F')
                                    else (0.0 if (r.get('SEX') or '').strip() else np.nan))
    return out


def session_of(scan_id):
    """'OAS30001_ses-d0129_run-02' -> 'OAS30001_ses-d0129'."""
    return re.sub(r'_run-\d+$', '', scan_id)


# ---------------------------------------------------------------------------
# 1. reproducibility
# ---------------------------------------------------------------------------

def reproducibility(values, cols):
    """eps_mu per column, over sessions that hold two or more re-scans.

    Returns (eps[n_col], n_sessions, n_scans). A session contributes the mean
    over its own re-scans, so a session scanned three times does not outweigh
    one scanned twice -- that is the 1/n_i in the paper's equation.
    """
    by_sess = defaultdict(list)
    for scan, v in values.items():
        by_sess[session_of(scan)].append(v)
    reps = {s: np.vstack(v) for s, v in by_sess.items() if len(v) >= 2}
    if not reps:
        return None, 0, 0
    per_session = []
    for s, M in reps.items():
        mu = np.nanmean(M, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            rel = np.abs(M - mu[None]) / mu[None]
        rel[~np.isfinite(rel)] = np.nan
        per_session.append(np.nanmean(rel, axis=0))      # the 1/n_i sum
    eps = 100.0 * np.nanmean(np.vstack(per_session), axis=0)
    return eps, len(reps), sum(M.shape[0] for M in reps.values())


# ---------------------------------------------------------------------------
# 2. atrophy by CDR
# ---------------------------------------------------------------------------

def cdr_group(c):
    if not np.isfinite(c):
        return None
    if c == 0:
        return 'CDR 0 (healthy)'
    if c == 0.5:
        return 'CDR 0.5 (questionable)'
    return 'CDR >= 1 (dementia)'


def atrophy_rates(values, cols, meta, min_years=1.0, ad_only=False):
    """{group: (slopes[n_subj, n_col], subject_ids)} in mm/year.

    One value per SESSION (re-scans averaged first), subjects needing at
    least two sessions spanning `min_years`, and an unchanged CDR group
    across the sessions used -- the paper's "did not change sub-cohort in
    that interval".
    """
    # session -> mean over its re-scans, plus the session's age and CDR
    per_sess = defaultdict(list)
    for scan, v in values.items():
        per_sess[session_of(scan)].append((scan, v))

    by_subj = defaultdict(list)
    for sess, items in per_sess.items():
        rows = [v for _, v in items]
        ages, cdrs, subj, diags = [], [], None, []
        for scan, _ in items:
            if scan not in meta:
                continue
            p, age, cdr, diag = meta[scan][:4]
            subj = subj or p
            ages.append(age)
            cdrs.append(cdr)
            diags.append(diag)
        if subj is None or not ages:
            continue
        cdrs = [c for c in cdrs if np.isfinite(c)]
        if not cdrs or len(set(cdrs)) != 1:
            continue                      # ambiguous within a single session
        if ad_only and not any('AD' in d or 'normal' in d.lower() for d in diags):
            continue
        by_subj[subj].append((float(np.nanmean(ages)), cdrs[0],
                              np.nanmean(np.vstack(rows), axis=0)))

    groups = defaultdict(lambda: ([], []))
    for subj, recs in by_subj.items():
        if len(recs) < 2:
            continue
        ages = np.array([r[0] for r in recs])
        if not np.isfinite(ages).all() or ages.max() - ages.min() < min_years:
            continue
        gset = {cdr_group(r[1]) for r in recs}
        if len(gset) != 1 or None in gset:
            continue                      # changed sub-cohort in the interval
        g = gset.pop()
        M = np.vstack([r[2] for r in recs])
        sl = np.full(M.shape[1], np.nan)
        for j in range(M.shape[1]):
            ok = np.isfinite(M[:, j]) & np.isfinite(ages)
            if ok.sum() >= 2 and ages[ok].max() - ages[ok].min() >= min_years:
                sl[j] = np.polyfit(ages[ok], M[ok, j], 1)[0]
        groups[g][0].append(sl)
        groups[g][1].append(subj)
    return {g: (np.vstack(s), ids) for g, (s, ids) in groups.items() if s}


# ---------------------------------------------------------------------------
# 3. how well a pipeline separates the CDR groups
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    """Standardised difference between two INDEPENDENT groups, pooled SD.

    Pooled rather than either group's own SD, because the groups here differ
    enormously in size (368 vs 31 vs 7) and using the small group's SD would
    make d swing on a handful of subjects.
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def subject_covariates(groups, meta, etiv_root=None):
    """{subject: [baseline_age, is_female, eTIV]} for every subject in `groups`.

    eTIV comes from FreeSurfer's aseg.stats when `etiv_root` is given; it is the
    only one of the three the OASIS metadata does not carry. Subjects whose eTIV
    cannot be read keep a NaN and are dropped from the adjusted comparison only.
    """
    scans = defaultdict(list)
    for scan, rec in meta.items():
        p, age, cdr, diag = rec[:4]
        if p is not None:
            scans[p].append((age, scan))
    out = {}
    for g, (sl, ids) in groups.items():
        for sub in ids:
            if sub in out or sub not in scans:
                continue
            rows = sorted(x for x in scans[sub] if np.isfinite(x[0]))
            if not rows:
                continue
            age0, scan0 = rows[0]
            sex = meta[scan0][4] if len(meta[scan0]) > 4 else np.nan
            tiv = np.nan
            if etiv_root:
                f = os.path.join(etiv_root, scan0, 'stats', 'aseg.stats')
                if os.path.exists(f):
                    for line in open(f):
                        if 'EstimatedTotalIntraCranialVol' in line:
                            try:
                                tiv = float(line.split(',')[3])
                            except (IndexError, ValueError):
                                pass
                            break
            out[sub] = [age0, sex, tiv]
    return out


def adjust_slopes(vec, subs, cov, extra=None, fit=None):
    """Residualise a per-subject slope vector on the available covariates.

    Columns with no usable values are dropped rather than failing, so an
    unavailable eTIV degrades the adjustment instead of disabling it. Returns
    (residuals + mean, list of covariate names actually used); adding the mean
    back keeps the output on the mm/year scale the rest of the report uses.

    `fit` is a boolean mask selecting the entries the coefficients are ESTIMATED
    on; the fit is then applied to every entry. Rebsamen et al. (2020) estimate
    the nuisance model on the healthy controls alone and apply it to all
    samples, which is what group_separation passes. Estimating on the pooled
    sample instead lets a group difference in age or eTIV bend the very
    covariate relationship used to remove it. Pass None to fit on everything.
    """
    names, cols = [], []
    base = np.array([cov.get(s, [np.nan, np.nan, np.nan]) for s in subs], float)
    for j, nm in enumerate(('age', 'sex', 'eTIV')):
        c = base[:, j]
        if np.isfinite(c).sum() > 0.8 * len(c) and np.nanstd(c) > 0:
            names.append(nm)
            cols.append(np.where(np.isfinite(c), c, np.nanmean(c)))
    if extra is not None:
        for nm, c in extra:
            if np.isfinite(c).sum() > 0.8 * len(c) and np.nanstd(c) > 0:
                names.append(nm)
                cols.append(np.where(np.isfinite(c), c, np.nanmean(c)))
    ok = np.isfinite(vec)
    if not names or ok.sum() < 10:
        return vec, []
    D = np.column_stack([np.ones(len(vec))] + cols)
    f = ok if fit is None else (ok & np.asarray(fit, bool))
    if f.sum() < 10:
        return vec, []
    beta, _, _, _ = np.linalg.lstsq(D[f], vec[f], rcond=None)
    out = np.full_like(vec, np.nan)
    out[ok] = vec[ok] - D[ok] @ beta + vec[f].mean()
    return out, names


def group_separation(groups, cols, ref='CDR 0 (healthy)', cov=None):
    """(d, p, n_ref, n_other) per group against the healthy group.

    Welch's t-test, not Student's: the groups are unequal in both size and
    variance, which is exactly the case Student's assumption fails. The sign
    convention is `other - healthy`, so a NEGATIVE d means the group atrophies
    faster than healthy controls.
    """
    from scipy import stats
    gi = global_index(cols)
    out = {}
    if ref not in groups:
        return out
    a = np.nanmean(groups[ref][0][:, gi], axis=1)
    a_ids = list(groups[ref][1])
    for g, (sl, ids) in groups.items():
        if g == ref:
            continue
        b = np.nanmean(sl[:, gi], axis=1)
        if cov:
            # Estimate the covariate model on the HEALTHY CONTROLS ONLY and
            # apply it to both groups, as Rebsamen et al. (2020) do. Fitting on
            # the pooled sample lets the group difference in age and eTIV bend
            # the relationship used to remove it; fitting per group would absorb
            # the group difference outright. Controls-only avoids both.
            pooled = np.concatenate([a, b])
            fit = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
            adj, used = adjust_slopes(pooled, a_ids + list(ids), cov, fit=fit)
            if used:
                a2, b2 = adj[:len(a)], adj[len(a):]
            else:
                a2, b2 = a, b
        else:
            a2, b2 = a, b
        aa, bb = a2[np.isfinite(a2)], b2[np.isfinite(b2)]
        if len(bb) < 2:
            continue
        t, pv = stats.ttest_ind(bb, aa, equal_var=False)
        out[g] = (cohens_d(bb, aa), pv, len(aa), len(bb))
    return out


def global_index(cols):
    return [cols.index(c) for c in GLOBAL_COLS if c in cols]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('Usage')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--subjects', required=True,
                    help='one scan id per line; the cohort for every pipeline')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tree', action='append', default=[], dest='trees',
                    help='LABEL,PREP_ROOT,OUT_NAME,METRIC')
    ap.add_argument('--table', action='append', default=[], dest='tables',
                    help='LABEL,PATH to a collected table')
    ap.add_argument('--min-years', type=float, default=1.0)
    ap.add_argument('--adjust', action='store_true',
                    help='also report the CDR separation adjusted for baseline '
                         'age, sex and (with --etiv-root) eTIV. The published '
                         'benchmark is UNADJUSTED, and in OASIS-3 the impaired '
                         'groups are several years older than the healthy one, '
                         'so the unadjusted d is inflated for every pipeline')
    ap.add_argument('--etiv-root',
                    help='FreeSurfer SUBJECTS_DIR to read '
                         'EstimatedTotalIntraCranialVol from, for --adjust')
    ap.add_argument('--ad-only', action='store_true',
                    help='drop non-AD dementias from the CDR groups')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    sessions = [l.strip() for l in open(args.subjects) if l.strip()]
    meta = load_metadata(args.metadata)
    print('cohort %d scans; metadata covers %d of them'
          % (len(sessions), sum(1 for s in sessions if s in meta)))

    sources = []
    for spec in args.trees:
        lab, root, on, met = spec.split(',')
        cols, vals = load_tree(root, on, met, sessions)
        sources.append((lab, cols, vals))
    for spec in args.tables:
        lab, path = spec.split(',', 1)
        cols, vals = load_table(path)
        sources.append((lab, cols, {k: v for k, v in vals.items()
                                    if k in set(sessions)}))
    if not sources:
        raise SystemExit('give at least one --tree or --table')

    # ---- 1. reproducibility -------------------------------------------------
    print('\n=== Reproducibility: average absolute change from the '
          'within-session mean (%) ===')
    print('%-26s %8s %8s   %s' % ('pipeline', 'global', 'ROI-avg', 'sessions/scans'))
    rep_rows = []
    for lab, cols, vals in sources:
        eps, ns, nsc = reproducibility(vals, cols)
        if eps is None:
            print('%-26s  no re-scan sessions' % lab)
            continue
        gi = global_index(cols)
        g = float(np.nanmean(eps[gi])) if gi else np.nan
        roi = float(np.nanmean(eps[:N_ROI]))
        print('%-26s %8.3f %8.3f   %d / %d' % (lab, g, roi, ns, nsc))
        rep_rows.append([lab, g, roi, ns, nsc] + list(eps[:N_ROI]))
    with open(os.path.join(args.out, 'reproducibility.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pipeline', 'global_pct', 'roi_avg_pct', 'n_sessions',
                    'n_scans'] + list(sources[0][1][:N_ROI]))
        w.writerows(rep_rows)

    # ---- 2. atrophy ---------------------------------------------------------
    print('\n=== Annual cortical thickness change, mm/year, mean (SD) ===')
    order = ['CDR 0 (healthy)', 'CDR 0.5 (questionable)', 'CDR >= 1 (dementia)']
    per_pipe = {}
    for lab, cols, vals in sources:
        per_pipe[lab] = (cols, atrophy_rates(vals, cols, meta, args.min_years,
                                             args.ad_only))
    print('%-26s %s' % ('pipeline', '  '.join('%-24s' % g for g in order)))
    atr_rows = []
    for lab, (cols, groups) in per_pipe.items():
        gi = global_index(cols)
        cells = []
        for g in order:
            if g not in groups:
                cells.append('%-24s' % '-')
                continue
            sl, ids = groups[g]
            v = np.nanmean(sl[:, gi], axis=1)
            cells.append('%-24s' % ('%+.4f (%.4f) n=%d'
                                    % (np.nanmean(v), np.nanstd(v, ddof=1), len(ids))))
            atr_rows.append([lab, g, len(ids), float(np.nanmean(v)),
                             float(np.nanstd(v, ddof=1))]
                            + list(np.nanmean(sl[:, :N_ROI], axis=0)))
        print('%-26s %s' % (lab, '  '.join(cells)))
    with open(os.path.join(args.out, 'atrophy.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pipeline', 'group', 'n_subjects', 'global_mean',
                    'global_sd'] + list(sources[0][1][:N_ROI]))
        w.writerows(atr_rows)

    # ---- separation of the CDR groups within each pipeline ------------------
    print('\n=== Separation from the healthy group (Welch t-test; d = Cohen\'s '
          'd, pooled SD; negative = faster thinning) ===')
    print('%-26s %s' % ('pipeline',
                        '  '.join('%-26s' % g for g in order[1:])))
    sep_rows = []
    for lab, (cols, groups) in per_pipe.items():
        sep = group_separation(groups, cols)
        sep_adj = None
        if args.adjust:
            cov = subject_covariates(groups, meta, args.etiv_root)
            sep_adj = group_separation(groups, cols, cov=cov)
        cells = []
        for g in order[1:]:
            if g not in sep:
                cells.append('%-26s' % '-')
                continue
            d, pv, na, nb = sep[g]
            if sep_adj and g in sep_adj:
                da, pa = sep_adj[g][0], sep_adj[g][1]
                cells.append('%-26s' % ('d=%+.2f (adj %+.2f) p=%.3g%s'
                                        % (d, da, pv, '*' if pv < 0.05 else '')))
                sep_rows.append([lab, g, d, pv, na, nb, da, pa])
            else:
                cells.append('%-26s' % ('d=%+.2f  p=%.3g%s' % (d, pv,
                                        '*' if pv < 0.05 else '')))
                sep_rows.append([lab, g, d, pv, na, nb, '', ''])
        print('%-26s %s' % (lab, '  '.join(cells)))
    with open(os.path.join(args.out, 'cdr_separation.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pipeline', 'group', 'cohens_d', 'p_welch',
                    'n_healthy', 'n_group', 'cohens_d_adjusted', 'p_adjusted'])
        w.writerows(sep_rows)

    # ---- paired comparison between pipelines --------------------------------
    if len(sources) > 1:
        from scipy import stats
        print('\n=== Paired t-test on the per-subject global atrophy rate '
              '(alpha = 0.05) ===')
        labs = [s[0] for s in sources]
        ref = labs[0]
        cols_r, gr_r = per_pipe[ref]
        for lab in labs[1:]:
            cols_o, gr_o = per_pipe[lab]
            for g in order:
                if g not in gr_r or g not in gr_o:
                    continue
                a_sl, a_id = gr_r[g]
                b_sl, b_id = gr_o[g]
                common = [i for i in a_id if i in set(b_id)]
                if len(common) < 3:
                    continue
                ai = {s: k for k, s in enumerate(a_id)}
                bi = {s: k for k, s in enumerate(b_id)}
                a = np.nanmean(a_sl[[ai[s] for s in common]][:, global_index(cols_r)], axis=1)
                b = np.nanmean(b_sl[[bi[s] for s in common]][:, global_index(cols_o)], axis=1)
                ok = np.isfinite(a) & np.isfinite(b)
                t, p = stats.ttest_rel(a[ok], b[ok])
                print('%-22s vs %-22s %-24s n=%3d  diff %+.4f  t=%+.2f  p=%.3g%s'
                      % (ref, lab, g, ok.sum(), np.mean(a[ok] - b[ok]), t, p,
                         '  *' if p < 0.05 else ''))
    print('\nwrote reproducibility.csv and atrophy.csv to %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
