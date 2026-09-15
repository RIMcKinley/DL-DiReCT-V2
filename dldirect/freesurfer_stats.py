#!/usr/bin/env python
"""Read a FreeSurfer run's ?h.aparc.stats into DL+DiReCT's result-thick columns.

WHAT IT TAKES FROM THE STATS FILE

`ThickAvg` (field 4) and `ThickStd` (field 5) per Desikan-Killiany parcel, and
the hemisphere mean from the `# Measure Cortex, MeanThickness` header line.

That header figure is FreeSurfer's own, and it is a PER-VERTEX mean, not an
area-weighted one. Checked on OAS30001 by reconstructing it from the parcel
rows: NumVert-weighting gives 2.43917 against the header's 2.43916 (lh) and
2.49533 against 2.49537 (rh) -- agreement to 1e-5, i.e. to the printed
precision -- while SurfArea-weighting is off by 0.005 and an unweighted mean of
the parcels by 0.06. The same identity implies each ThickAvg is itself an
unweighted mean over its parcel's vertices, which is what makes these columns
comparable with regional_stats' output.

`lh/rh-MeanThickness` has no ThickStd counterpart, so those two cells are NaN in
the std table.

ONE MISMATCH WORTH KNOWING: the header's NumVert (106103 on that subject) is a
handful short of the sum over parcel rows (106108), so FreeSurfer's cortex label
and its parcel partition are not quite the same vertex set. Immaterial at this
scale, but it is why a hemisphere column can differ slightly from a re-average
of the parcels.
"""

import argparse
import csv
import os
import sys

import numpy as np

DEFAULT_FS_ROOT = '/str/nas/MORPHOMETRY/OASIS3/FS_RESULTS'


def parse_aparc_stats(path):
    """(thick, std, hemisphere_mean) from one ?h.aparc.stats."""
    thick, std, mean = {}, {}, None
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('#'):
                if 'Measure Cortex, MeanThickness' in ln:
                    mean = float(ln.split(',')[3])
                continue
            f = ln.split()
            if len(f) >= 6:
                thick[f[0]] = float(f[4])
                std[f[0]] = float(f[5])
    return thick, std, mean


def collect(subjects, columns, fs_root=DEFAULT_FS_ROOT, hemis=('lh', 'rh')):
    """(thick_rows, std_rows, missing) in `columns` order, one row per subject."""
    rows_t, rows_s, missing = [], [], set()
    for s in subjects:
        vt, vs = {}, {}
        for h in hemis:
            t, sd, m = parse_aparc_stats(
                os.path.join(fs_root, s, 'stats', '%s.aparc.stats' % h))
            for k, v in t.items():
                vt['%s-%s' % (h, k)] = v
                vs['%s-%s' % (h, k)] = sd[k]
            vt['%s-MeanThickness' % h] = m
            vs['%s-MeanThickness' % h] = np.nan
        rows_t.append([s] + [vt.get(c, np.nan) for c in columns])
        rows_s.append([s] + [vs.get(c, np.nan) for c in columns])
        missing |= {c for c in columns if c not in vt}
    return rows_t, rows_s, sorted(missing)


def read_csv_row(path):
    """(column names, values) from a one-row result-thick style CSV."""
    with open(path) as fh:
        r = list(csv.reader(fh))
    return r[0][1:], np.array([float(x) if x not in ('', 'nan') else np.nan
                               for x in r[1][1:]])


def load_comparison(subjects, metric, fs_csv, prep_root, out_name='field_pial_sigma0.65'):
    """(columns, ours[n_subj, n_col], fs[n_subj, n_col]) for one metric.

    Reads each subject's result-thick-<metric>.csv written by regional_stats and
    lines it up against the collected FreeSurfer table. The column orders are
    asserted equal rather than assumed -- they come from different code paths.
    """
    ours, cols = [], None
    for s in subjects:
        c, v = read_csv_row(os.path.join(prep_root, s, out_name,
                                         'result-thick-%s.csv' % metric))
        cols = cols or c
        if c != cols:
            raise ValueError('column order differs at %s' % s)
        ours.append(v)
    with open(fs_csv) as fh:
        rd = list(csv.reader(fh))
    fscols = rd[0][1:]
    if fscols != cols:
        raise ValueError('FreeSurfer table columns do not match %s' % metric)
    by = {r[0]: np.array([float(x) if x not in ('', 'nan') else np.nan
                          for x in r[1:]]) for r in rd[1:]}
    return cols, np.array(ours), np.array([by[s] for s in subjects])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subjects', required=True,
                   help='file with one FreeSurfer subject directory name per line')
    p.add_argument('--columns-from', required=True,
                   help='a result-thick-*.csv whose column order to follow')
    p.add_argument('--fs-root', default=DEFAULT_FS_ROOT)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--prefix', default='fs')
    args = p.parse_args()

    subs = [l.strip() for l in open(args.subjects) if l.strip()]
    cols, _ = read_csv_row(args.columns_from)
    rows_t, rows_s, missing = collect(subs, cols, args.fs_root)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (('%s_thick.csv' % args.prefix, rows_t),
                       ('%s_thickstd.csv' % args.prefix, rows_s)):
        with open(os.path.join(args.out_dir, name), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['SUBJECT'] + cols)
            w.writerows(rows)
    print('%d subjects, %d columns; unmatched columns: %s'
          % (len(rows_t), len(cols), missing or 'none'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
