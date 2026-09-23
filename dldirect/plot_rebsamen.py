"""Figures 5 and 6 of Rebsamen et al. (2020), from rebsamen_benchmarks.py output.

  fig5   per-ROI reproducibility error (%), one row per pipeline
  fig6   per-ROI annual thickness change (mm/year), one row per CDR group

Both read the CSVs that rebsamen_benchmarks.py writes and paint the 68
Desikan-Killiany columns onto an fsaverage inflated surface, four views.
Rows within a figure share one colour scale so they can be read against each
other; a per-row scale would make every row look alike.

The painting follows plot_thickness.brain -- same surfaces, same annot, same
projection and lighting -- so these figures sit beside the agreement figures
without a change of visual language.
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel.freesurfer.io as fsio

SURF, INK, MUTED, NA = '#faf9f5', '#1a1a18', '#6b6a63', '#d8d7d0'
VIEWS = [('lh', (0, 180), 'lh lateral'), ('lh', (0, 0), 'lh medial'),
         ('rh', (0, 180), 'rh medial'), ('rh', (0, 0), 'rh lateral')]


def load_surfaces(fsavg):
    surf = {}
    for h in ('lh', 'rh'):
        v, f = fsio.read_geometry('%s/surf/%s.inflated' % (fsavg, h))
        lab, _ctab, names = fsio.read_annot('%s/label/%s.aparc.annot' % (fsavg, h))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        v = np.asarray(v, float)
        v -= v.mean(0)
        surf[h] = (v, np.asarray(f), lab, names)
    return surf


def facecolors(surf, h, values, norm, cmap):
    """Per-face RGBA from a {column name: value} mapping."""
    v, f, lab, names = surf[h]
    val = np.full(len(names), np.nan)
    for i, n in enumerate(names):
        key = '%s-%s' % (h, n)
        if key in values and np.isfinite(values[key]):
            val[i] = values[key]
    fl = lab[f[:, 0]]
    fv = np.where((fl >= 0) & (fl < len(names)),
                  val[np.clip(fl, 0, len(names) - 1)], np.nan)
    rgba = cmap(norm(fv))
    rgba[~np.isfinite(fv)] = matplotlib.colors.to_rgba(NA)
    return rgba


def paint(rows, title, subtitle, cbar_label, out, cmap='magma', vlo=None,
          vhi=None, fsavg='/data/disk2/freesurfer/subjects/fsaverage6',
          extend='neither'):
    """rows: [(row_label, {column: value}), ...] -- one brain row each."""
    surf = load_surfaces(fsavg)
    allv = np.array([x for _, d in rows for x in d.values() if np.isfinite(x)])
    lo = np.nanpercentile(allv, 2) if vlo is None else vlo
    hi = np.nanpercentile(allv, 98) if vhi is None else vhi
    norm = Normalize(lo, hi)
    cm = plt.get_cmap(cmap)
    nr = len(rows)
    # Wider and shorter per row than the first version: a 3-D axis with a
    # fixed box aspect leaves the spare space as vertical padding, so making
    # the figure wider fills it with brain instead of background.
    fig = plt.figure(figsize=(17.6, 3.15 * nr + 0.95), facecolor=SURF)
    for ri, (rlabel, values) in enumerate(rows):
        for ci, (h, (elev, azim), vlabel) in enumerate(VIEWS):
            ax = fig.add_subplot(nr, 4, ri * 4 + ci + 1, projection='3d',
                                 facecolor=SURF)
            v, f, _l, _n = surf[h]
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolors=facecolors(surf, h, values, norm, cm),
                linewidths=0, antialiased=False, shade=True,
                lightsource=matplotlib.colors.LightSource(azdeg=azim + 45,
                                                          altdeg=35)))
            # Per-axis limits, not a cube scaled by 0.60 of max|v|. The
            # inflated surface needs 0.96 of that along anterior-posterior,
            # so the cube clipped the occipital and frontal poles off every
            # medial view. box_aspect matches the extents, so fitting the
            # whole brain costs no distortion and fills the panel better.
            half = np.ptp(v, axis=0) / 2.0 * 1.02
            ax.set_xlim(-half[0], half[0])
            ax.set_ylim(-half[1], half[1])
            ax.set_zlim(-half[2], half[2])
            ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
            try:
                ax.set_box_aspect(tuple(half))
            except Exception:
                pass
            # Transparent, so a negative wspace overlaps neighbours without
            # the later axis painting its background over the earlier brain.
            ax.patch.set_alpha(0.0)
            if ri == nr - 1:
                ax.text2D(.5, .04, vlabel, transform=ax.transAxes, ha='center',
                          va='bottom', color=MUTED, fontsize=10.5)
        fig.text(0.017, 1 - (0.60 + ri) / nr * 0.93, rlabel, color=INK,
                 fontsize=12, ha='center', va='center', rotation=90)
    cax = fig.add_axes([0.952, 0.30, 0.011, 0.40])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax,
                      extend=extend)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=9)
    cb.set_label(cbar_label, color=MUTED, fontsize=10)
    H = fig.get_size_inches()[1]
    fig.text(0.012, 1 - 0.28 / H, title, color=INK, fontsize=15, ha='left',
             va='top')
    fig.text(0.012, 1 - 0.58 / H, subtitle, color=MUTED, fontsize=10,
             ha='left', va='top')
    fig.subplots_adjust(left=0.040, right=0.940, top=1 - 0.88 / H,
                        bottom=0.004, wspace=-0.14, hspace=-0.04)
    fig.savefig(out, dpi=135, facecolor=SURF)
    plt.close(fig)
    return out


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench-dir', required=True,
                    help="directory holding rebsamen_benchmarks.py's output")
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--atrophy-pipelines', nargs='+', default=None,
                    help='which pipelines get an atrophy figure (default: all)')
    ap.add_argument('--exclude', nargs='+', default=(),
                    help='pipelines to leave out of the reproducibility figure')
    ap.add_argument('--fsaverage',
                    default='/data/disk2/freesurfer/subjects/fsaverage6')
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    written = []

    rep = read_rows(os.path.join(args.bench_dir, 'reproducibility.csv'))
    skip = {'pipeline', 'global_pct', 'roi_avg_pct', 'n_sessions', 'n_scans'}
    rows = []
    for r in rep:
        if r['pipeline'] in set(args.exclude):
            continue
        vals = {k: float(v) for k, v in r.items()
                if k not in skip and v not in ('', 'nan')}
        rows.append(('%s\n(global %.3f%%)' % (r['pipeline'],
                                              float(r['global_pct'])), vals))
    n_sess, n_scans = rep[0]['n_sessions'], rep[0]['n_scans']
    written.append(paint(
        rows,
        'Reproducibility on same-session re-scans, per Desikan-Killiany parcel',
        'average absolute change from the within-session mean; %s sessions, '
        '%s scans. One colour scale across pipelines; grey = no value'
        % (n_sess, n_scans),
        'reproducibility error (%)', os.path.join(args.out_dir, 'fig5_reproducibility.png'),
        cmap='magma', vlo=0.0, extend='max'))

    atr = read_rows(os.path.join(args.bench_dir, 'atrophy.csv'))
    skip = {'pipeline', 'group', 'n_subjects', 'global_mean', 'global_sd'}
    pipes = args.atrophy_pipelines or sorted({r['pipeline'] for r in atr})
    order = ['CDR 0 (healthy)', 'CDR 0.5 (questionable)', 'CDR >= 1 (dementia)']
    # One colour scale across every atrophy figure produced in this call.
    # Computed per figure, the same colour means a different rate in each and
    # two pipelines cannot be read against one another.
    pool = [float(v) for r in atr if r['pipeline'] in set(pipes)
            for k, v in r.items() if k not in skip and v not in ('', 'nan')]
    a_lo = float(np.nanpercentile(pool, 2)) if pool else None
    a_hi = float(np.nanpercentile(pool, 98)) if pool else None
    for pipe in pipes:
        rows = []
        for g in order:
            hit = [r for r in atr if r['pipeline'] == pipe and r['group'] == g]
            if not hit:
                continue
            r = hit[0]
            vals = {k: float(v) for k, v in r.items()
                    if k not in skip and v not in ('', 'nan')}
            rows.append(('%s\nn = %s' % (g, r['n_subjects']), vals))
        if not rows:
            continue
        safe = pipe.replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        written.append(paint(
            rows,
            'Annual cortical thickness change by CDR group  |  %s' % pipe,
            'within-subject slope, mm/year; negative = thinning. One colour '
            'scale across groups AND pipelines; grey = no value',
            'mm/year', os.path.join(args.out_dir, 'fig6_atrophy_%s.png' % safe),
            cmap='viridis', vlo=a_lo, vhi=a_hi, extend='both'))

    for w in written:
        print('wrote', w)
    return 0


if __name__ == '__main__':
    sys.exit(main())
