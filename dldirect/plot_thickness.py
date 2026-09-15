#!/usr/bin/env python
"""Plot regional_stats output against a FreeSurfer reference.

Three views, one subcommand each:

  scatter   each metric against FreeSurfer, one point per subject, on the
            hemisphere means. Identity line and least-squares fit; r, bias and
            slope annotated.
  regions   the same, faceted region x metric, for named parcels (or the best /
            median / worst agreeing ones, chosen from the data).
  brain     r and slope per Desikan-Killiany parcel painted on an fsaverage
            inflated surface, four views. The ggseg idea, drawn with matplotlib
            because neither R/ggseg nor nilearn/surfplot is installed here.

WHY SLOPE AND NOT ONLY r

On the hemisphere means every metric comes out at slope ~0.99 against
FreeSurfer, which reads as a clean additive offset. Per parcel it is not: the
median slope is 0.83 for sym_nn and most parcels compress FreeSurfer's
between-subject range. Averaging 68 parcels into one number cancels the
compression against the regions that expand, and manufactures the unity slope.
Plot both, and prefer the regional view when deciding whether a definition
merely shifts the scale or distorts it.

COLOUR. r is a magnitude on a fixed 0-1 scale so the metric figures are
comparable with each other; the default ramp is plasma. slope is shown on a
window centred on 1.0 (default 0.75-1.25) with `extend` arrows, because at the
full data range the interesting mid-band compresses to one colour -- but a third
of parcels fall outside that window for the nearest-neighbour metrics, so the
clipped counts are annotated at each end rather than left to look saturated.
"""

import argparse
import os
import sys

import numpy as np
import nibabel.freesurfer.io as fsio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .freesurfer_stats import load_comparison

INK, SEC, MUTED, GRID, SURF, BLUE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb', '#2a78d6'
NA = '#dedcd4'
METRICS = ('field', 'travel', 'nn', 'sym_nn')


def _fit(a, b):
    """(r, bias, slope) of `a` against reference `b`, finite entries only."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    return (np.corrcoef(a[ok], b[ok])[0, 1], (a[ok] - b[ok]).mean(),
            np.polyfit(b[ok], a[ok], 1)[0])


def _panel(ax, x, y, label):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.08
    lim = (lo - pad, hi + pad)
    ax.set_facecolor(SURF)
    ax.plot(lim, lim, ls='--', lw=1.1, color=MUTED, zorder=1)
    ax.scatter(x, y, s=24, color=BLUE, alpha=.75, edgecolors=SURF, linewidths=.5, zorder=3)
    c = np.polyfit(x, y, 1)
    ax.plot(np.array(lim), np.polyval(c, np.array(lim)), color=BLUE, lw=1.3, alpha=.5, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.grid(color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.text(.04, .955, label, transform=ax.transAxes, va='top', color=SEC,
            fontsize=9.2, linespacing=1.45)


def scatter(subjects, fs_csv, prep_root, out, out_name, metrics=METRICS):
    """Hemisphere-mean scatter, one panel per metric."""
    n = len(metrics)
    fig, axes = plt.subplots(2, int(np.ceil((n + 1) / 2)),
                             figsize=(4.7 * int(np.ceil((n + 1) / 2)), 9.6), facecolor=SURF)
    axes = axes.ravel()
    for ax, m in zip(axes, metrics):
        cols, A, FS = load_comparison(subjects, m, fs_csv, prep_root, out_name)
        a = np.nanmean(A[:, -2:], axis=1)
        b = np.nanmean(FS[:, -2:], axis=1)
        r, bias, slope = _fit(a, b)
        _panel(ax, b, a, 'r = %.3f\nbias %+.3f mm\nslope %.2f' % (r, bias, slope))
        ax.set_title(m, color=INK, fontsize=13, loc='left', pad=8)
        ax.set_xlabel('FreeSurfer (mm)', color=MUTED, fontsize=10)
        ax.set_ylabel('%s (mm)' % m, color=MUTED, fontsize=10)
    for ax in axes[len(metrics):]:
        ax.axis('off')
    axes[len(metrics)].text(.02, .8, 'dashed = identity (y = x)\nsolid  = least-squares fit\n\n'
                            'one point per subject, n = %d\nhemisphere-mean thickness'
                            % len(subjects), transform=axes[len(metrics)].transAxes,
                            color=SEC, fontsize=10.5, va='top', linespacing=1.6)
    fig.text(0.008, 0.985, 'Each thickness definition against FreeSurfer, per subject',
             color=INK, fontsize=15, ha='left', va='top')
    fig.subplots_adjust(top=0.93, hspace=0.30, wspace=0.28, left=0.055, right=0.985, bottom=0.06)
    fig.savefig(out, dpi=140, facecolor=SURF)
    return out


def regions(subjects, fs_csv, prep_root, out, out_name, names=None,
            rank_by='sym_nn', metrics=METRICS):
    """Region x metric scatter grid. `names=None` picks best/median/worst by r."""
    cols, A, FS = load_comparison(subjects, rank_by, fs_csv, prep_root, out_name)
    if names is None:
        rs = np.array([_fit(A[:, j], FS[:, j])[0] for j in range(68)])
        o = np.argsort(rs)
        names = [(cols[o[-1]], 'best agreement'),
                 (cols[o[len(o) // 2]], 'typical (median r)'),
                 (cols[o[0]], 'worst agreement')]
    else:
        names = [(n, '') for n in names]

    tabs = {m: load_comparison(subjects, m, fs_csv, prep_root, out_name) for m in metrics}
    fig, axes = plt.subplots(len(names), len(metrics),
                             figsize=(3.75 * len(metrics), 3.8 * len(names)), facecolor=SURF)
    axes = np.atleast_2d(axes)
    for i, (reg, tag) in enumerate(names):
        j = cols.index(reg)
        for k, m in enumerate(metrics):
            _, Am, FSm = tabs[m]
            r, bias, slope = _fit(Am[:, j], FSm[:, j])
            _panel(axes[i, k], FSm[:, j], Am[:, j],
                   'r %.3f\nbias %+.2f\nslope %.2f' % (r, bias, slope))
            if i == 0:
                axes[i, k].set_title(m, color=INK, fontsize=12.5, pad=8)
            if k == 0:
                axes[i, k].set_ylabel('%s\n\nthis pipeline (mm)' % reg,
                                      color=SEC, fontsize=10.5, linespacing=1.3)
            if i == len(names) - 1:
                axes[i, k].set_xlabel('FreeSurfer (mm)', color=MUTED, fontsize=9.5)
        if tag:
            axes[i, -1].text(1.04, .5, tag, transform=axes[i, -1].transAxes, rotation=270,
                             va='center', ha='left', color=MUTED, fontsize=10)
    fig.text(0.006, 0.988, 'Per-region thickness against FreeSurfer, one point per subject',
             color=INK, fontsize=15, ha='left', va='top')
    fig.subplots_adjust(top=0.925, hspace=0.26, wspace=0.30, left=0.075, right=0.975, bottom=0.055)
    fig.savefig(out, dpi=135, facecolor=SURF)
    return out


def brain(subjects, fs_csv, prep_root, out, out_name, metric='sym_nn',
          fsavg='/data/disk2/freesurfer/subjects/fsaverage6',
          r_cmap='plasma', slope_cmap='viridis', slope_lo=0.75, slope_hi=1.25):
    """r and slope per parcel on an inflated surface, four views."""
    cols, A, FS = load_comparison(subjects, metric, fs_csv, prep_root, out_name)
    stat = {}
    for j, c in enumerate(cols[:68]):
        r, _b, s = _fit(A[:, j], FS[:, j])
        if np.isfinite(r):
            stat[c] = (r, s)

    surf = {}
    for h in ('lh', 'rh'):
        v, f = fsio.read_geometry('%s/surf/%s.inflated' % (fsavg, h))
        lab, _ctab, names = fsio.read_annot('%s/label/%s.aparc.annot' % (fsavg, h))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        v = np.asarray(v, float); v -= v.mean(0)
        surf[h] = (v, np.asarray(f), lab, names)

    def facecolors(h, which, norm, cmap):
        v, f, lab, names = surf[h]
        val = np.full(len(names), np.nan)
        for i, n in enumerate(names):
            key = '%s-%s' % (h, n)
            if key in stat:
                val[i] = stat[key][which]
        fl = lab[f[:, 0]]
        fv = np.where((fl >= 0) & (fl < len(names)), val[np.clip(fl, 0, len(names) - 1)], np.nan)
        rgba = cmap(norm(fv))
        rgba[~np.isfinite(fv)] = matplotlib.colors.to_rgba(NA)
        return rgba

    vals = np.array(list(stat.values()))
    VIEWS = [('lh', (0, 180), 'lh lateral'), ('lh', (0, 0), 'lh medial'),
             ('rh', (0, 180), 'rh medial'), ('rh', (0, 0), 'rh lateral')]
    ROWS = [('correlation r', 0, plt.get_cmap(r_cmap), Normalize(0.0, 1.0)),
            ('slope', 1, plt.get_cmap(slope_cmap), Normalize(slope_lo, slope_hi))]
    fig = plt.figure(figsize=(15.5, 7.6), facecolor=SURF)
    for ri, (rlabel, which, cmap, norm) in enumerate(ROWS):
        for ci, (h, (elev, azim), vlabel) in enumerate(VIEWS):
            ax = fig.add_subplot(2, 4, ri * 4 + ci + 1, projection='3d', facecolor=SURF)
            v, f, _lab, _names = surf[h]
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolors=facecolors(h, which, norm, cmap), linewidths=0,
                antialiased=False, shade=True,
                lightsource=matplotlib.colors.LightSource(azdeg=azim + 45, altdeg=35)))
            lim = np.abs(v).max() * 0.60
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            ax.text2D(.5, .04, vlabel, transform=ax.transAxes, ha='center', va='bottom',
                      color=MUTED, fontsize=10.5)
        cax = fig.add_axes([0.950, 0.55 - ri * 0.45, 0.011, 0.28])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          extend='both' if which == 1 else 'neither')
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=9)
        if which == 1:
            t = [slope_lo, (slope_lo + 1) / 2, 1.0, (1 + slope_hi) / 2, slope_hi]
            cb.set_ticks(t); cb.set_ticklabels(['%.2f' % x for x in t])
            cb.ax.axhline(0.5, color='#ffffff', lw=1.6)   # 1.0, the exact middle
            cb.ax.get_yticklabels()[2].set_color(INK)
            cb.ax.get_yticklabels()[2].set_fontweight('bold')
            n_lo = int((vals[:, 1] < slope_lo).sum()); n_hi = int((vals[:, 1] > slope_hi).sum())
            cb.ax.text(0.5, -0.14, '%d parcels\nbelow %.2f' % (n_lo, slope_lo),
                       transform=cb.ax.transAxes, ha='center', va='top',
                       color=MUTED, fontsize=8.5, linespacing=1.3)
            cb.ax.text(0.5, 1.14, '%d above %.2f' % (n_hi, slope_hi),
                       transform=cb.ax.transAxes, ha='center', va='bottom',
                       color=MUTED, fontsize=8.5)
        fig.text(0.017, 0.695 - ri * 0.455, rlabel, color=INK, fontsize=12.5,
                 ha='center', va='center', rotation=90)
    fig.text(0.012, 0.985, 'Regional agreement with FreeSurfer, per Desikan-Killiany parcel  (%s)'
             % metric, color=INK, fontsize=15, ha='left', va='top')
    fig.text(0.012, 0.955, 'n = %d subjects - %s inflated; grey = unknown / corpus callosum '
             '(no value). r on a fixed 0-1 scale, so the metric figures are comparable'
             % (len(subjects), os.path.basename(fsavg)),
             color=MUTED, fontsize=10, ha='left', va='top')
    fig.subplots_adjust(left=0.045, right=0.935, top=0.93, bottom=0.005, wspace=-0.06, hspace=0.02)
    fig.savefig(out, dpi=135, facecolor=SURF)
    print('%s  r: %.3f-%.3f (median %.3f)   slope: %.3f-%.3f (median %.3f)'
          % (metric, vals[:, 0].min(), vals[:, 0].max(), np.median(vals[:, 0]),
             vals[:, 1].min(), vals[:, 1].max(), np.median(vals[:, 1])))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('kind', choices=('scatter', 'regions', 'brain'))
    p.add_argument('--subjects', required=True, help='one subject directory name per line')
    p.add_argument('--fs-csv', required=True, help='table from freesurfer_stats.py')
    p.add_argument('--prep-root', required=True, help='directory holding the subject preps')
    p.add_argument('--out', required=True)
    p.add_argument('--out-name', default='field_pial_sigma0.65',
                   help='subdirectory inside each prep holding the result-thick CSVs')
    p.add_argument('--metric', default='sym_nn', help='brain: which metric to map')
    p.add_argument('--metrics', nargs='+', default=list(METRICS))
    p.add_argument('--regions', nargs='+', help='regions: name them instead of ranking')
    p.add_argument('--slope-lo', type=float, default=0.75)
    p.add_argument('--slope-hi', type=float, default=1.25)
    p.add_argument('--r-cmap', default='plasma')
    p.add_argument('--slope-cmap', default='viridis')
    p.add_argument('--fsaverage', default='/data/disk2/freesurfer/subjects/fsaverage6')
    args = p.parse_args()

    subs = [l.strip() for l in open(args.subjects) if l.strip()]
    if args.kind == 'scatter':
        out = scatter(subs, args.fs_csv, args.prep_root, args.out, args.out_name,
                      tuple(args.metrics))
    elif args.kind == 'regions':
        out = regions(subs, args.fs_csv, args.prep_root, args.out, args.out_name,
                      args.regions, metrics=tuple(args.metrics))
    else:
        out = brain(subs, args.fs_csv, args.prep_root, args.out, args.out_name,
                    args.metric, args.fsaverage, args.r_cmap, args.slope_cmap,
                    args.slope_lo, args.slope_hi)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
