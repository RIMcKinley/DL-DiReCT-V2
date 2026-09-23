"""Sweep --g-scale, judged PER PARCEL PAIR against FreeSurfer's border depth.

g = clip(median_z / scale, 0, 1): a small scale saturates every pair at g=1
(all borders loosened at fundi -- plain Potts), a large one spreads them down
(more borders made to pay full price, as a gyral border should).

The cohort run showed our per-pair border depths are COMPRESSED toward the
mean relative to FreeSurfer's -- too shallow where FreeSurfer's border is
deep, too deep where it is shallow. If g-scale can decompress them, it shows
up here as the crown bias falling without the sulcal bias falling with it.

Runs off the cached unary/fields: no GPU, no model.
"""
import os, sys, glob, csv, numpy as np
sys.path.insert(0,'/data/disk2/projects/DL-DiReCT-V2')
from dldirect.mesh_crf import (border_sulcality_prior, icm_pairwise, n_stray,
                               dice_per_label, boundary_sulcality)
from dldirect.hull_depth import crf_edge_weights
from dldirect.field_pial_prototype import _mesh_adjacency
S = os.path.dirname(os.path.abspath(__file__))
THETA, BETA = 2.0, 8.0
SCALES = [0.0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0]   # 0.0 = uniform G (no prior)
MIN_EDGES = 30
OUT = os.path.join(S, 'gscale.csv')

def pair_medians(ids, edges, z, keep, pairs=None):
    a, b = ids[edges[:,0]], ids[edges[:,1]]
    ok = keep[edges[:,0]] & keep[edges[:,1]] & (a != b)
    ze = np.maximum(z[edges[:,0]], z[edges[:,1]])
    lo, hi, zz = np.minimum(a,b)[ok], np.maximum(a,b)[ok], ze[ok]
    key = lo.astype(np.int64) * 100000 + hi
    out = {}
    for k in np.unique(key):
        m = key == k
        if m.sum() >= MIN_EDGES:
            out[int(k)] = float(np.median(zz[m]))
    return out

rows = []
files = sorted(glob.glob(os.path.join(S,'crf30_cache','*.npz')))
for fi, f in enumerate(files):
    d = np.load(f, allow_pickle=True)
    un, zf = d['unary'], np.asarray(d['z'], float)
    w, wf, Ln = d['verts'], d['faces'], d['labels']
    fsl = d['fsl']; keep = d['cort'] & (fsl > 0); L34 = Ln[Ln > 0]
    sulc = np.asarray(d['sulc'], float)
    _m, Wm, _dg = _mesh_adjacency(np.asarray(w,float), np.asarray(wf))
    edges, ew = crf_edge_weights(zf, w, wf, theta=THETA, floor=0.05)
    base = Ln[un.argmin(1)]
    fz = pair_medians(fsl, edges, zf, keep)          # FreeSurfer's own
    for sc in SCALES:
        if sc == 0.0:
            G = np.ones((len(Ln), len(Ln)), np.float32); np.fill_diagonal(G, 0.0)
        else:
            G = border_sulcality_prior(base, edges, zf, Ln, scale=sc)
        out = Ln[icm_pairwise(un, edges, ew, G, beta=BETA)[0]]
        oz = pair_medians(out, edges, zf, keep)
        common = [k for k in fz if k in oz]
        fv = np.array([fz[k] for k in common]); ov = np.array([oz[k] for k in common])
        hi = fv >= np.median(fv)
        rows.append(dict(
            case=os.path.basename(f)[:-4], gscale=sc,
            npair=len(common),
            r=round(float(np.corrcoef(fv, ov)[0,1]), 4),
            mae=round(float(np.mean(np.abs(ov-fv))), 4),
            bias=round(float(np.mean(ov-fv)), 4),
            sulcal=round(float(np.mean(ov[hi]-fv[hi])), 4),
            crown=round(float(np.mean(ov[~hi]-fv[~hi])), 4),
            spread=round(float(ov.std()/max(fv.std(),1e-9)), 4),
            dice=round(float(np.mean(list(dice_per_label(out[keep], fsl[keep], L34).values()))), 5),
            gap_sulc=round(boundary_sulcality(out, edges, sulc, keep)[2], 4),
            strays=n_stray(out, Wm, Ln)))
    print('%d/%d %s' % (fi+1, len(files), os.path.basename(f)), flush=True)
with open(OUT,'w',newline='') as fh:
    wri = csv.DictWriter(fh, fieldnames=list(rows[0])); wri.writeheader(); wri.writerows(rows)
print('wrote', OUT)
