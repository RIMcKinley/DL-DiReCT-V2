"""Judge the per-border prior G where it actually claims to act: PER PAIR.

The aggregate fundus gap cannot score G. G's whole purpose is that a pair of
parcels whose true border runs over a gyral crown should NOT be pushed into a
fundus, so a lower global gap under G may be correct behaviour. The test is
whether our per-pair border sulcality tracks FREESURFER'S per-pair border
sulcality -- deep where FreeSurfer's border is deep, shallow where it is not.
"""
import os, sys, glob, numpy as np
sys.path.insert(0,'/data/disk2/projects/DL-DiReCT-V2')
from dldirect.mesh_crf import (border_sulcality_prior, icm_pairwise, n_stray)
from dldirect.hull_depth import crf_edge_weights
from dldirect.field_pial_prototype import _mesh_adjacency
S = os.path.dirname(os.path.abspath(__file__))
CELL = (2.0, 8.0)
MIN_EDGES = 30

def pair_z(ids, edges, z, keep, pairs):
    a, b = ids[edges[:,0]], ids[edges[:,1]]
    ok = keep[edges[:,0]] & keep[edges[:,1]] & (a != b)
    ze = np.maximum(z[edges[:,0]], z[edges[:,1]])
    lo, hi = np.minimum(a,b)[ok], np.maximum(a,b)[ok]; zz = ze[ok]
    out = {}
    for (p,q) in pairs:
        m = (lo==p)&(hi==q)
        if m.sum() >= MIN_EDGES: out[(p,q)] = float(np.median(zz[m]))
    return out

def pairs_of(ids, edges, keep):
    a, b = ids[edges[:,0]], ids[edges[:,1]]
    ok = keep[edges[:,0]] & keep[edges[:,1]] & (a != b)
    lo, hi = np.minimum(a,b)[ok], np.maximum(a,b)[ok]
    u, c = np.unique(np.stack([lo,hi],1), axis=0, return_counts=True)
    return [tuple(x) for x, n in zip(u, c) if n >= MIN_EDGES]

rows = {'Gsulc': [], 'Guni': []}
files = sorted(glob.glob(os.path.join(S,'crf30_cache','*.npz')))
for f in files:
    d = np.load(f, allow_pickle=True)
    un, zf = d['unary'], np.asarray(d['z'], float)
    w, wf, Ln = d['verts'], d['faces'], d['labels']
    fsl, keep = d['fsl'], d['cort'] & (d['fsl'] > 0)
    _m, Wm, _dg = _mesh_adjacency(np.asarray(w,float), np.asarray(wf))
    th, beta = CELL
    edges, ew = crf_edge_weights(zf, w, wf, theta=th, floor=0.05)
    base = Ln[un.argmin(1)]
    Gs = border_sulcality_prior(base, edges, zf, Ln, scale=0.6)
    Gu = np.ones((len(Ln),len(Ln)), np.float32); np.fill_diagonal(Gu, 0.0)
    P = pairs_of(fsl, edges, keep)
    fz = pair_z(fsl, edges, zf, keep, P)             # FreeSurfer's own, per pair
    for tag, G in (('Gsulc', Gs), ('Guni', Gu)):
        out = Ln[icm_pairwise(un, edges, ew, G, beta=beta)[0]]
        oz = pair_z(out, edges, zf, keep, P)
        common = [k for k in fz if k in oz]
        rows[tag].append((np.array([fz[k] for k in common]),
                          np.array([oz[k] for k in common]),
                          os.path.basename(f)))
print('%d hemispheres cached\n' % len(files))
print('%-7s %8s %8s %9s %9s %9s' % ('G','r','MAE','bias','sulcal','crown'))
for tag in ('Gsulc','Guni'):
    R, MAE, BIAS, SU, CR = [], [], [], [], []
    for fzv, ozv, _n in rows[tag]:
        R.append(np.corrcoef(fzv, ozv)[0,1]); MAE.append(np.mean(np.abs(ozv-fzv)))
        BIAS.append(np.mean(ozv-fzv))
        hi = fzv >= np.median(fzv)                    # FS says this border IS sulcal
        SU.append(np.mean(ozv[hi]-fzv[hi])); CR.append(np.mean(ozv[~hi]-fzv[~hi]))
    print('%-7s %8.3f %8.3f %+9.3f %+9.3f %+9.3f'
          % (tag, np.median(R), np.median(MAE), np.median(BIAS), np.median(SU), np.median(CR)))
print('\nper-hemisphere, Gsulc vs Guni:')
r1=np.array([np.corrcoef(a,b)[0,1] for a,b,_ in rows['Gsulc']])
r2=np.array([np.corrcoef(a,b)[0,1] for a,b,_ in rows['Guni']])
e1=np.array([np.mean(np.abs(b-a)) for a,b,_ in rows['Gsulc']])
e2=np.array([np.mean(np.abs(b-a)) for a,b,_ in rows['Guni']])
print('  correlation with FS per-pair depth: Gsulc better in %d/%d (median %+.4f)'
      %( (r1>r2).sum(), len(r1), np.median(r1-r2)))
print('  |error| vs FS per-pair depth:       Gsulc better in %d/%d (median %+.4f)'
      %( (e1<e2).sum(), len(e1), np.median(e1-e2)))
