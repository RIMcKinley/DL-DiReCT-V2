"""Is the fundus gain the SULCAL GATING, or just Potts?

Four arms at matched beta, crossing the two places sulcality enters the
energy: the edge weights w, and the per-border prior G. The flat control
uses mean(w_gated) rather than 1.0, so total pairwise mass matches and the
only difference is WHERE the smoothness sits, not how much there is.

Caches the unary and the per-vertex fields, so later sweeps need no GPU.
"""
import os, sys, csv, time, subprocess, shutil
import numpy as np, nibabel as nib
sys.path.insert(0,'/data/disk2/projects/DL-DiReCT-V2')
from dldirect.mesh_crf import (column_unary, soft_unary, border_sulcality_prior,
                               icm_pairwise, dice_per_label, n_stray,
                               freesurfer_labels_on_our_mesh, boundary_sulcality,
                               fs_vertex_map, fs_morph_on_our_mesh)
from dldirect.hull_depth import (rasterize, hull_depth_field, sample,
                                 geodesic_zscore, crf_edge_weights)
from dldirect.field_pial_prototype import make_transforms, _mesh_adjacency
from dldirect import regional_stats as rs
fsio = nib.freesurfer.io

S   = os.path.dirname(os.path.abspath(__file__))
OAS = '/data/disk2/oasis840'
FSR = '/str/nas/MORPHOMETRY/OASIS3/FS_RESULTS'
PY_ = '/home/student/miniconda3/envs/DL_DiReCT/bin/python'
RUN = 'field_pial_sigma0.65'
OUT = os.path.join(S, 'crf30_control.csv')
CACHE = os.path.join(S, 'crf30_cache')
CELLS = [(0.5, 1.0), (0.25, 4.0), (2.0, 8.0)]

def md(a, b, L):
    return float(np.mean(list(dice_per_label(a, b, L).values())))

def build(case, h, logdir):
    """unary + the per-vertex fields, cached so this never needs the GPU twice"""
    f = os.path.join(CACHE, '%s.%s.npz' % (case, h))
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files}
    c = os.path.join(OAS, case); fs = os.path.join(FSR, case)
    lut, valid, _ = rs.get_labels()
    ref = nib.load(os.path.join(c, 'mri', 'aparc.atlas+aseg.nii.gz'))
    tovox, _ = make_transforms(ref)
    parc = np.asarray(ref.dataobj).astype(np.int32)
    spn = tuple(float(z) for z in ref.header.get_zooms()[:3])
    names = [k for k in valid if k.startswith(h + '-') and lut[k] > 1000]
    L34 = np.array([lut[k] for k in names]); Ln = np.concatenate([L34, [0]])
    w, wf = fsio.read_geometry(os.path.join(c, RUN, '%s.white' % h))
    p, pf = fsio.read_geometry(os.path.join(c, RUN, '%s.pial' % h))
    wv = tovox(np.asarray(w, float))
    un, _pr, _ = column_unary(w, p, logdir, names, tovox)
    _, Wm, deg = _mesh_adjacency(np.asarray(w, float), np.asarray(wf))
    mask = rasterize(tovox(np.asarray(p, float)), pf, tuple(ref.shape[:3]))
    dep, _ = hull_depth_field(mask, 5.0, spn)
    z, _, _ = geodesic_zscore(sample(dep, wv).astype(np.float64), w, wf, 50,
                              adjacency=(Wm, deg))
    _v, votes = soft_unary(wv, parc, L34); cort = votes.sum(1) > 0
    j = fs_vertex_map(c, fs, h, np.asarray(w, float))
    fsl = freesurfer_labels_on_our_mesh(c, fs, h, w, Ln, names + ['%s-NULL' % h], j=j)
    sulc = fs_morph_on_our_mesh(c, fs, h, w, 'sulc', j=j)
    d = dict(unary=un.astype(np.float32), z=np.asarray(z, np.float32),
             verts=np.asarray(w, np.float32), faces=np.asarray(wf, np.int32),
             fsl=fsl.astype(np.int32), sulc=sulc.astype(np.float32),
             cort=cort, labels=Ln.astype(np.int32))
    os.makedirs(CACHE, exist_ok=True); np.savez_compressed(f, **d)
    return d

def score(case, logdir, rows):
    for h in ('lh', 'rh'):
        d = build(case, h, logdir)
        un, zf = d['unary'], np.asarray(d['z'], float)
        w, wf, Ln = d['verts'], d['faces'], d['labels']
        L34 = Ln[Ln > 0]
        fsl, sulc = d['fsl'], np.asarray(d['sulc'], float)
        keep = d['cort'] & (fsl > 0)
        _, Wm, _dg = _mesh_adjacency(np.asarray(w, float), np.asarray(wf))
        base = Ln[un.argmin(1)]
        Guni = np.ones((len(Ln), len(Ln)), np.float32); np.fill_diagonal(Guni, 0.0)
        for th, beta in CELLS:
            edges, ew = crf_edge_weights(zf, w, wf, theta=th, floor=0.05)
            flat = np.full_like(ew, float(ew.mean()))
            Gsul = border_sulcality_prior(base, edges, zf, Ln, scale=0.6)
            for wtag, W in (('gated', ew), ('flat', flat)):
                for gtag, G in (('Gsulc', Gsul), ('Guni', Guni)):
                    out = Ln[icm_pairwise(un, edges, W, G, beta=beta)[0]]
                    rows.append(dict(
                        case=case, hemi=h, cell='t%gb%g' % (th, beta),
                        weights=wtag, prior=gtag,
                        dice=round(md(out[keep], fsl[keep], L34), 5),
                        gap_z=round(boundary_sulcality(out, edges, zf, keep)[2], 4),
                        gap_sulc=round(boundary_sulcality(out, edges, sulc, keep)[2], 4),
                        strays=n_stray(out, Wm, Ln),
                        moved=round(float((out[keep] != base[keep]).mean()), 5)))
        # the unary row, once per hemisphere, on the theta=1 edge set
        ed = crf_edge_weights(zf, w, wf, theta=1.0, floor=0.05)[0]
        rows.append(dict(case=case, hemi=h, cell='unary', weights='-', prior='-',
                         dice=round(md(base[keep], fsl[keep], L34), 5),
                         gap_z=round(boundary_sulcality(base, ed, zf, keep)[2], 4),
                         gap_sulc=round(boundary_sulcality(base, ed, sulc, keep)[2], 4),
                         strays=n_stray(base, Wm, Ln), moved=0.0))

def main():
    cases = [l.strip() for l in open(os.path.join(S, 'subjects30.txt')) if l.strip()]
    done = set()
    if os.path.exists(OUT):
        done = {r['case'] for r in csv.DictReader(open(OUT))}
    for i, case in enumerate(cases):
        if case in done:
            print('skip %s' % case, flush=True); continue
        t0 = time.time(); rows = []
        cached = all(os.path.exists(os.path.join(CACHE, '%s.%s.npz' % (case, h)))
                     for h in ('lh', 'rh'))
        logdir = os.path.join(S, 'logit_tmp', case)
        try:
            if not cached:
                os.makedirs(logdir, exist_ok=True)
                subprocess.run([PY_, os.path.join(S, 'ds_all_logits.py'), '--model',
                                'v0_f1', os.path.join(OAS, case,
                                'T1w_norm_noskull_cropped.nii.gz'), logdir, case],
                               check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            score(case, logdir, rows)
        except Exception as e:
            print('FAIL %-32s %s: %s' % (case, type(e).__name__, e), flush=True)
            shutil.rmtree(logdir, ignore_errors=True); continue
        shutil.rmtree(logdir, ignore_errors=True)
        new = not os.path.exists(OUT)
        with open(OUT, 'a', newline='') as fh:
            wri = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new: wri.writeheader()
            wri.writerows(rows)
        print('[%2d/%d] %-32s %5.0fs' % (i + 1, len(cases), case, time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
