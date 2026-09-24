"""Test-retest reproducibility of the parcellation, on same-session re-scans.

Two acquisitions of the same brain minutes apart: the true parcellation is
identical by construction, so any difference is error, with NO reference
parcellation involved. This is the measurement FreeSurfer's aparc cannot
referee and Dice cannot substitute for.

PRIMARY METRIC IS SURFACE AREA, not thickness. Thickness varies smoothly
across a border, so exchanging vertices between adjacent parcels swaps nearly
equal values: measured on one scan, relabelling 2.87% of vertices moved parcel
AREA by 1.70% (median) and mean THICKNESS by 0.25% -- area responds 6.7x more
to exactly the thing being varied. Thickness is reported alongside because it
is what the pipeline ships.

Five arms, same surfaces within a scan so only the labelling differs. They are
crossed deliberately: comparing only `voxel` against `ribbon` would confound
the unary source with the presence of the CRF.

    voxel        nearest-voxel HARD label from aparc.atlas+aseg, no inference
    *-clean      the raw labelling with small islands dissolved (absorb_islands),
                 which fixes contiguity WITHOUT moving any border
    column-raw   column_unary argmax, no inference
    column-crf   column_unary + banded anneal
    ribbon-raw   ribbon_unary argmax, no inference
    ribbon-crf   ribbon_unary + banded anneal

so: hard vs soft = voxel / ribbon-raw; the CRF's own contribution =
ribbon-raw / ribbon-crf and column-raw / column-crf; the unary source =
ribbon-* / column-*.
"""
import os, sys, csv, glob, time, shutil, subprocess
import numpy as np, nibabel as nib, scipy.sparse as sp
from scipy import spatial
sys.path.insert(0,'/data/disk2/projects/DL-DiReCT-V2')
from dldirect.mesh_crf import (META_CLASSES, column_unary, ribbon_unary,
                               border_sulcality_prior, banded_anneal,
                               absorb_islands)
from dldirect.hull_depth import (rasterize, hull_depth_field, sample,
                                 geodesic_zscore, crf_edge_weights)
from dldirect.field_pial_prototype import make_transforms, _mesh_adjacency
from dldirect import regional_stats as rs
S=os.path.dirname(os.path.abspath(__file__)); OAS='/data/disk2/oasis840'
PY_='/home/student/miniconda3/envs/DL_DiReCT/bin/python'
OUT=os.path.join(S,'repro30.csv'); THETA,BETA=2.0,1.0
LAB=os.path.join(S,'repro30_labels'); os.makedirs(LAB,exist_ok=True)
fsio=nib.freesurfer.io

def posterior_at(logit_dir, coords):
    files=sorted(glob.glob(os.path.join(logit_dir,'seg_*.nii.gz')))
    names=[os.path.basename(f)[4:-7] for f in files]
    k=[i for i,n in enumerate(names) if n not in META_CLASSES]
    files=[files[i] for i in k]; names=[names[i] for i in k]
    idx=tuple(coords.T)
    L=np.empty((len(idx[0]),len(files)),np.float32)
    for i,f in enumerate(files):
        L[:,i]=np.asarray(nib.load(f).dataobj,dtype=np.float32)[idx]
    L-=L.max(1,keepdims=True); P=np.exp(L); P/=np.maximum(P.sum(1,keepdims=True),1e-12)
    return P,names

def measure(scan, logit_dir, rows):
    C=os.path.join(OAS,scan)
    lut,valid,_=rs.get_labels()
    ref=nib.load(os.path.join(C,'mri','aparc.atlas+aseg.nii.gz'))
    parc=np.asarray(ref.dataobj).astype(np.int32)
    tovox,totkr=make_transforms(ref); spn=tuple(float(x) for x in ref.header.get_zooms()[:3])
    for h in ('lh','rh'):
        names=[k for k in valid if k.startswith(h+'-') and lut[k]>1000]
        ctx=np.array([lut[k] for k in names]); Ln=np.concatenate([ctx,[0]])
        lo,hi=(1000,1036) if h=='lh' else (2000,2036)
        w,wf=fsio.read_geometry(os.path.join(C,'field_pial_sigma0.65','%s.white'%h))
        p,pf=fsio.read_geometry(os.path.join(C,'field_pial_sigma0.65','%s.pial'%h))
        V=np.asarray(w,float); F=np.asarray(wf,int); P3=np.asarray(p,float)
        tri=np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)/2.0
        vor=np.zeros(len(V))
        for k in range(3): np.add.at(vor,F[:,k],tri/3.0)
        thick=np.linalg.norm(P3-V,axis=1)
        _m,Wm,deg=_mesh_adjacency(V,F)
        mask=rasterize(tovox(P3),pf,tuple(ref.shape[:3]))
        dep,_=hull_depth_field(mask,5.0,spn)
        z=np.asarray(geodesic_zscore(sample(dep,tovox(V)).astype(np.float64),V,F,50,
                                     adjacency=(Wm,deg))[0],float)
        edges,ew=crf_edge_weights(z,V,F,theta=THETA,floor=0.05)
        gm=(parc>lo)&(parc<hi); co=np.array(np.nonzero(gm)).T
        post,allnames=posterior_at(logit_dir,co)
        # arm 1: nearest ribbon voxel, hard label from the model's parcellation
        tre=spatial.cKDTree(totkr(co.astype(float)))
        dn,jn=tre.query(V)
        vox=np.where(dn<=1.5,parc[tuple(co[jn].T)],0).astype(np.int32)
        arms={'voxel':vox}
        for tag,un in (('column',column_unary(V,P3,logit_dir,names,tovox)[0]),
                       ('ribbon',ribbon_unary(V,parc,lo,hi,post,names,allnames,totkr)[0])):
            raw=Ln[un.argmin(1)]
            arms[tag+'-raw']=raw
            # the minimal alternative to smoothing: dissolve small islands,
            # move no borders
            arms[tag+'-clean']=absorb_islands(raw,Wm,vertex_area=vor)[0]
            G=border_sulcality_prior(raw,edges,z,Ln,scale=0.6)
            lab,_n=banded_anneal(un,edges,ew,G,Wm,beta=BETA)
            arms[tag+'-crf']=Ln[lab]
        from scipy.sparse.csgraph import connected_components
        Ab=(Wm>0).tocsr()
        # SAVE THE LABELLINGS. The expensive part of a scan is the model and
        # the logits; every label-space experiment after that (island
        # absorption, a different threshold, any other post-process) is seconds
        # of work and must never require re-running the chain.
        np.savez_compressed(os.path.join(LAB,'%s.%s.npz'%(scan,h)),
                            vor=vor.astype(np.float32),
                            thick=thick.astype(np.float32),
                            **{t:v.astype(np.int32) for t,v in arms.items()})
        for tag,ids in arms.items():
            for pid,nm in zip(ctx,names):
                m=ids==pid
                if not m.any(): continue
                # islands: components of this parcel, and the share of its area
                # outside the largest one. The raw arms carry no smoothing at
                # all, so fragmentation is the obvious thing they might be
                # trading reproducibility against.
                ncomp, frag = 1, 0.0
                if m.sum() > 1:
                    k, cc = connected_components(Ab[m][:, m], directed=False)
                    ncomp = int(k)
                    if k > 1:
                        a = np.array([vor[np.where(m)[0][cc == c]].sum() for c in range(k)])
                        frag = float(1.0 - a.max() / a.sum())
                rows.append(dict(scan=scan,hemi=h,arm=tag,parcel=nm,
                                 area=round(float(vor[m].sum()),4),
                                 thickness=round(float(np.average(thick[m],weights=vor[m])),5),
                                 nvert=int(m.sum()),ncomp=ncomp,frag=round(frag,5)))

def main():
    pairs=[l.strip() for l in open(os.path.join(S,'pairs30.txt')) if l.strip()]
    done=set()
    if os.path.exists(OUT):
        done={r['scan'] for r in csv.DictReader(open(OUT))}
    for i,ses in enumerate(pairs):
        for run in ('run-01','run-02'):
            scan='%s_%s'%(ses,run)
            if scan in done: continue
            t0=time.time(); rows=[]; ld=os.path.join(S,'repro_logits',scan)
            try:
                os.makedirs(ld,exist_ok=True)
                subprocess.run([PY_,os.path.join(S,'ds_all_logits.py'),'--model','v0_f1',
                                os.path.join(OAS,scan,'T1w_norm_noskull_cropped.nii.gz'),ld,scan],
                               check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                measure(scan,ld,rows)
            except Exception as e:
                print('FAIL %-34s %s: %s'%(scan,type(e).__name__,e),flush=True)
                shutil.rmtree(ld,ignore_errors=True); continue
            shutil.rmtree(ld,ignore_errors=True)
            new=not os.path.exists(OUT)
            with open(OUT,'a',newline='') as fh:
                wri=csv.DictWriter(fh,fieldnames=list(rows[0]))
                if new: wri.writeheader()
                wri.writerows(rows)
            print('[%2d/%d] %-34s %4.0fs'%(i+1,len(pairs),scan,time.time()-t0),flush=True)

if __name__=='__main__': main()
