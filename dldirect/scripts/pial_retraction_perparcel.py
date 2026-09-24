"""Where does the retraction's thickness cost fall?

-0.020 mm of mean thickness at cap 1.0 is the whole-hemisphere figure. The
crossings being repaired are opposing sulcal banks in contact, so the cost
should concentrate in parcels with tight sulci rather than spread evenly --
and if it does, it is a per-parcel bias, not a global offset.
"""
import os,sys,collections,numpy as np,nibabel as nib
sys.path.insert(0,'/data/disk2/projects/DL-DiReCT-V2')
from dldirect import pial_clean as pc
from dldirect import field_pial_prototype as fp
from dldirect.field_pial_prototype import make_transforms, _mesh_adjacency
from dldirect import regional_stats as rs
S=os.path.dirname(os.path.abspath(__file__)); OAS='/data/disk2/oasis840'
fsio=nib.freesurfer.io; lut,valid,_=rs.get_labels(); inv={v:k for k,v in lut.items()}
CAP=1.0
cases=[l.strip() for l in open(S+'/pairs30.txt')][:6]
acc=collections.defaultdict(list)
for ses in cases:
    scan=ses+'_run-01'; C=os.path.join(OAS,scan)
    vp=os.path.join(C,'field_pial_sigma0.65','pial_Velocity.nii.gz')
    if not os.path.exists(vp): continue
    ref=nib.load(os.path.join(C,'mri','aparc.atlas+aseg.nii.gz'))
    parc=np.asarray(ref.dataobj).astype(np.int32); tovox,totkr=make_transforms(ref)
    vel=np.asarray(nib.load(vp).dataobj)
    segv=np.zeros(parc.shape,np.int32); segv[parc>0]=2; segv[(parc==2)|(parc==41)]=3
    for h in ('lh','rh'):
        lab_f=os.path.join(S,'repro30_labels','%s.%s.npz'%(scan,h))
        if not os.path.exists(lab_f): continue
        ids=np.load(lab_f)['column-raw']
        w,wf=fsio.read_geometry(os.path.join(C,'field_pial_sigma0.65','%s.white'%h))
        V=np.asarray(w,float); F=np.asarray(wf,int)
        pial,path=pc.propagate_pial(V,F,vel,segv,tovox,totkr,return_path=True)
        path=path.astype(np.float64)
        _m,Wm,deg=_mesh_adjacency(V,F)
        vr,s,_i=fp.retract_self_intersections(path,F,Wm=Wm,max_move=CAP)
        vs,s2,_j=fp.smooth_retraction(path,s,F,Wm,deg,max_move=CAP)
        th0=np.linalg.norm(pial-V,axis=1); th1=np.linalg.norm(vs-V,axis=1)
        for pid in np.unique(ids):
            if pid<=1000: continue
            m=ids==pid
            if m.sum()<50: continue
            nm=inv.get(int(pid),str(pid)).split('-',1)[1]
            acc[nm].append((float(th0[m].mean()),float(th1[m].mean()),
                            float((th1[m]-th0[m]).mean()),
                            100*float((th1[m]-th0[m]).mean()/max(th0[m].mean(),1e-9)),
                            float((th1[m]<th0[m]-0.5).mean()*100)))
        print('%s.%s done'%(scan[:18],h),flush=True)
rows=[]
for nm,v in acc.items():
    a=np.array(v)
    rows.append((np.median(a[:,3]),np.median(a[:,2]),np.median(a[:,0]),np.median(a[:,4]),nm,len(v)))
rows.sort()
print('\n%-26s %9s %9s %9s %9s'%('parcel','dThick mm','dThick %','thick mm','vtx>0.5mm%'))
for r in rows[:10]:
    print('%-26s %+9.4f %+8.2f%% %9.3f %8.2f%%'%(r[4],r[1],r[0],r[2],r[3]))
print('   ... %d parcels ...'%(len(rows)-20))
for r in rows[-10:]:
    print('%-26s %+9.4f %+8.2f%% %9.3f %8.2f%%'%(r[4],r[1],r[0],r[2],r[3]))
a=np.array([r[0] for r in rows]); b=np.array([r[1] for r in rows])
print('\nacross %d parcels: dThick median %+.4f mm (%+.2f%%), range %+.4f .. %+.4f mm'
      %(len(rows),np.median(b),np.median(a),b.min(),b.max()))
