"""Tube stamping vs the graph walks, over many hemispheres.

Everything runs off the saved labellings and the stored velocity fields, so no
model and no logits. Four propagations per hemisphere from the SAME seeds:

    plain      26-connected Dijkstra over grey-matter voxels
    bank-cut   the same, with edges cut where the owning vertices' normals
               oppose -- the surface telling the graph where the sulcus is
    tube       stamp each column's tube, gaps filled along the bank-cut graph
    (agreement is against the model's own voxel parcellation, which is not
     truth but is what the labels are carried from)
"""
import os,sys,glob,numpy as np,nibabel as nib
from scipy import spatial
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import map_coordinates
sys.path.insert(0,'/data/disk2/projects/DL-DiReCT-V2')
from dldirect.ribbon_labels import _ribbon_graph, seed_voxels
from dldirect.field_pial_prototype import make_transforms
from dldirect import pial_clean as pc
from dldirect import regional_stats as rs
S=os.path.dirname(os.path.abspath(__file__)); OAS='/data/disk2/oasis840'
fsio=nib.freesurfer.io
SUB=4; RADIUS=0.87; ARM='column-raw'
OFF=np.array([(a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)])
def vnorm(V,F):
    n=np.zeros_like(V); fn=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    for k in range(3): np.add.at(n,F[:,k],fn)
    return n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
files=sorted(glob.glob(os.path.join(S,'repro30_labels','*_run-01.*.npz')))[:12]
print('%-30s %8s %8s %8s %8s'%('hemisphere','plain','bank-cut','tube','unstamp%'),flush=True)
res=[]
for f in files:
    base=os.path.basename(f)[:-4]; scan,h=base.rsplit('.',1)
    C=os.path.join(OAS,scan)
    vp=os.path.join(C,'field_pial_sigma0.65','pial_Velocity.nii.gz')
    if not os.path.exists(vp): continue
    ref=nib.load(os.path.join(C,'mri','aparc.atlas+aseg.nii.gz'))
    parc=np.asarray(ref.dataobj).astype(np.int32)
    spn=tuple(float(x) for x in ref.header.get_zooms()[:3]); tovox,totkr=make_transforms(ref)
    vel=np.asarray(nib.load(vp).dataobj)
    lo,hi=(1000,1036) if h=='lh' else (2000,2036)
    gm=(parc>lo)&(parc<hi); shape=gm.shape
    w,wf=fsio.read_geometry(os.path.join(C,'field_pial_sigma0.65','%s.white'%h))
    V=np.asarray(w,float); F=np.asarray(wf,int); N=vnorm(V,F); wv=tovox(V)
    ids=np.load(f)[ARM]; ok=np.where(ids>0)[0]
    G,idx,coords=_ribbon_graph(gm,spn)
    own=spatial.cKDTree(V).query(totkr(coords.astype(float)))[1]
    Gc=G.tocoo(); kp=(N[own[Gc.row]]*N[own[Gc.col]]).sum(1)>0.0
    Gk=sp.coo_matrix((Gc.data[kp],(Gc.row[kp],Gc.col[kp])),shape=G.shape).tocsr()
    seeds,_s,_d=seed_voxels(wv[ok],ids[ok],gm,idx,2)
    nodes=np.fromiter(seeds.keys(),np.int64); slab=np.fromiter((seeds[x] for x in nodes),np.int64)
    walk={}
    for tag,graph in (('plain',G),('bank-cut',Gk)):
        _dist,_p,src=dijkstra(graph,directed=False,indices=nodes,min_only=True,
                              return_predecessors=True)
        look=np.full(graph.shape[0],-1,np.int64); look[nodes]=slab
        lb=np.where(src>=0,look[np.clip(src,0,None)],-1)
        v=np.zeros(shape,np.int32); v[coords[:,0],coords[:,1],coords[:,2]]=np.where(lb>0,lb,0)
        walk[tag]=v
    cur=V[ok].copy(); path=[cur.copy()]
    for _ in range(pc.ROUNDS):
        pos=tovox(cur)
        vv=np.stack([map_coordinates(vel[...,k],pos.T,order=1,mode='nearest') for k in range(3)],axis=1)
        cur=cur+(totkr(pos-vv)-totkr(pos))*pc.STEP_SCALE; path.append(cur.copy())
    path=np.stack(path,0)
    lab=np.zeros(shape,np.int32); best=np.full(shape,np.inf,np.float32)
    for r in range(path.shape[0]-1):
        for s in range(SUB):
            t=s/float(SUB); q=tovox(path[r]*(1-t)+path[r+1]*t); bs=np.floor(q).astype(int)
            for o in OFF:
                qi=bs+o
                good=np.all((qi>=0)&(qi<np.array(shape)),axis=1)
                qq=qi[good]; dd=np.linalg.norm(q[good]-qq,axis=1); lb=ids[ok][good]
                inr=gm[qq[:,0],qq[:,1],qq[:,2]]&(dd<=RADIUS)
                qq,dd,lb=qq[inr],dd[inr],lb[inr]
                if not len(qq): continue
                o2=np.argsort(-dd); qq,dd,lb=qq[o2],dd[o2],lb[o2]
                tk=dd<best[qq[:,0],qq[:,1],qq[:,2]]
                qq,dd,lb=qq[tk],dd[tk],lb[tk]
                best[qq[:,0],qq[:,1],qq[:,2]]=dd; lab[qq[:,0],qq[:,1],qq[:,2]]=lb
    unst=100*float(((gm)&(lab==0)).sum())/gm.sum()
    miss=gm&(lab==0)
    if miss.any():
        srcn=np.where(lab[coords[:,0],coords[:,1],coords[:,2]]>0)[0]
        _d2,_p2,s2=dijkstra(Gk,directed=False,indices=srcn,min_only=True,return_predecessors=True)
        look=np.full(Gk.shape[0],-1,np.int64)
        look[srcn]=lab[coords[srcn,0],coords[srcn,1],coords[srcn,2]]
        fill=np.where(s2>=0,look[np.clip(s2,0,None)],0)
        mm=lab[coords[:,0],coords[:,1],coords[:,2]]==0
        lab[coords[mm,0],coords[mm,1],coords[mm,2]]=np.maximum(fill[mm],0)
    row=[100*float(((walk['plain']!=parc)&gm).sum())/gm.sum(),
         100*float(((walk['bank-cut']!=parc)&gm).sum())/gm.sum(),
         100*float(((lab!=parc)&gm).sum())/gm.sum(),unst]
    res.append(row)
    print('%-30s %7.2f%% %7.2f%% %7.2f%% %7.2f%%'%(base,*row),flush=True)
a=np.array(res)
print('\n%-30s %7.2f%% %7.2f%% %7.2f%% %7.2f%%   (median of %d)'
      %('MEDIAN',*np.median(a,0),len(a)))
print('tube better than bank-cut in %d/%d hemispheres'%(int((a[:,2]<a[:,1]).sum()),len(a)))
