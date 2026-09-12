"""Build the topologically-correct, smoothed WM surface from a segmentation.

This is the mesh half of dl_wm_surface_parallel_dev.py, factored out so the
surface can be built in-process -- pial_pipeline.reconstruct() calls it when no
white surface is supplied, which removes the round trip through ?h.white on
disk. The script remains the CLI and keeps the file writing; both go through
build_hemisphere() here, so there is one implementation of the geometry.

Original: Victor B. B. Mello, 09/2024, SCAN / University of Bern.
"""

import numpy as np
import nibabel as nib
import pymeshlab

from . import wm_labels

NSMOOTH_DEFAULT = 50          # Taubin steps; the shell pipelines pass -ns 50


def get_vox2ras_tkr(t1):
    """Transformation for FreeView visualisation.

    Derived from the header's pixdim/dim rather than the affine, so it is the
    tkrRAS of whatever grid the segmentation was written on.
    """
    ds = t1.header._structarr['pixdim'][1:4]
    ns = t1.header._structarr['dim'][1:4] * ds / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                       [0, 0, ds[2], -ns[2]],
                       [0, -ds[1], 0, ns[1]],
                       [0, 0, 0, 1]], dtype=np.float32)

    return v2rtkr


def rec_surf(binary, affine, r):
    # Reconstruct topologically correct surfaces
    # https://nighres.readthedocs.io/en/latest/shape/topology_correction.html
    # https://nighres.readthedocs.io/en/latest/surface/levelset_to_mesh.html
    # Ref: Bazin and Pham (2007). Topology correction of segmented medical images using a fast marching algorithm doi:10.1016/j.cmpb.2007.08.006
    # nighres reads only dims and zooms from this image and returns voxel
    # coordinates, so it does not need the 256^3 conform; this runs unchanged on
    # the cropped grid (preparedata.py --space cropped). 'background->object'
    # propagation does need background around the object, which the crop
    # (the brain mask's bounding box) always leaves: measured >= 4 voxels of
    # margin on every face for both hemispheres. Guard it anyway with an
    # integer pad that is subtracted from the vertices -- exact, no resampling.
    # Note the correction is NOT invariant to how much zero padding surrounds
    # the object (it is deterministic for a given array): on sub-POBHC0002 rh
    # the 256^3 conform and the bare crop disagree on 35 of 264k object voxels,
    # all boundary voxels along the medial side, which moves ~260 of 145k
    # vertices by more than 0.1mm (mean point-to-surface 0.0015mm). So pad only
    # when the object actually touches a face.
    import nighres                       # heavy; only needed to build a surface
    pad = 0
    touches = any(b[0].any() or b[-1].any() for b in
                  (binary, np.moveaxis(binary, 1, 0), np.moveaxis(binary, 2, 0)))
    if touches:
        pad = 2
        print('WARNING: %s WM mask touches the volume border; padding by %d voxels for topology correction' % (r, pad))
        binary = np.pad(binary, pad)
    farray_img = nib.Nifti1Image(binary.astype(np.float64), affine)

    propag = 'background->object'
    connect = '6/18'
    minimum_distance = 1e-5

    ret = nighres.shape.topology_correction(farray_img, 'binary_object', minimum_distance=minimum_distance, propagation=propag,connectivity=connect)
    l2m_ret = nighres.surface.levelset_to_mesh(ret['corrected'], connectivity=connect)
    vertices = l2m_ret['result']['points'] - pad
    faces = l2m_ret['result']['faces']

    return vertices, faces


def hemisphere_binary(seg, df_labels, region, excluded):
    """The filled WM mask for one hemisphere.

    Regions based on dl output label_def.csv. The excluded set comes from the
    record preparedata.py wrote for THIS run (wm_labels), not from a second
    copy of the list: the two used to disagree -- filled.mgz dropped the
    hippocampus but this mask kept it, so the surface produced here enclosed
    a structure the pipeline believed it had excluded. Every label that is
    not excluded is deliberately filled: the subcortical structures are what
    make the WM mask simply connected for topology_correction.
    """
    if region == 'lh':
        labels = wm_labels.hemisphere_labels(df_labels, 'Left', excluded)
    elif region == 'rh':
        labels = wm_labels.hemisphere_labels(df_labels, 'Right', excluded)
    else:
        raise ValueError('not a valid region: %r' % (region,))
    mask = np.isin(seg, labels)
    return np.array(np.where(mask, 1, 0), dtype=np.int32)


def build_hemisphere(seg, df_labels, affine, region, excluded, nsmooth=NSMOOTH_DEFAULT):
    """Segmentation -> (vertices, faces) for one hemisphere's white surface.

    Topology correction, marching cubes, the tkrRAS affine, then `nsmooth`
    Taubin steps at pymeshlab's defaults. Vertices come back in the tkrRAS of
    the grid `affine` describes.
    """
    binary = hemisphere_binary(seg, df_labels, region, excluded)
    vertices, faces = rec_surf(binary, affine, region)

    # apply affine for FS visualization and matching with the MRI
    transf_vertx = nib.affines.apply_affine(affine, vertices)
    # get a smooth mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=transf_vertx, face_matrix=faces), region)
    ms.meshing_invert_face_orientation()
    ms.apply_coord_taubin_smoothing(stepsmoothnum=nsmooth)

    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()


def _build_one(job):
    seg, df_labels, affine, region, excluded, nsmooth = job
    return region, build_hemisphere(seg, df_labels, affine, region, excluded, nsmooth)


def load_inputs(prep_dir):
    """(seg, df_labels, affine, excluded) from a prep directory.

    Whatever grid preparedata.py wrote the segmentation on (256^3 conform by
    default, the cropped grid with --space cropped) is the grid the surfaces
    are built on, and the tkrRAS returned is that grid's.
    """
    import os
    import pandas as pd
    seg_img = nib.load(os.path.join(prep_dir, 'mri', 'aparc.atlas+aseg.nii.gz'))
    affine = get_vox2ras_tkr(seg_img)
    seg = seg_img.get_fdata()
    df_labels = pd.read_csv(os.path.join(prep_dir, 'label_def.csv')) \
        .set_index('LABEL').to_dict()
    # The structures preparedata.py left out of the hemisphere fill for this
    # run. Read, not re-specified, so the two masks cannot disagree.
    excluded = wm_labels.read_record(os.path.join(prep_dir, 'mri'))
    return seg, df_labels, affine, excluded


def build_white_surfaces(prep_dir, regions=('lh', 'rh'), nsmooth=NSMOOTH_DEFAULT,
                         parallel=True, verbose=True):
    """{region: (vertices, faces)} built from a prep directory.

    The two hemispheres are independent, so they run in a process pool by
    default -- the topology correction is the expensive part and it is
    single-threaded.
    """
    seg, df_labels, affine, excluded = load_inputs(prep_dir)
    if verbose:
        print('WM fill excludes: %s' % ', '.join(excluded))
    jobs = [(seg, df_labels, affine, r, excluded, nsmooth) for r in regions]
    if parallel and len(jobs) > 1:
        from multiprocessing.pool import Pool
        with Pool(len(jobs)) as pool:
            res = pool.map(_build_one, jobs)
    else:
        res = [_build_one(j) for j in jobs]
    return {r: vf for r, vf in res}
