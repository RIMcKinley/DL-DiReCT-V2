"""Express a surface built on one grid in another grid's tkrRAS frame.

The MESH is untouched -- same vertices, same faces, same order. Only the
coordinates are mapped, through scanner RAS:

    world = A_src @ inv(vox2ras_tkr(src_ref)) @ v_src
    v_dst = vox2ras_tkr(dst_ref) @ inv(A_dst) @ world

Nothing is resampled or interpolated: this is an exact affine on vertex
coordinates, so vertex i still means the same anatomical point afterwards.

The point of it is to let two packages that require different grids start from
ONE white surface. fast_surface_reconstruction.sh needs the 256^3 conform for
mri_normalize / mri_pretess / mris_make_surfaces; field_pial_prototype works on
the cropped grid the segmentation and velocity field live on. Handing the
prototype the conform-built white through this mapping makes the two pial
surfaces vertex-for-vertex comparable.

The destination frame comes from the grid the DESTINATION pipeline actually
uses -- for field_pial_prototype that is seg_<Label>.nii.gz, which is what
load_gm_wm_probability returns -- and the written surface carries matching
volume_info so any reader places it correctly.
"""
import os
import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio

from .compare_surfaces import tkr_to_world
from .surface_frames import volume_info_from_image


def tkr_to_tkr(src_prep, dst_ref):
    """Affine taking tkr coordinates of `src_prep`'s grid into `dst_ref`'s."""
    A_src = np.loadtxt(os.path.join(src_prep, 'mri', 'conform_vox2ras.txt'))
    ref_src = nib.load(os.path.join(src_prep, 'mri', 'aparc.atlas+aseg.nii.gz'))
    return (np.linalg.inv(tkr_to_world(dst_ref, dst_ref.affine))
            @ tkr_to_world(ref_src, A_src))


def retarget(src_prep, dst_ref, surf_path, out_path, dst_name=None):
    """Rewrite one surface into `dst_ref`'s frame, with matching volume_info."""
    M = tkr_to_tkr(src_prep, dst_ref)
    v, f = fsio.read_geometry(surf_path)
    vd = (M[:3, :3] @ v.T).T + M[:3, 3]
    fsio.write_geometry(out_path, vd, f, create_stamp=None,
                        volume_info=volume_info_from_image(dst_ref, dst_name or ''))
    return len(v), M


def main():
    import argparse
    from .field_pial_prototype import load_gm_wm_probability
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src-prep', required=True,
                    help='prep the surfaces were built in (needs mri/conform_vox2ras.txt '
                         'and mri/aparc.atlas+aseg.nii.gz)')
    p.add_argument('--dst-prep', required=True,
                    help="prep whose grid the surfaces should be expressed in; its "
                         "seg_<Label>.nii.gz is the reference, matching what "
                         "field_pial_prototype uses")
    p.add_argument('--out-dir', required=True)
    p.add_argument('--surf', nargs='+', default=['white'],
                    help='surface basenames to map (default: white)')
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--rename', default=None,
                    help='write as <hemi>.<rename> instead of <hemi>.<surf>; only valid '
                         'with a single --surf')
    args = p.parse_args()
    if args.rename and len(args.surf) != 1:
        p.error('--rename takes a single --surf')
    os.makedirs(args.out_dir, exist_ok=True)
    _gm, _wm, dst_ref = load_gm_wm_probability(args.dst_prep)
    for surf in args.surf:
        for hemi in args.hemi:
            n, M = retarget(args.src_prep, dst_ref,
                            os.path.join(args.src_prep, 'surf', '%s.%s' % (hemi, surf)),
                            os.path.join(args.out_dir, '%s.%s'
                                         % (hemi, args.rename or surf)),
                            dst_name=args.dst_prep)
            print('%s.%s: %d verts, tkr shift %s mm'
                  % (hemi, surf, n, np.round(M[:3, 3], 3)))


if __name__ == '__main__':
    main()
