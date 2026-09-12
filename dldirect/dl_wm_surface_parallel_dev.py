# Reconstruction of a topollogically correct WM surface
# based on DL+DiReCT segmentation
#
# DeepSCAN: a deep learning-based neuroanatomy segmentation and cortex parcellation
# DiReCT: A Diffeomorphic registration based cortical thickness
#
# For instructions to produce a Segmentation: https://github.com/SCAN-NRAD/DL-DiReCT-V2
#
# this code produces for each hemisphere: The white matter surface
#
# Victor B. B. Mello, 09/2024
# Support Center for Advanced Neuroimaging (SCAN)
# University Institute of Diagnostic and Interventional Neuroradiology
# University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.
#
# The geometry now lives in dldirect/wm_surface.py so that the pial pipeline can
# build the same surface in-process instead of reading it back off disk. This
# file is the CLI and the file writing; the mesh it writes is unchanged.

import argparse
import nibabel as nib
import numpy as np
from surface_frames import volume_info_from_prep
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dldirect import wm_surface

# For FS visualization
get_vox2ras_tkr = wm_surface.get_vox2ras_tkr
rec_surf = wm_surface.rec_surf


def write_region(out_dir, region, verts, faces, vol_info):
    # FreeSurfer binary files
    # Visualization together with T1w_norm.nii.gz
    # A surface file should state its own frame: FreeSurfer's carry the volume
    # geometry (dimensions, voxel size, direction cosines, cras) so a reader can
    # place them without outside knowledge. Writing volume_info=None is why
    # freeview reported "Did not find any volume info" for these.
    for name in ('white', 'orig', 'white.preaparc'):
        nib.freesurfer.io.write_geometry(os.path.join(out_dir, 'surf', '%s.%s' % (region, name)),
                                         verts, faces, create_stamp=None, volume_info=vol_info)

    # create annotation with unknown labels
    # will be changed in the future
    names = [b'unknown']
    ctab = np.array([[25, 5, 25, 0,  1639705]], dtype=np.int32)
    annot = np.array( np.zeros(len(verts)), dtype = np.int32)
    nib.freesurfer.io.write_annot(os.path.join(out_dir, 'label', '%s.aparc.annot' % region),
                                  annot, ctab, names, fill_ctab=True)


def main():
    # Parser for shell script
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputpath', '--input')
    parser.add_argument('-outputpath', '--output')
    parser.add_argument('-ns', '--nsmooth')
    args = parser.parse_args()
    nsmooth = int(args.nsmooth)

    # reconstruct different ROIs
    surfaces = wm_surface.build_white_surfaces(args.input, regions=('lh', 'rh'),
                                               nsmooth=nsmooth)
    vol_info = volume_info_from_prep(args.input)
    for region, (verts, faces) in surfaces.items():
        write_region(args.output, region, verts, faces, vol_info)


if __name__ == '__main__':
    main()
