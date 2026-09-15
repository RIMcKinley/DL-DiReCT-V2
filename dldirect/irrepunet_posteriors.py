"""Monkey patch: recover irrepunet-seg's tissue posteriors, which it never writes.

WHY A PATCH AND NOT A FLAG

`irrepunet-seg` emits hard labels only. `inference.py` has a `save_probabilities`
parameter, but `validate.py:451` says plainly that it "is not supported for
lr-dualpass" -- and the dual pass is the production path the CLI always uses.

The reason is in the hybrid cortex relabel a few lines above it. For every
cortical voxel the dual pass does

    nprob[:, ci, cj, ck] = 0.0
    nprob[target[ci, cj, ck], ci, cj, ck] = 1.0

i.e. it OVERWRITES the posterior with a one-hot, taking the parcel identity from
the 1 mm whole-brain pass and the shape from the 0.7 mm pass. By the time the
final label map exists the cortical posteriors are gone by construction, so
there is no "posterior of the shipped segmentation" to extract after that point.

WHERE THIS TAPS

The hybrid block opens with `from seg_label_map import get_scheme as _gs`, which
is called AFTER the L/R fold and BEFORE the one-hot. Wrapping `get_scheme` and
reading `nprob` out of the calling frame therefore catches the last moment at
which a true posterior exists, without altering a single value the pipeline goes
on to use -- the wrapper delegates to the real function and changes nothing.

So the posteriors this recovers are the FOLDED NATIVE-PASS posteriors: correct
probabilities from the 0.7 mm tissue pass with the L/R mass moved to the right
homologue, but BEFORE the cortex parcel identities are replaced from the 1 mm
pass. For the tissue question this pipeline actually asks -- is this voxel
cortex, is it white matter -- that is the quantity you want; the hybrid only
reassigns WHICH parcel a cortical voxel belongs to, not whether it is cortex.

SUM, NOT MAX. These channels are a softmax over 95 classes, so P(cortex) is the
SUM over the 68 parcel channels, not the max. (`load_gm_wm_probability` takes a
max because DeepSCAN's per-label sigmoids are independent and do not partition.)

USAGE

    IRREPUNET_POSTERIOR_DIR=/somewhere \
    PYTHONPATH=/path/to/dldirect:$PYTHONPATH \
        irrepunet-seg --input t1.nii.gz --output seg.nii.gz --skull-stripped

`irrepunet-seg` runs the real work as a SUBPROCESS, forwarding PYTHONPATH, so
this file has to be importable as `sitecustomize` for the interpreter to pick it
up automatically. `install_sitecustomize(dir)` writes that shim for you; the
`__main__` block below does it.

Writes <dir>/prob_gm.nii.gz and <dir>/prob_wm.nii.gz in the INPUT's own space.

Getting there needs the same two steps `save_prediction_nifti` applies to the
label map, because run_inference_lr2pass works in a cropped RAS space, not the
input's: uncrop into `props['original_shape']` using `props['bbox']`, then
reverse the RAS reorientation with `nibabel.orientations`. Skipping that leaves
the axes permuted -- measured here as (127, 159, 124) against the emitted
segmentation's (127, 124, 159). `props` is picked up from the calling stack
alongside `nprob`. If it cannot be found, the raw cropped-RAS arrays are written
to posteriors.npz instead, so nothing is silently mis-oriented.
"""

import os
import sys

import numpy as np

ENV_DIR = 'IRREPUNET_POSTERIOR_DIR'

# channel names that make up each tissue, in seg_label_map's scheme
_WM_NAMES = ('Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter',
             'Corpus-Callosum')


def _find_frame_with(name, limit=12):
    """The nearest calling frame that has `name` in its locals."""
    f = sys._getframe(1)
    for _ in range(limit):
        if f is None:
            return None
        if name in f.f_locals:
            return f
        f = f.f_back
    return None


def _to_input_space(vol, props):
    """Cropped RAS -> the input's own grid, exactly as save_prediction_nifti does."""
    import nibabel as nib
    from nibabel import orientations as nib_orient
    full_ras = np.zeros(props['original_shape'], np.float32)
    full_ras[tuple(slice(b[0], b[1]) for b in props['bbox'])] = vol
    aff = props['original_affine']
    rev = nib_orient.ornt_transform(nib_orient.axcodes2ornt(('R', 'A', 'S')),
                                    nib.io_orientation(aff))
    return nib_orient.apply_orientation(full_ras, rev).astype(np.float32), aff


def _capture(scheme_map):
    """Read nprob out of the hybrid's calling frame and write the tissue sums."""
    out_dir = os.environ.get(ENV_DIR)
    if not out_dir:
        return
    frame = _find_frame_with('nprob')
    if frame is None:
        return                      # not the dual-pass call site; nothing to do
    loc = frame.f_locals
    nprob = loc.get('nprob')
    if nprob is None or getattr(nprob, 'ndim', 0) != 4:
        return

    gm_ch = [i for i, (n, _) in scheme_map.items() if n.startswith(('lh-', 'rh-'))]
    wm_ch = [i for i, (n, _) in scheme_map.items() if n in _WM_NAMES]
    if not gm_ch or not wm_ch:
        return

    # marginal over a softmax partition -> sum
    gm = np.asarray(nprob[gm_ch]).sum(0).astype(np.float32)
    wm = np.asarray(nprob[wm_ch]).sum(0).astype(np.float32)

    csl = loc.get('csl')
    full = loc.get('full')
    if csl is None or full is None:
        return
    sl = tuple(csl)
    g = np.zeros(tuple(full), np.float32); g[sl] = gm
    w = np.zeros(tuple(full), np.float32); w[sl] = wm

    os.makedirs(out_dir, exist_ok=True)
    pf = _find_frame_with('props', limit=20)
    props = pf.f_locals.get('props') if pf is not None else None
    if not (isinstance(props, dict) and 'original_affine' in props):
        path = os.path.join(out_dir, 'posteriors.npz')
        np.savez_compressed(path, gm=g, wm=w,
                            csl=np.array([[c.start, c.stop] for c in sl]))
        print('[irrepunet-posteriors] no props on the stack; wrote CROPPED-RAS '
              'arrays to %s (NOT in the input grid)' % path, flush=True)
        return

    import nibabel as nib
    for nm, vol in (('prob_gm', g), ('prob_wm', w)):
        out, aff = _to_input_space(vol, props)
        path = os.path.join(out_dir, '%s.nii.gz' % nm)
        nib.save(nib.Nifti1Image(out, aff), path)
        print('[irrepunet-posteriors] wrote %s  shape %s  >0.5 %d  (pre-hybrid)'
              % (path, out.shape, int((out > 0.5).sum())), flush=True)


def install():
    """Wrap seg_label_map.get_scheme. Safe to call more than once."""
    try:
        import seg_label_map
    except Exception:
        return False                # not an irrepunet interpreter; stay out of the way
    real = getattr(seg_label_map, 'get_scheme', None)
    if real is None or getattr(real, '_posterior_patched', False):
        return False

    def get_scheme(*a, **kw):
        result = real(*a, **kw)
        try:
            _capture(result)
        except Exception as exc:    # never break the segmentation for a debug dump
            print('[irrepunet-posteriors] capture skipped (%s: %s)'
                  % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        return result

    get_scheme._posterior_patched = True
    get_scheme.__doc__ = real.__doc__
    seg_label_map.get_scheme = get_scheme
    return True


def install_sitecustomize(target_dir):
    """Write a sitecustomize.py in `target_dir` that calls install() at startup.

    Put `target_dir` on PYTHONPATH before invoking irrepunet-seg; the interpreter
    imports `sitecustomize` automatically, and irrepunet-seg forwards PYTHONPATH
    to the subprocess that does the real work.
    """
    os.makedirs(target_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(target_dir, 'sitecustomize.py')
    with open(path, 'w') as fh:
        fh.write(
            '# generated by dldirect.irrepunet_posteriors -- see that module\n'
            'import sys\n'
            'sys.path.insert(0, %r)\n'
            'try:\n'
            '    from dldirect.irrepunet_posteriors import install\n'
            '    install()\n'
            'except Exception:\n'
            '    pass\n' % os.path.dirname(here))
    return path


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(install_sitecustomize(d))
