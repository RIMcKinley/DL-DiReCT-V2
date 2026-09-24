"""The model's parcel posteriors, small enough to carry in memory.

The ribbon parcellation needs the model's per-parcel probabilities at the
ribbon voxels. Routing those through disk costs, measured on this hardware,
21.4 s to write the 94 volumes and 33.4 s to read them back -- about 55 s of
the ~76 s the whole parcellation adds to a case, against ~16 s of actual
computation. On a 840-case batch that is ~13 hours of I/O and 320 GB written.

Carrying them instead is cheap IF the buffer is shaped properly:

    full 94-class posterior, float32                       ~940 MB
    float16                                                ~470 MB
    cortical classes only, at voxels whose argmax is cortex  ~55 MB

The last is what this class holds, and the restriction needs no surfaces: the
argmax is available the moment the model runs, so 95% of the buffer can be
dropped immediately and ~55 MB carried through to the surface stage. The
softmax denominator is not lost -- the mass the model put on everything that
is not a parcel of this hemisphere is summed into one number per voxel, which
is what the null rule and the vote gate need.

`save`/`load` exist because holding it in memory means the segmentation and
the surface stage share a process: a failure in the latter throws away the
former, which the two-process arrangement survives. They are also what makes
iterating on the labelling cheap -- a labelling experiment should never have
to re-run the model.
"""

import glob
import os

import numpy as np
import nibabel as nib

META_CLASSES = ('Left-Cerebral-Cortex', 'Right-Cerebral-Cortex',
                'left-hemisphere', 'right-hemishpere', 'brain')


class CorticalPosterior(object):
    """Per-parcel probabilities at the voxels the model calls cortex.

    coords   (N, 3) int32 voxel indices, C order
    names    the K parcel names, in the order of `probs` columns
    probs    (N, K) float16, the parcel probabilities
    other    (N,)  float16, the mass on everything else (WM, hippocampus, the
             other hemisphere, ...). probs.sum(1) + other == 1 to rounding.
    rival    (N,)  float16, the LARGEST SINGLE non-parcel probability. Kept
             separately because the vote gate asks whether the model's argmax
             over the real classes is a parcel -- a question the summed mass
             cannot answer, since a total spread over thirty classes can
             exceed any one parcel without any of them winning.
    shape    the volume the coordinates index
    """

    __slots__ = ('coords', 'names', 'probs', 'other', 'rival', 'shape')

    def __init__(self, coords, names, probs, other, shape, rival=None):
        self.coords = np.asarray(coords, np.int32)
        self.names = list(names)
        self.probs = np.asarray(probs, np.float16)
        self.other = np.asarray(other, np.float16)
        self.rival = (np.zeros(len(self.coords), np.float16) if rival is None
                      else np.asarray(rival, np.float16))
        self.shape = tuple(int(x) for x in shape)

    def __len__(self):
        return len(self.coords)

    @property
    def nbytes(self):
        return int(self.probs.nbytes + self.other.nbytes + self.rival.nbytes
                   + self.coords.nbytes)

    def mask(self):
        """Boolean volume of the voxels held."""
        m = np.zeros(self.shape, bool)
        m[self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]] = True
        return m

    def restrict(self, names):
        """A view holding only `names`; the dropped mass moves into `other`."""
        keep = [self.names.index(n) for n in names]
        drop = [k for k in range(len(self.names)) if k not in set(keep)]
        dropped = self.probs[:, drop].astype(np.float32)
        other = self.other.astype(np.float32) + dropped.sum(1)
        rival = np.maximum(self.rival.astype(np.float32),
                           dropped.max(1) if dropped.shape[1] else 0.0)
        return CorticalPosterior(self.coords, names, self.probs[:, keep],
                                 other, self.shape, rival=rival)

    def dense(self, names):
        """(N, K+1) float32 for `names`, with the non-parcel mass last.

        This is the layout mesh_crf.ribbon_unary reads: a softmax over the real
        classes, restricted to the ones asked for, with the remainder kept so
        the null rule still sees it.
        """
        sub = self.restrict(names)
        out = np.empty((len(sub), len(names) + 1), np.float32)
        out[:, :-1] = sub.probs.astype(np.float32)
        out[:, -1] = sub.rival.astype(np.float32)     # the gate's competitor
        return out

    def rows_for(self, coords, names):
        """(N, K+1) float32 aligned to `coords`, parcels then the rest.

        A voxel the posterior does not hold -- the model's argmax there was not
        cortex, so it was dropped at construction -- comes back as all mass on
        the last column, which is exactly what it means: nothing cortical here.
        """
        want = np.asarray(coords, np.int64)
        if want.ndim != 2 or want.shape[1] != 3:
            raise ValueError('coords must be (N, 3), got %s' % (want.shape,))
        key = (want[:, 0] * self.shape[1] + want[:, 1]) * self.shape[2] + want[:, 2]
        have = (self.coords[:, 0].astype(np.int64) * self.shape[1]
                + self.coords[:, 1]) * self.shape[2] + self.coords[:, 2]
        order = np.argsort(have)
        pos = np.searchsorted(have[order], key)
        pos = np.clip(pos, 0, len(order) - 1)
        row = order[pos]
        hit = have[row] == key
        sub = self.restrict(names)
        out = np.zeros((len(want), len(names) + 1), np.float32)
        out[:, -1] = 1.0                              # not held -> nothing cortical
        out[hit, :-1] = sub.probs[row[hit]].astype(np.float32)
        out[hit, -1] = sub.rival[row[hit]].astype(np.float32)
        return out

    def save(self, path):
        np.savez_compressed(path, coords=self.coords, probs=self.probs,
                            other=self.other, rival=self.rival,
                            names=np.array(self.names), shape=np.array(self.shape))
        return path

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d['coords'], [str(x) for x in d['names']], d['probs'],
                   d['other'], tuple(int(x) for x in d['shape']),
                   rival=d['rival'] if 'rival' in d else None)

    @classmethod
    def from_logits(cls, logits, names, cortical=None, keep=None):
        """Build from the model's raw output, [C, D, H, W] or [D, H, W, C].

        `cortical` names the classes to keep (default: every name that looks
        like a parcel, i.e. starts with lh- or rh-). `keep` optionally gives
        the voxel mask directly; the default keeps voxels whose argmax over the
        real classes is one of `cortical`, which is the cheapest honest
        restriction and needs nothing but the model output.
        """
        a = np.asarray(logits)
        if a.ndim != 4:
            raise ValueError('logits must be 4-D, got %s' % (a.shape,))
        if a.shape[0] == len(names):
            a = np.moveaxis(a, 0, -1)
        if a.shape[-1] != len(names):
            raise ValueError('logits has %d channels for %d names'
                             % (a.shape[-1], len(names)))
        real = [k for k, n in enumerate(names) if n not in META_CLASSES]
        rn = [names[k] for k in real]
        a = a[..., real]
        if cortical is None:
            cortical = [n for n in rn if n.startswith('lh-') or n.startswith('rh-')]
        ci = [rn.index(n) for n in cortical]
        shape = a.shape[:3]
        flat = a.reshape(-1, a.shape[-1])
        top = flat.argmax(1)
        sel = np.isin(top, np.asarray(ci)) if keep is None else \
            np.asarray(keep).reshape(-1)
        idx = np.where(sel)[0]
        L = flat[idx].astype(np.float32)
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
        coords = np.stack(np.unravel_index(idx, shape), axis=1)
        probs = P[:, ci]
        other = np.maximum(1.0 - probs.sum(1), 0.0)
        rest = np.ones(P.shape[1], bool)
        rest[ci] = False
        rival = P[:, rest].max(1) if rest.any() else np.zeros(len(P), np.float32)
        return cls(coords, cortical, probs, other, shape, rival=rival)

    @classmethod
    def from_dir(cls, prep_dir, cortical=None, keep=None):
        """Build from seg_<Label>.nii.gz on disk -- the slow route, kept as the
        fallback and for reproducing a run whose posteriors were dumped."""
        files = sorted(glob.glob(os.path.join(prep_dir, 'seg_*.nii.gz')))
        if not files:
            raise ValueError('no seg_<Label>.nii.gz in %s' % prep_dir)
        names = [os.path.basename(f)[4:-7] for f in files]
        first = nib.load(files[0])
        shape = tuple(first.shape[:3])
        vol = np.empty((len(files),) + shape, np.float32)
        for k, f in enumerate(files):
            vol[k] = np.asarray(nib.load(f).dataobj, dtype=np.float32)
        return cls.from_logits(vol, names, cortical=cortical, keep=keep)
