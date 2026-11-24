"""Large-scale particle image velocimetry (LS-PIV) at the stations of a section.

At each station a small square of water surface, square to the flow, is
sampled on a world-uniform grid -- an ortho-image patch -- from every frame,
through the stabilisation transform exactly as the STI search lines are.  The
centre of the patch from frame ``t - k`` (the interrogation area) is then
cross-correlated with the whole patch from frame ``t`` (the search area); the
correlation peak lies at the displacement of the surface texture between the
two.

Two things make it robust on river footage:

* **Ensemble correlation.**  The correlation planes of every pair are summed
  before the peak is found, rather than one peak being found per pair and the
  peaks averaged.  A surface with sparse or intermittent texture gives a noisy
  plane from any one pair, but the true peak adds up while the noise does not.
* **Background removal.**  The clip's mean patch is subtracted from every
  patch, so static texture -- glare, the bank, a rock -- does not correlate at
  zero displacement and drag the velocity towards zero.

The mean is not known until the last frame, yet the frames are not kept.  The
correlation is bilinear, so the correlation of the mean-subtracted patches
expands into the correlation of the raw ones, which is summed as the frames
stream past, and terms in the running sums of the patches, which are combined
with the mean at the end -- the same numbers as subtracting the mean first,
from a few patch-sized sums per station.  A *running* mean will not do: the
texture moving through the patch blurs into streaks along the flow, the
streaks stay in the running mean, and subtracting them leaves a ridge through
zero displacement that biased the velocity low by several percent.

The ensemble is also kept in sub-ensembles about a second long, merged in
pairs whenever there are too many, whose scatter gives the spread.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import Config, LspivConfig
from .geometry import SectionGeometry, finest_m_per_px, station_boxes
from .sti import sample_bilinear
from .stabilise import transform_points
from .vectors import VectorEstimate, subpixel_peak

log = logging.getLogger(__name__)

#: Interrogation-area size limits, in samples.  Fewer than 16 holds too little
#: texture to correlate; more than 64 costs a lot and gains little, so the
#: sample spacing is coarsened instead.  The automatic size follows the
#: station spacing down to the minimum: a larger area than the spacing smears
#: the steep gradient by the banks, where the faster, less-sheared water then
#: dominates the correlation -- 0.19 m/s too fast beside the bank of the demo
#: river at 24 samples, 0.14 at 17, with the interior error a third smaller.
_MIN_IA_SAMPLES = 16
_AUTO_MIN_IA_SAMPLES = 16
_MAX_IA_SAMPLES = 64
#: Extra samples of search margin beyond the fastest expected displacement.
_MARGIN_PAD = 2
#: An automatic pair step keeps the fastest displacement within this fraction
#: of the interrogation area.  The usual quarter-area rule is for correlating
#: two windows of the same size, which lose the texture that moves out of
#: them; here the search area reaches a margin beyond the interrogation area
#: on every side, so nothing is lost, and the larger step resolves slow water
#: -- which moves a fraction of a sample per frame -- far better.
_AUTO_PAIR_FRACTION = 1.0
#: Length of a sub-ensemble, seconds, and the most kept per station: past
#: that, neighbours are merged in pairs and later ones made twice as long.
_BLOCK_S = 1.0
_MAX_BLOCKS = 16
#: The main peak's region, for the peak ratio: where the plane stays above
#: this fraction of the peak, joined to it.
_PEAK_REGION_FRACTION = 0.5
#: Reported peak ratio when nothing else in the plane is positive.
_PEAK_RATIO_CAP = 99.0


@dataclass
class PatchLayout:
    """Where one station's patch is sampled.

    The grid is ``n x n`` samples, row ``iy`` across the flow and column
    ``ix`` along it, ``gsd_m`` apart and centred on the station.  The
    interrogation area is its central ``ia_n x ia_n``, leaving ``margin_n``
    samples of search on every side.
    """

    index: int
    centre_world: np.ndarray
    along_world: np.ndarray
    across_world: np.ndarray
    gsd_m: float
    ia_n: int
    margin_n: int
    pts_world: np.ndarray
    pts_px: np.ndarray

    @property
    def n(self) -> int:
        return self.ia_n + 2 * self.margin_n

    @property
    def ia_size_m(self) -> float:
        return self.ia_n * self.gsd_m

    def corners_world(self, ia_only: bool = False) -> np.ndarray:
        h = 0.5 * (self.ia_n if ia_only else self.n) * self.gsd_m
        a, c = self.along_world * h, self.across_world * h
        return np.array([self.centre_world + s * a + t * c
                         for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


@dataclass
class CorrelationResult:
    """The ensemble correlation of one station, as collected."""

    layout: PatchLayout
    #: Normalised ensemble correlation, ``(2 * margin_n + 1)`` square, with
    #: zero displacement at its centre: 1 where the mean-subtracted
    #: interrogation areas match their displaced search windows exactly.
    #: All zero if no pair was correlated.
    plane: np.ndarray
    #: Peak displacement of each sub-ensemble, ``(n, 2)`` as ``(along,
    #: across)`` samples.
    block_disp: np.ndarray
    n_pairs: int
    pair_step: int
    st_s_per_px: float
    out_of_frame_fraction: float


def _odd(n: float) -> int:
    k = int(round(n))
    return k if k % 2 else k + 1


def plan_patches(
    cfg: Config, geom: SectionGeometry, st_s_per_px: float
) -> tuple[list[PatchLayout], int]:
    """Lay out every station's patch and choose the pair step.

    Returns the layouts and the pair step (frames after ``video.frame_step``).
    """
    lp: LspivConfig = cfg.lspiv
    rect = geom.rectifier
    boxes = station_boxes(geom, cfg.stiv.search_line_length_m)
    if not boxes:
        return [], 1
    along, across = boxes[0].along_world, boxes[0].across_world
    v_lo, v_hi = cfg.stiv.velocity_range_ms
    v_fast = max(abs(v_lo), abs(v_hi))

    # One sample spacing for the whole section: the finest image pixel at any
    # station, so no station is undersampled.  An oblique view's far field is
    # then oversampled, which costs time but not accuracy.
    if lp.gsd_m is not None:
        gsd = float(lp.gsd_m)
    else:
        probe = 0.25 * cfg.stiv.search_line_length_m
        gsd = min(finest_m_per_px(rect, b.centre_world, along, across, probe) for b in boxes)
    if lp.ia_size_m is not None:
        ia_m = float(lp.ia_size_m)
    else:
        ia_m = float(np.clip(geom.station_spacing_m, _AUTO_MIN_IA_SAMPLES * gsd,
                             max(cfg.stiv.search_line_length_m, _AUTO_MIN_IA_SAMPLES * gsd)))
    if ia_m / gsd > _MAX_IA_SAMPLES:
        gsd = ia_m / _MAX_IA_SAMPLES
    ia_n = max(_MIN_IA_SAMPLES + 1, _odd(ia_m / gsd))

    per_frame = v_fast * st_s_per_px / gsd      # samples per frame at the fastest speed
    if lp.pair_step is not None:
        k = int(lp.pair_step)
    elif per_frame > 0:
        k = max(1, int(math.floor(_AUTO_PAIR_FRACTION * ia_n / per_frame)))
    else:
        k = 1
    margin = int(math.ceil(per_frame * k)) + _MARGIN_PAD

    n = ia_n + 2 * margin
    offs = (np.arange(n) - 0.5 * (n - 1)) * gsd
    # Row iy across the flow, column ix along it.
    grid = (offs[None, :, None] * along[None, None, :]
            + offs[:, None, None] * across[None, None, :])          # (n, n, 2)
    layouts = []
    for b in boxes:
        pts_w = (b.centre_world[None, None, :] + grid).reshape(-1, 2)
        layouts.append(PatchLayout(
            index=b.index, centre_world=b.centre_world, along_world=along,
            across_world=across, gsd_m=gsd, ia_n=ia_n, margin_n=margin,
            pts_world=pts_w, pts_px=rect.world_to_image(pts_w),
        ))
    return layouts, k


def _ccorr(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Unnormalised correlation of ``template`` over every position in ``image``."""
    return cv2.matchTemplate(np.ascontiguousarray(image, dtype=np.float32),
                             np.ascontiguousarray(template, dtype=np.float32),
                             cv2.TM_CCORR).astype(np.float64)


def _box(a: np.ndarray, ia: int) -> np.ndarray:
    """Sum of ``a`` over every ``ia x ia`` window that fits, in float64."""
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    c[1:, 1:] = np.cumsum(np.cumsum(np.asarray(a, dtype=np.float64), axis=0), axis=1)
    return c[ia:, ia:] - c[:-ia, ia:] - c[ia:, :-ia] + c[:-ia, :-ia]


class _Block:
    """Running sums of one sub-ensemble, all additive, so blocks merge by adding.

    They are what the zero-mean normalised correlation of the
    background-subtracted patches expands into: see :meth:`_Station.finish`.
    """

    __slots__ = ("corr", "tz", "q_tz", "s_cur", "q_cur", "q_box", "n")

    def __init__(self, n: int, ia: int, margin: int):
        size = 2 * margin + 1
        self.corr = np.zeros((size, size))   # sum of window x zero-mean template
        self.tz = np.zeros((ia, ia))         # sum of the zero-mean templates
        self.q_tz = 0.0                      # sum of their squares
        self.s_cur = np.zeros((n, n))        # sum of the later patches
        self.q_cur = np.zeros((n, n))        # ... squared
        self.q_box = np.zeros((size, size))  # sum of squared window sums
        self.n = 0

    def absorb(self, other: "_Block") -> None:
        for name in self.__slots__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @staticmethod
    def total(blocks: Sequence["_Block"]) -> "_Block":
        b0 = blocks[0]
        out = _Block(b0.s_cur.shape[0], b0.tz.shape[0], (b0.corr.shape[0] - 1) // 2)
        for b in blocks:
            out.absorb(b)
        return out


class _Station:
    """Everything one station accumulates as the frames stream past."""

    def __init__(self, layout: PatchLayout, pair_step: int, block_pairs: int):
        n = layout.n
        self.layout, self.k = layout, int(pair_step)
        self.block_pairs = max(1, int(block_pairs))
        self.ring: dict[int, np.ndarray] = {}
        self.blocks: list[_Block] = []
        self.block: Optional[_Block] = None
        self.s_all = np.zeros((n, n))     # sum of every patch, for the mean
        self.n_all = 0
        self.n_missing = 0

    def add(self, t: int, patch: np.ndarray) -> None:
        self.s_all += patch
        self.n_all += 1
        self.ring[t] = patch
        self.ring.pop(t - self.k - 1, None)
        prev = self.ring.get(t - self.k)
        if prev is None:
            return
        lay = self.layout
        m, ia = lay.margin_n, lay.ia_n
        if self.block is None:
            self.block = _Block(lay.n, ia, m)
        b = self.block
        tz = prev[m:m + ia, m:m + ia].astype(np.float64)
        tz -= tz.mean()
        b.corr += _ccorr(patch, tz)
        b.tz += tz
        b.q_tz += float(np.sum(tz * tz))
        b.s_cur += patch
        b.q_cur += np.square(patch, dtype=np.float64)
        b.q_box += np.square(_box(patch, ia))
        b.n += 1
        if b.n >= self.block_pairs:
            self.blocks.append(b)
            self.block = None
            if len(self.blocks) >= _MAX_BLOCKS:
                merged = []
                for i in range(0, len(self.blocks) - 1, 2):
                    self.blocks[i].absorb(self.blocks[i + 1])
                    merged.append(self.blocks[i])
                if len(self.blocks) % 2:
                    merged.append(self.blocks[-1])
                self.blocks = merged
                self.block_pairs *= 2

    def finish(self, subtract_mean: bool) -> tuple[np.ndarray, np.ndarray, int]:
        """The normalised ensemble plane, each sub-ensemble's peak, the pair count."""
        lay = self.layout
        m, ia = lay.margin_n, lay.ia_n
        blocks = self.blocks + ([self.block] if self.block is not None and self.block.n else [])
        n_pairs = sum(b.n for b in blocks)
        size = 2 * m + 1
        if not n_pairs:
            return np.zeros((size, size)), np.empty((0, 2)), 0

        mean = self.s_all / max(1, self.n_all) if subtract_mean else np.zeros_like(self.s_all)
        mean_z = mean[m:m + ia, m:m + ia] - mean[m:m + ia, m:m + ia].mean()
        mean_corr = _ccorr(mean, mean_z)
        mean_box = _box(mean, ia)
        n_ia = float(ia * ia)

        def normalised(b: _Block) -> np.ndarray:
            """Block ``b``'s zero-mean normalised correlation of P - B with T - B_T.

            Each pair contributes, for a window at offset ``d``,

                sum (W_d - mean W_d)(Tz - Bz) / sqrt(sum (Tz - Bz)^2 * sum (W_d - mean W_d)^2)

            with ``W = P - B`` the later patch less the background and ``Tz``,
            ``Bz`` the template and its background with their means removed;
            numerator and energies are summed over the pairs before dividing.
            Every term is bilinear or quadratic in the patches, so it expands
            into the block's running sums and the background ``B`` -- which is
            why the background need not be known while the frames stream.
            """
            num = (b.corr - _ccorr(b.s_cur, mean_z) - _ccorr(mean, b.tz)
                   + b.n * mean_corr)
            e_tmpl = b.q_tz - 2.0 * float(np.sum(b.tz * mean_z)) + b.n * float(np.sum(mean_z ** 2))
            sq = _box(b.q_cur - 2.0 * mean * b.s_cur + b.n * mean ** 2, ia)
            box_sq = b.q_box - 2.0 * mean_box * _box(b.s_cur, ia) + b.n * mean_box ** 2
            e_win = sq - box_sq / n_ia
            denom = np.sqrt(max(e_tmpl, 0.0) * np.maximum(e_win, 0.0))
            ok = denom > 1e-12
            return np.where(ok, num / np.where(ok, denom, 1.0), 0.0)

        plane = normalised(_Block.total(blocks))
        disp = []
        if len(blocks) > 1:
            for p in (normalised(b) for b in blocks):
                iy, ix = (int(v) for v in np.unravel_index(int(np.argmax(p)), p.shape))
                if 0 < iy < size - 1 and 0 < ix < size - 1:
                    dy, dx = subpixel_peak(p, iy, ix)
                    disp.append((ix + dx - m, iy + dy - m))
        return plane, np.array(disp, dtype=float).reshape(-1, 2), n_pairs


class PatchCollector:
    """Samples every station's patch from each frame and correlates the pairs.

    One collector serves every section of a survey.  ``pair_steps`` gives, for
    each layout, the frames between the two images of a pair; a section's
    layouts share theirs.
    """

    def __init__(self, layouts: Sequence[PatchLayout], pair_steps: Sequence[int],
                 st_s_per_px: float, cfg: LspivConfig):
        self.layouts = list(layouts)
        self.st_s_per_px = float(st_s_per_px)
        self.subtract_mean = cfg.background == "mean"
        self.counts = [lay.n * lay.n for lay in self.layouts]
        self.offsets = np.cumsum([0] + self.counts)
        self.all_pts = (np.vstack([lay.pts_px for lay in self.layouts])
                        if self.layouts else np.empty((0, 2)))
        block_pairs = int(round(_BLOCK_S / self.st_s_per_px))
        self._stations = [_Station(lay, k, block_pairs)
                          for lay, k in zip(self.layouts, pair_steps)]
        self.n_rows = 0

    def add_frame(self, grey: np.ndarray, transform: Optional[np.ndarray] = None) -> None:
        if self.layouts:
            pts = self.all_pts if transform is None else transform_points(transform, self.all_pts)
            vals = sample_bilinear(grey, pts)
            for i, st in enumerate(self._stations):
                n = st.layout.n
                patch = vals[self.offsets[i]:self.offsets[i + 1]].reshape(n, n)
                finite = np.isfinite(patch)
                st.n_missing += int(patch.size - finite.sum())
                # The correlation is zero-mean in every window, so this changes
                # nothing but the size of the sums; samples off the frame take
                # the patch's mean, which adds no texture.
                mu = float(patch[finite].mean()) if finite.any() else 0.0
                st.add(self.n_rows, np.where(finite, patch - mu, 0.0))
        self.n_rows += 1

    def finish(self) -> list[CorrelationResult]:
        out = []
        for st, count in zip(self._stations, self.counts):
            plane, block_disp, n_pairs = st.finish(self.subtract_mean)
            out.append(CorrelationResult(
                layout=st.layout, plane=plane, block_disp=block_disp, n_pairs=n_pairs,
                pair_step=st.k, st_s_per_px=self.st_s_per_px,
                out_of_frame_fraction=st.n_missing / max(1, count * self.n_rows),
            ))
        return out


def _secondary_peak(plane: np.ndarray, iy: int, ix: int) -> float:
    """The highest value outside the main peak's region round ``(iy, ix)``.

    The region is where the plane stays above half the peak, joined to it, and
    one sample beyond: an ensemble peak is several samples wide, and its own
    shoulder must not count as a rival.
    """
    from scipy.ndimage import binary_dilation, label

    peak = float(plane[iy, ix])
    labels, _ = label(plane > _PEAK_REGION_FRACTION * peak)
    region = binary_dilation(labels == labels[iy, ix], iterations=1)
    rest = plane[~region]
    return float(rest.max()) if rest.size else 0.0


def estimate(corr: CorrelationResult) -> VectorEstimate:
    """Reduce one station's ensemble correlation to a velocity vector."""
    lay = corr.layout
    m = lay.margin_n
    dt = corr.pair_step * corr.st_s_per_px
    to_world = np.column_stack([lay.along_world, lay.across_world]) * (lay.gsd_m / dt)

    samples = corr.block_disp @ to_world.T if len(corr.block_disp) else np.empty((0, 2))
    est = VectorEstimate(
        velocity_world=np.full(2, np.nan), samples_world=samples, n_used=corr.n_pairs,
        out_of_frame_fraction=corr.out_of_frame_fraction,
    )
    plane = corr.plane
    if corr.n_pairs == 0 or not np.any(plane > 0):
        est.flags.append("no_correlation_peak")
        return est

    iy, ix = (int(v) for v in np.unravel_index(int(np.argmax(plane)), plane.shape))
    peak = float(plane[iy, ix])
    second = _secondary_peak(plane, iy, ix)
    est.peak_ratio = _PEAK_RATIO_CAP if second <= 0 else min(_PEAK_RATIO_CAP, peak / second)

    if iy in (0, plane.shape[0] - 1) or ix in (0, plane.shape[1] - 1):
        # The texture moved as far as the search reaches, or further: the
        # true peak may lie outside the plane.
        est.flags.append("displacement_at_search_edge")
    dy, dx = subpixel_peak(plane, iy, ix)
    disp = np.array([ix + dx - m, iy + dy - m])
    est.velocity_world = to_world @ disp
    est.displacement_px = float(np.hypot(*disp))
    return est


def advice(cfg: Config, ests: Sequence[VectorEstimate], pair_step: int) -> Optional[str]:
    """Suggest an ``lspiv.pair_step`` when the displacements are too small to resolve well.

    Sub-pixel peak fitting resolves a few hundredths of a sample, so a
    displacement of a sample or less leaves the velocity with a relative error
    of several percent; a few samples is better.  Large displacements are
    already guarded against by the search margin and the edge flag.
    """
    d = np.array([e.displacement_px for e in ests if np.isfinite(e.displacement_px)])
    if d.size == 0:
        return None
    med = float(np.median(d))
    if med >= 1.5:
        return None
    suggested = max(pair_step + 1, int(round(pair_step * 3.0 / max(med, 0.1))))
    return (
        f"the median LS-PIV displacement is only {med:.2f} samples per pair, so the "
        f"sub-pixel peak fit limits the accuracy; try lspiv.pair_step: {suggested} "
        f"(it is {pair_step}), which moves the texture further between the two images"
    )
