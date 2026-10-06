"""Geometric distortion calibration of an MRI scan of a calibration cube.

The phantom contains small dots on a regular 3D grid with 20 mm spacing.
Pipeline:
  1. Load the DICOM series into a volume with scanner (patient) coordinates in mm.
  2. Detect candidate dots (small bright blobs; large structures such as the
     bars and marker cubes are rejected by size).
  3. Grow the grid: start with the dot closest to the scanner origin, then
     repeatedly assign the candidate that best matches the expected 20 mm step
     from an already identified neighbouring dot, processing dots in order of
     increasing distance to the origin.
  4. Register the ideal grid rigidly to the dots near the isocentre, where
     gradient non-linearity is negligible.
  5. Fit 5th order 3D polynomials:
       correction  : distorted (measured) position -> true position
       forward     : true position -> distorted position
"""

import glob
import heapq
import itertools
import json
import os

import numpy as np
import pydicom
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SPACING_MM = 20.0
POLY_ORDER = 5


# ---------------------------------------------------------------- loading ---

def load_volume(folder):
    """Return (volume[z, y, x] float32, origin_xyz_mm, voxel_size_xyz_mm)."""
    slices = [pydicom.dcmread(f) for f in glob.glob(os.path.join(folder, "*.dcm"))]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    orient = np.array(slices[0].ImageOrientationPatient, float)
    if not np.allclose(orient, [1, 0, 0, 0, 1, 0]):
        raise NotImplementedError("only axial, unrotated acquisitions are supported")
    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    dy, dx = (float(v) for v in slices[0].PixelSpacing)
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    dz = float(np.mean(np.diff(z)))
    origin = np.array([*map(float, slices[0].ImagePositionPatient[:2]), z[0]])
    return volume, origin, np.array([dx, dy, dz])


def voxel_to_mm(ijk_zyx, origin, voxel):
    """Convert (z, y, x) voxel coordinates to (x, y, z) mm."""
    return origin + np.asarray(ijk_zyx)[..., ::-1] * voxel


# -------------------------------------------------------------- detection ---

def detect_dots(volume, origin, voxel, threshold=250.0, min_size=4, max_size=200):
    """Detect small bright blobs; return (centroids_mm[N, 3], label_volume, keep_labels)."""
    smooth = ndi.gaussian_filter(volume, 0.7)
    # Remove slowly varying background (coil sensitivity, noise floor).
    small = ndi.median_filter(volume[::2, ::2, ::2], size=9)
    background = ndi.zoom(small, np.array(volume.shape) / np.array(small.shape), order=1)
    contrast = smooth - background
    labels, n = ndi.label(contrast > threshold)
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero((sizes >= min_size) & (sizes <= max_size))
    keep = keep[keep > 0]
    weights = np.clip(contrast, 0, None)
    centroids = np.array(ndi.center_of_mass(weights, labels, keep))
    return voxel_to_mm(centroids, origin, voxel), labels, keep


# ------------------------------------------------------------ grid growth ---

def grow_grid(points, spacing=SPACING_MM, tolerance=0.3):
    """Assign integer grid indices to detected dots.

    Starts at the dot closest to the scanner origin. Each identified dot
    predicts its six neighbours; the prediction extrapolates the local step
    (from the opposite neighbour, if known) so that it follows the distortion.
    The candidate closest to the prediction is accepted when it lies within
    tolerance * spacing. Dots are expanded in order of distance to the origin.

    Returns dict {(i, j, k): point_index}.
    """
    tree = cKDTree(points)
    seed = int(np.argmin(np.linalg.norm(points, axis=1)))
    grid = {(0, 0, 0): seed}
    used = {seed}
    heap = [(np.linalg.norm(points[seed]), (0, 0, 0))]
    steps = [np.array(s) for s in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                   (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    while heap:
        _, idx = heapq.heappop(heap)
        p = points[grid[idx]]
        for s in steps:
            nb = tuple(np.add(idx, s))
            if nb in grid:
                continue
            back = tuple(np.subtract(idx, s))
            if back in grid:
                step = p - points[grid[back]]
            else:
                step = spacing * s
            prediction = p + step
            dist, j = tree.query(prediction)
            if dist > tolerance * spacing or j in used:
                continue
            grid[nb] = int(j)
            used.add(int(j))
            heapq.heappush(heap, (np.linalg.norm(points[j]), nb))
    return grid


# -------------------------------------------------------- registration ---

def rigid_fit(src, dst):
    """Least squares rigid transform (Kabsch): dst ~= src @ R.T + t."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cd - cs @ r.T


# ------------------------------------------------------------ polynomial ---

def poly_exponents(order=POLY_ORDER):
    """All (a, b, c) with a + b + c <= order (56 terms for order 5)."""
    return [e for e in itertools.product(range(order + 1), repeat=3) if sum(e) <= order]


class Polynomial3D:
    """Vector valued 3D polynomial, evaluated on coordinates scaled by `scale` mm."""

    def __init__(self, exponents, coef, scale):
        self.exponents = np.asarray(exponents)
        self.coef = np.asarray(coef)          # [n_terms, 3]
        self.scale = float(scale)

    def design(self, xyz):
        u = np.asarray(xyz) / self.scale
        return np.prod(u[:, None, :] ** self.exponents[None], axis=2)

    def __call__(self, xyz):
        return self.design(xyz) @ self.coef

    @classmethod
    def fit(cls, src, dst, order=POLY_ORDER, scale=100.0):
        exps = poly_exponents(order)
        p = cls(exps, np.zeros((len(exps), 3)), scale)
        coef, *_ = np.linalg.lstsq(p.design(src), dst, rcond=None)
        p.coef = coef
        return p

    def to_ras(self):
        """Same mapping expressed in RAS instead of LPS coordinates.

        RAS = D @ LPS with D = diag(-1, -1, 1), so P_ras(r) = D P_lps(D r):
        each coefficient gains a factor d_i * (-1)^(a + b).
        """
        d = np.array([-1.0, -1.0, 1.0])
        sign = (-1.0) ** (self.exponents[:, 0] + self.exponents[:, 1])
        return Polynomial3D(self.exponents, self.coef * sign[:, None] * d[None, :], self.scale)

    def to_dict(self):
        return {"order": int(self.exponents.sum(1).max()),
                "scale_mm": self.scale,
                "exponents": self.exponents.tolist(),
                "coefficients_xyz": self.coef.tolist()}


# ------------------------------------------------------------- pipeline ---

def calibrate(folder, central_radius=60.0):
    volume, origin, voxel = load_volume(folder)
    points, labels, keep = detect_dots(volume, origin, voxel)
    grid = grow_grid(points)

    idx = np.array(list(grid.keys()), float)
    measured = points[list(grid.values())]
    nominal = idx * SPACING_MM

    # Rigid registration of the nominal grid using only near-isocentre dots.
    central = np.linalg.norm(measured, axis=1) < central_radius
    r, t = rigid_fit(nominal[central], measured[central])
    true = nominal @ r.T + t

    correction = Polynomial3D.fit(measured, true)
    forward = Polynomial3D.fit(true, measured)
    return dict(volume=volume, origin=origin, voxel=voxel, points=points,
                labels=labels, keep=keep, dot_index=np.array(list(grid.values())), grid_index=idx.astype(int),
                measured=measured, true=true, rotation=r, translation=t,
                correction=correction, forward=forward)


def statistics(res):
    measured, true = res["measured"], res["true"]
    dist = np.linalg.norm(measured - true, axis=1)
    radius = np.linalg.norm(true, axis=1)
    corr_err = np.linalg.norm(res["correction"](measured) - true, axis=1)
    fwd_err = np.linalg.norm(res["forward"](true) - measured, axis=1)

    def summary(v):
        return {"max": float(v.max()), "mean": float(v.mean()),
                "rms": float(np.sqrt(np.mean(v ** 2))), "p95": float(np.percentile(v, 95))}

    per_radius = []
    for rmax in (50, 75, 100, 125, 150, 175, 200):
        m = radius <= rmax
        if m.any():
            per_radius.append({"radius_mm": rmax, "n": int(m.sum()),
                               **summary(dist[m]),
                               "corrected_max": float(corr_err[m].max())})
    axis_err = np.abs(measured - true)
    return {
        "n_candidates": int(len(res["points"])),
        "n_dots": int(len(measured)),
        "grid_extent": {"min": res["grid_index"].min(0).tolist(),
                        "max": res["grid_index"].max(0).tolist()},
        "distortion": summary(dist),
        "distortion_max_per_axis": axis_err.max(0).tolist(),
        "worst_dot": {"true_mm": true[dist.argmax()].tolist(),
                      "measured_mm": measured[dist.argmax()].tolist()},
        "correction_residual": summary(corr_err),
        "forward_residual": summary(fwd_err),
        "per_radius": per_radius,
    }


def export_web(res, stats, outdir):
    """Write data files consumed by docs/index.html."""
    os.makedirs(outdir, exist_ok=True)
    vol = res["volume"]
    lo, hi = np.percentile(vol, [1, 99.9])
    vol8 = np.clip((vol - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    import gzip
    with gzip.open(os.path.join(outdir, "volume.u8.gz"), "wb", compresslevel=9) as f:
        f.write(vol8.tobytes())

    # Segmentation: voxel centres of accepted dot components (and rejected large ones).
    labels = res["labels"]
    sizes = np.bincount(labels.ravel())
    dot_mask = np.isin(labels, res["keep"])
    big_mask = (labels > 0) & (sizes[labels] > 200)
    seg = voxel_to_mm(np.argwhere(dot_mask), res["origin"], res["voxel"])
    big = voxel_to_mm(np.argwhere(big_mask)[::4], res["origin"], res["voxel"])

    # Dense evaluation grid of the forward polynomial (distortion field) inside the cube.
    gmin, gmax = res["grid_index"].min(0), res["grid_index"].max(0)
    axes = [np.arange(a, b + 0.5, 0.5) * SPACING_MM for a, b in zip(gmin, gmax)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    nominal = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    field_true = nominal @ res["rotation"].T + res["translation"]
    field_meas = res["forward"](field_true)
    # Only trust the fit near identified dots (no extrapolation into empty corners).
    near, _ = cKDTree(res["grid_index"] * SPACING_MM).query(nominal)
    field_meas[near > 0.75 * SPACING_MM] = np.nan

    def r2(a):
        return np.round(np.asarray(a), 2).tolist()
    data = {
        "volume": {"shape_zyx": list(vol.shape), "origin_xyz": res["origin"].tolist(),
                   "voxel_xyz": res["voxel"].tolist(), "file": "volume.u8.gz"},
        "dots": {"grid_index": res["grid_index"].tolist(), "measured": r2(res["measured"]),
                 "true": r2(res["true"]),
                 "corrected": r2(res["correction"](res["measured"])),
                 "forward": r2(res["forward"](res["true"]))},
        "stray": r2(np.delete(res["points"], res["dot_index"], axis=0)),
        "segmentation": r2(seg),
        "rejected": r2(big),
        "field": {"shape": [len(a) for a in axes], "true": r2(field_true),
                  "measured": [None if np.isnan(p[0]) else p for p in r2(field_meas)]},
        "rigid": {"rotation": res["rotation"].tolist(), "translation_mm": res["translation"].tolist()},
        "correction_polynomial": res["correction"].to_dict(),
        "forward_polynomial": res["forward"].to_dict(),
        "stats": stats,
    }
    with open(os.path.join(outdir, "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))


def export_polynomials(res, path):
    """Write correction and forward polynomials in both LPS and RAS coordinates."""
    out = {"description": "correction maps distorted (image) positions to true positions; "
                          "forward maps true positions to distorted (image) positions. "
                          "Positions in mm. Each output coordinate is "
                          "sum_k c_k * (x/s)^a_k * (y/s)^b_k * (z/s)^c_k with s = scale_mm. "
                          "lps: DICOM patient coordinates (+x left, +y posterior, +z superior); "
                          "ras: +x right, +y anterior, +z superior.",
           "valid_region_mm": {"min": res["true"].min(0).round(1).tolist(),
                               "max": res["true"].max(0).round(1).tolist()}}
    for frame, conv in (("lps", lambda p: p), ("ras", Polynomial3D.to_ras)):
        out[frame] = {"correction": conv(res["correction"]).to_dict(),
                      "forward": conv(res["forward"]).to_dict()}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with the DICOM series")
    ap.add_argument("--out", default="docs", help="output folder for web data")
    args = ap.parse_args()
    res = calibrate(args.folder)
    stats = statistics(res)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_radius"}, indent=1))
    for row in stats["per_radius"]:
        print(row)
    export_polynomials(res, os.path.join(args.out, "polynomials.json"))
    export_web(res, stats, args.out)
