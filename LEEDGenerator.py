# -*- coding: utf-8 -*-
# Synthetic LEED image generator (two methods): reciprocal-lattice and FFT
# Dependencies: numpy only
#%%
import numpy as np
import math
from typing import Tuple, List, Optional

# ------------------------ small utilities ------------------------

def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k.astype(np.float32)

def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = _gaussian_kernel_1d(sigma)
    # separable convolution, reflect padding
    def conv1d_along_axis(a: np.ndarray, axis: int) -> np.ndarray:
        pad = len(k) // 2
        a_pad = np.pad(a, pad_width=[(pad, pad) if i == axis else (0, 0) for i in range(a.ndim)],
                       mode="reflect")
        out = np.zeros_like(a, dtype=np.float32)
        # roll-convolution along the chosen axis
        for i, kv in enumerate(k):
            shift = i - pad
            out += kv * np.roll(a_pad, shift, axis=axis)[tuple(
                slice(pad, -pad) if j == axis else slice(None) for j in range(a.ndim)
            )]
        return out
    tmp = conv1d_along_axis(img.astype(np.float32), axis=1)
    out = conv1d_along_axis(tmp, axis=0)
    return out

def _bilinear_resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Simple bilinear resize for 2D images (float32)."""
    h, w = img.shape
    if h == new_h and w == new_w:
        return img.astype(np.float32)
    y = np.linspace(0, h - 1, new_h, dtype=np.float32)
    x = np.linspace(0, w - 1, new_w, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32); x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1);     x1 = np.clip(x0 + 1, 0, w - 1)
    wy = y - y0; wx = x - x0
    # gather rows/cols
    Ia = img[y0[:, None], x0[None, :]]
    Ib = img[y0[:, None], x1[None, :]]
    Ic = img[y1[:, None], x0[None, :]]
    Id = img[y1[:, None], x1[None, :]]
    out = (Ia * (1 - wy)[:, None] * (1 - wx)[None, :] +
           Ib * (1 - wy)[:, None] * wx[None, :] +
           Ic * wy[:, None] * (1 - wx)[None, :] +
           Id * wy[:, None] * wx[None, :])
    return out.astype(np.float32)

def _normalize01(I: np.ndarray) -> np.ndarray:
    I = I.astype(np.float32)
    I -= I.min()
    I /= (I.max() + 1e-8)
    return I

# ------------------------ lattice helpers ------------------------

BRAVAIS_2D = ["oblique", "rectangular", "centered-rectangular", "square", "hexagonal"]

def _sample_real_space_basis(bravais: str,
                             a_range=(1.0, 3.0),
                             b_range=(1.0, 3.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Return a1, a2 as 2D vectors (shape (2,)). Units arbitrary but consistent."""
    bravais = bravais.lower()
    a = np.random.uniform(*a_range)
    b = np.random.uniform(*b_range)

    if bravais == "square":
        a = b = np.random.uniform(*a_range)
        gamma = math.radians(90.0)
    elif bravais == "hexagonal":
        a = b = np.random.uniform(*a_range)
        gamma = math.radians(60.0)
    elif bravais == "rectangular":
        while abs(a / b - 1.0) < 0.25:
            b = np.random.uniform(*b_range)
        gamma = math.radians(90.0)
    elif bravais == "centered-rectangular":
        # conventional rectangular cell, centering handled later via structure factor
        while abs(a / b - 1.0) < 0.25:
            b = np.random.uniform(*b_range)
        gamma = math.radians(90.0)
    else:  # oblique
        gamma_deg = np.random.uniform(60.0, 120.0)
        while abs(gamma_deg - 90.0) < 10.0:
            gamma_deg = np.random.uniform(60.0, 120.0)
        gamma = math.radians(gamma_deg)

    a1 = np.array([a, 0.0], dtype=np.float32)
    a2 = np.array([b * math.cos(gamma), b * math.sin(gamma)], dtype=np.float32)
    return a1, a2

def _reciprocal_basis_2d(a1: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = a1[0] * a2[1] - a1[1] * a2[0]  # scalar z-component of cross
    b1 = (2 * math.pi / A) * np.array([ a2[1], -a2[0] ], dtype=np.float32)
    b2 = (2 * math.pi / A) * np.array([ -a1[1], a1[0] ], dtype=np.float32)
    return b1, b2

# ------------------------ main generator class ------------------------

class LEEDGenerator:
    """
    Generate synthetic 128x128 LEED-like images using either:
      - method="reciprocal": Gaussian spots at reciprocal lattice points
      - method="fft":  build real-space supercell, FFT to k-space intensity
    """

    def __init__(self,
                 img_size: int = 128,
                 # Reciprocal-lattice method params
                 HK: int = 8,
                 spot_sigma: float = 1.2,
                 base_amp: float = 1.0,
                 rotation_deg: Optional[float] = None,
                 random_rotation: bool = True,
                 noise_std_recip: float = 0.02,
                 # FFT method params
                 super_N: int = 32,
                 grid_N: int = 256,
                 noise_std_fft: float = 0.01,
                 k_crop: float = 0.45,
                 k_blur_sigma: float = 0.6):
        self.img_size = img_size
        self.HK = HK
        self.spot_sigma = spot_sigma
        self.base_amp = base_amp
        self.rotation_deg = rotation_deg
        self.random_rotation = random_rotation
        self.noise_std_recip = noise_std_recip

        self.super_N = super_N
        self.grid_N = grid_N
        self.noise_std_fft = noise_std_fft
        self.k_crop = k_crop
        self.k_blur_sigma = k_blur_sigma

    @staticmethod
    def classes() -> List[str]:
        return BRAVAIS_2D.copy()

    # -------------- public API --------------

    def generate(self, bravais: str, method: str = "reciprocal") -> np.ndarray:
        bravais = bravais.lower()
        assert bravais in BRAVAIS_2D, f"Unknown bravais '{bravais}'. Choose from {BRAVAIS_2D}."
        method = method.lower()
        if method == "reciprocal":
            return self._generate_reciprocal_image(bravais)
        elif method == "fft":
            return self._generate_fft_image(bravais)
        else:
            raise ValueError("method must be 'reciprocal' or 'fft'.")

    def _rotation_matrix(self) -> Optional[np.ndarray]:
        if self.rotation_deg is not None:
            angle = self.rotation_deg
        elif self.random_rotation:
            angle = float(np.random.uniform(0.0, 360.0))
        else:
            return None

        theta = math.radians(angle)
        return np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta),  math.cos(theta)]], dtype=np.float32)

    # -------------- method 1: reciprocal lattice ----------------

    def _generate_reciprocal_image(self, bravais: str) -> np.ndarray:
        a1, a2 = _sample_real_space_basis(bravais)
        b1, b2 = _reciprocal_basis_2d(a1, a2)

        # Enumerate reciprocal lattice points
        H = K = self.HK
        Gs = []
        flags_center_extinction = (bravais == "centered-rectangular")
        for h in range(-H, H + 1):
            for k in range(-K, K + 1):
                if h == 0 and k == 0:
                    continue
                if flags_center_extinction and ((h + k) % 2 != 0):
                    # C-centering extinction: only even (h+k) spots survive
                    continue
                Gs.append(h * b1 + k * b2)
        if not Gs:
            # fallback: ensure at least something
            Gs.append(1 * b1)
        G = np.stack(Gs, axis=0)  # (N,2)

        # Scale to fit image radius
        rmax = np.max(np.linalg.norm(G, axis=1) + 1e-8)
        s = 0.45 * self.img_size / rmax
        P = s * G

        # Random global rotations make the classifier orientation-invariant.
        R = self._rotation_matrix()
        if R is not None:
            P = P @ R.T

        # Shift to image center
        P[:, 0] += self.img_size / 2.0
        P[:, 1] += self.img_size / 2.0

        # Rasterize Gaussian spots
        I = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        yy, xx = np.mgrid[0:self.img_size, 0:self.img_size]
        sig2 = 2.0 * (self.spot_sigma ** 2)
        # amplitude decay with |G| (slight) to mimic intensity falloff
        norms = np.linalg.norm(G, axis=1)
        amp = self.base_amp * (1.0 / (1.0 + 0.05 * norms))
        for i, p in enumerate(P):
            dx2 = (xx - p[0]) ** 2
            dy2 = (yy - p[1]) ** 2
            I += amp[i] * np.exp(-(dx2 + dy2) / sig2)

        # Noise + normalize
        if self.noise_std_recip > 0:
            I += self.noise_std_recip * np.random.randn(*I.shape).astype(np.float32)
        I = _normalize01(I)
        return I

    # -------------- method 2: FFT of real-space lattice ---------

    def _generate_fft_image(self, bravais: str) -> np.ndarray:
        a1, a2 = _sample_real_space_basis(bravais)
        R = self._rotation_matrix()
        if R is not None:
            a1 = R @ a1
            a2 = R @ a2

        # Choose basis (structure factor):
        # - centered-rectangular: conventional rectangular cell with 2-atom basis at (0,0) and (1/2,1/2)
        # - others: single atom at (0,0)
        if bravais == "centered-rectangular":
            basis = [(0.0, 0.0, 1.0), (0.5, 0.5, 1.0)]
        else:
            basis = [(0.0, 0.0, 1.0)]

        # Build supercell atoms
        N = self.super_N
        atoms = []
        # We'll map coordinates later to a regular grid; for now, keep them in real units.
        for n in range(-N // 2, N // 2):
            for m in range(-N // 2, N // 2):
                R = n * a1 + m * a2
                for (tx, ty, f) in basis:
                    pos = R + tx * a1 + ty * a2
                    atoms.append((pos[0], pos[1], f))
        atoms = np.array(atoms, dtype=np.float32)  # (M,3)

        # Map to a fixed bounding box and rasterize impulses on grid_N × grid_N
        grid_N = self.grid_N
        margin = 1.0
        xy = atoms[:, :2]
        mins = xy.min(axis=0) - margin
        maxs = xy.max(axis=0) + margin
        scale = (np.array([grid_N - 1, grid_N - 1], dtype=np.float32)) / (maxs - mins)

        px = ((xy[:, 0] - mins[0]) * scale[0]).round().astype(np.int32)
        py = ((xy[:, 1] - mins[1]) * scale[1]).round().astype(np.int32)
        px = np.clip(px, 0, grid_N - 1)
        py = np.clip(py, 0, grid_N - 1)

        rho = np.zeros((grid_N, grid_N), dtype=np.float32)
        # accumulate atomic form factors at nearest pixels
        for (ix, iy, w) in zip(px, py, atoms[:, 2]):
            rho[iy, ix] += w

        # FFT → intensity in k-space
        F = np.fft.fftshift(np.fft.fft2(rho))
        Ik = (F * np.conj(F)).real.astype(np.float32)

        # Instrument effects: blur in k-space (point spread)
        if self.k_blur_sigma > 0:
            Ik = _gaussian_blur(Ik, sigma=self.k_blur_sigma)

        # Central crop (band-limit around DC)
        c = grid_N // 2
        r = max(8, int(self.k_crop * grid_N / 2))
        Ik_cropped = Ik[c - r:c + r, c - r:c + r]

        # Resize to target
        Ik_small = _bilinear_resize(Ik_cropped, self.img_size, self.img_size)

        # Noise + normalize
        if self.noise_std_fft > 0:
            Ik_small += self.noise_std_fft * np.random.randn(*Ik_small.shape).astype(np.float32)
        Ik_small = _normalize01(Ik_small)
        return Ik_small


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gen = LEEDGenerator(img_size=128, HK=8, spot_sigma=1.2, rotation_deg=None)

    # Generate one image per class with each method
    for cls in gen.classes():
        I_rec = gen.generate(cls, method="reciprocal")
        I_fft = gen.generate(cls, method="fft")
        print(cls, "reciprocal mean/std:", float(I_rec.mean()), float(I_rec.std()),
              "| fft mean/std:", float(I_fft.mean()), float(I_fft.std()))
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(I_rec, cmap="gray"); ax[0].set_title(f"{cls} (reciprocal)")
        ax[1].imshow(I_fft, cmap="gray"); ax[1].set_title(f"{cls} (FFT)")
    plt.show()
