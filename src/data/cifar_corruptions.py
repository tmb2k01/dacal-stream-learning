"""On-the-fly CIFAR-10-C-style corruptions.

Approximate re-implementation of a subset of the corruptions of
Hendrycks & Dietterich (2019), "Benchmarking Neural Network Robustness to
Common Corruptions and Perturbations", using their CIFAR severity
parameters.  Used only as a fallback when the official
``data/CIFAR-10-C/<corruption>.npy`` files are not available
(``scripts/download_CIFAR-10-C.sh``); results on synthesized corruptions
are close to, but not bit-identical with, the official release.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from scipy.ndimage import gaussian_filter

SEVERITY_PARAMS: dict[str, list] = {
    "gaussian_noise": [0.04, 0.06, 0.08, 0.09, 0.10],
    "shot_noise": [500, 250, 100, 75, 50],
    "impulse_noise": [0.01, 0.02, 0.03, 0.05, 0.07],
    "speckle_noise": [0.06, 0.1, 0.12, 0.16, 0.2],
    "gaussian_blur": [0.4, 0.6, 0.7, 0.8, 1.0],
    "contrast": [0.75, 0.5, 0.4, 0.3, 0.15],
    "brightness": [0.05, 0.1, 0.15, 0.2, 0.3],
    "saturate": [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)],
    "pixelate": [0.95, 0.9, 0.85, 0.75, 0.65],
    "jpeg_compression": [80, 65, 58, 50, 40],
}

SYNTHESIZABLE_CORRUPTIONS = tuple(SEVERITY_PARAMS)


def corrupt_images(
    images: np.ndarray,
    corruption: str,
    severity: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Corrupt a ``(N, 32, 32, 3)`` uint8 batch; returns uint8 of the same shape."""
    if corruption not in SEVERITY_PARAMS:
        raise ValueError(
            f"Cannot synthesize corruption {corruption!r}; "
            f"supported: {sorted(SEVERITY_PARAMS)}"
        )
    if not 1 <= severity <= 5:
        raise ValueError("severity must be between 1 and 5")
    rng = np.random.default_rng(rng)
    c = SEVERITY_PARAMS[corruption][severity - 1]
    x = np.asarray(images, dtype=np.float64) / 255.0

    if corruption == "gaussian_noise":
        out = x + rng.normal(scale=c, size=x.shape)
    elif corruption == "shot_noise":
        out = rng.poisson(x * c) / c
    elif corruption == "impulse_noise":
        out = x.copy()
        flip = rng.random(x.shape) < c
        salt = rng.random(x.shape) < 0.5
        out[flip & salt] = 1.0
        out[flip & ~salt] = 0.0
    elif corruption == "speckle_noise":
        out = x + x * rng.normal(scale=c, size=x.shape)
    elif corruption == "gaussian_blur":
        out = gaussian_filter(x, sigma=(0, c, c, 0))
    elif corruption == "contrast":
        means = x.mean(axis=(1, 2), keepdims=True)
        out = (x - means) * c + means
    elif corruption in {"brightness", "saturate"}:
        hsv = rgb_to_hsv(np.clip(x, 0, 1))
        if corruption == "brightness":
            hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
        else:
            hsv[..., 1] = np.clip(hsv[..., 1] * c[0] + c[1], 0, 1)
        out = hsv_to_rgb(hsv)
    elif corruption == "pixelate":
        size = int(32 * c)
        out = np.stack(
            [
                np.asarray(
                    Image.fromarray(img)
                    .resize((size, size), Image.Resampling.BOX)
                    .resize((32, 32), Image.Resampling.BOX)
                )
                for img in np.asarray(images, dtype=np.uint8)
            ]
        ).astype(np.float64) / 255.0
    else:  # jpeg_compression
        decoded = []
        for img in np.asarray(images, dtype=np.uint8):
            buffer = io.BytesIO()
            Image.fromarray(img).save(buffer, "JPEG", quality=int(c))
            decoded.append(np.asarray(Image.open(buffer)))
        out = np.stack(decoded).astype(np.float64) / 255.0

    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
