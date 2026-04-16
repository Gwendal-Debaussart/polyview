"""View-augmentation utilities.

This package contains helpers that turn one input matrix into multiple
augmented views, such as random projections and kernel-based view generation.
"""

from .random_projections import RandomProjectionViews, random_projection
from .multi_kernels import MultiKernel, multi_kernels

__all__ = [
    "random_projection",
    "RandomProjectionViews",
    "MultiKernel",
    "multi_kernels",
]
