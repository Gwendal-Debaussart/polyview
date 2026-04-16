"""View-augmentation utilities.

This package contains helpers that turn one input matrix into multiple
augmented views, such as random projections, random subspaces, and kernel-based view generation.
"""

from .random_projections import RandomProjectionViews, random_projection
from .random_subspace import RandomSubspaceViews, random_subspace
from .multi_kernels import MultiKernel, multi_kernels

__all__ = [
    "random_projection",
    "RandomProjectionViews",
    "random_subspace",
    "RandomSubspaceViews",
    "MultiKernel",
    "multi_kernels",
]
