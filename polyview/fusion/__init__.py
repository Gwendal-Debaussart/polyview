"""
polyview.fusion
---------------
View fusion strategies.

Early fusion  (combine views before learning)
    ConcatFusion        horizontal concatenation (baseline)
    WeightedFusion      per-view scalar weighting
    NormalizedFusion    z-score each view, then concatenate

Kernel fusion  (combine views in kernel space)
    KernelSpec          per-view kernel configuration
    KernelFusion        weighted sum of per-view kernels

Kernel utilities  (standalone helpers)
    center_kernel       remove mean in RKHS
    normalize_kernel    scale so K[i,i] = 1
    is_valid_kernel     check symmetry + PSD

Late fusion  (combine per-view predictions)
    MajorityVote        sample-wise vote across views
"""

from polyview.fusion.early import ConcatFusion, WeightedFusion, NormalizedFusion
from polyview.fusion.kernel_fusion import (
    KernelFusion,
    KernelSpec,
    center_kernel,
    normalize_kernel,
    is_valid_kernel,
)
from polyview.fusion.late import MajorityVote

__all__ = [
    # early
    "ConcatFusion",
    "WeightedFusion",
    "NormalizedFusion",
    # kernel fusion
    "KernelFusion",
    "KernelSpec",
    # kernel utilities
    "center_kernel",
    "normalize_kernel",
    "is_valid_kernel",
    # late fusion
    "MajorityVote",
]