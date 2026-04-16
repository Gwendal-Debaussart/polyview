"""polyview public package API.

This module exposes the most commonly used classes and helpers at the
package root to support a concise import style:

        import polyview as pv
        mvd = pv.MultiViewDataset([...])
        model = pv.MultiviewKMeans(...)
"""

from importlib.metadata import PackageNotFoundError, version

from . import augmentation, cluster, dataset, fusion
from .base import (
    BaseLateFusion,
    BaseMultiView,
    BaseMultiViewClusterer,
    BaseMultiViewEmbedder,
    BaseMultiViewTransformer,
)
from .cluster.mv_kmeans import (
    MultiViewKMeans,
)
from .cluster.mv_coreg_sc import (
    MultiViewCoRegSpectralClustering,
)
from .cluster.mv_nmf import MultiViewNMF
from .dataset.multiviewdataset import MultiViewDataset
from .dataset.make_multiview_gaussian import make_multiview_gaussian
from .fusion.early import ConcatFusion, NormalizedFusion, WeightedFusion
from .fusion.kernel_fusion import (
    KernelFusion,
    KernelSpec,
    center_kernel,
    is_valid_kernel,
    normalize_kernel,
)
from .pipeline.polypipeline import PolyPipeline
from .augmentation.random_projections import RandomProjectionViews, random_projection
from .augmentation.multi_kernels import MultiKernel, multi_kernels
from .embed.gcca import GCCA

# Backward-compatible alias matching the README namespace.
datasets = dataset

try:
    __version__ = version("polyview")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "cluster",
    "dataset",
    "datasets",
    "fusion",
    "augmentation",
    "BaseMultiView",
    "BaseLateFusion",
    "BaseMultiViewTransformer",
    "BaseMultiViewClusterer",
    "BaseMultiViewEmbedder",
    "MultiViewDataset",
    "MultiViewKMeans",
    "MultiViewCoRegSpectralClustering",
    "ConcatFusion",
    "WeightedFusion",
    "NormalizedFusion",
    "KernelSpec",
    "KernelFusion",
    "center_kernel",
    "normalize_kernel",
    "is_valid_kernel",
    "PolyPipeline",
    "random_projection",
    "RandomProjectionViews",
    "MultiKernel",
    "multi_kernels",
    "make_multiview_gaussian",
    "GCCA",
]
