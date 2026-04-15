"""polyview public package API.

This module exposes the most commonly used classes and helpers at the
package root to support a concise import style:

        import polyview as pv
        mvd = pv.MultiViewDataset([...])
        model = pv.MultiviewKMeans(...)
"""

from importlib.metadata import PackageNotFoundError, version

from . import cluster, dataset, fusion
from .base import (
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
from .fusion.early import ConcatFusion, NormalizedFusion, WeightedFusion
from .fusion.kernel_fusion import (
    KernelFusion,
    KernelSpec,
    center_kernel,
    is_valid_kernel,
    normalize_kernel,
)
from .pipeline.polypipeline import PolyPipeline

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
    "BaseMultiView",
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
    "make_multiview_gaussian",
    "GCCA",
]
