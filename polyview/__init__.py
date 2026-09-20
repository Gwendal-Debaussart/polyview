"""polyview public package API.

This module exposes the most commonly used classes and helpers at the
package root to support a concise import style:

        import polyview as pv
        mvd = pv.MultiViewDataset([...])
        model = pv.MultiviewKMeans(...)
"""

from importlib.metadata import PackageNotFoundError, version

from . import augmentation, cluster, dataset, fusion, imputation
from .base import (
    BaseFusion,
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
from .augmentation.random_subspace import RandomSubspaceViews, random_subspace
from .augmentation.multi_kernels import MultiKernel, multi_kernels
from .augmentation.view_splitter import ViewSplitter
from .imputation.cross_view import CrossViewRegressionImputer
from .imputation.drop import DropIncompleteSamples, drop_missing_views
from .imputation.knn import KNNViewImputer
from .imputation.mask import (
    complete_case_mask,
    mark_missing_views,
    missing_entry_mask,
    missing_rate,
    missing_view_mask,
    simulate_missing_views,
)
from .imputation.simple import SimpleViewImputer
from .embed.gcca import GCCA
from .embed.mcca import MCCA

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
    "imputation",
    "BaseMultiView",
    "BaseFusion",
    "BaseLateFusion",
    "BaseMultiViewTransformer",
    "BaseMultiViewClusterer",
    "BaseMultiViewEmbedder",
    "MultiViewDataset",
    "MultiViewKMeans",
    "MultiViewNMF",
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
    "random_subspace",
    "RandomSubspaceViews",
    "MultiKernel",
    "multi_kernels",
    "ViewSplitter",
    "make_multiview_gaussian",
    "GCCA",
    "MCCA",
    "SimpleViewImputer",
    "KNNViewImputer",
    "CrossViewRegressionImputer",
    "DropIncompleteSamples",
    "drop_missing_views",
    "missing_entry_mask",
    "missing_view_mask",
    "complete_case_mask",
    "missing_rate",
    "mark_missing_views",
    "simulate_missing_views",
]
