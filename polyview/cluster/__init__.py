"""Clustering algorithms for multi-view data."""

from .mv_kmeans import MultiViewKMeans
from .mv_coreg_sc import MultiViewCoRegSpectralClustering
from .mv_cotrain_sc import MultiViewCoTrainSpectralClustering

__all__ = ["MultiViewKMeans", "MultiViewCoRegSpectralClustering", "MultiViewCoTrainSpectralClustering"]
