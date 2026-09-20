"""
polyview.imputation
-------------------
Missing-view handling.

Real multi-view datasets are rarely complete: a sensor fails, a modality is
too expensive to acquire for every subject, a questionnaire goes unanswered.
This package encodes a missing view as a row of ``NaN`` and offers three
ways of dealing with it, from the cheapest to the most informed.

Imputers  (fill the gaps, keep every sample)
    SimpleViewImputer           per-view column mean/median/constant
    KNNViewImputer              average the nearest samples that have the view
    CrossViewRegressionImputer  predict the view from the other views

Complete-case analysis  (drop the gaps, keep the data untouched)
    DropIncompleteSamples       transformer discarding incomplete samples
    drop_missing_views          the same, as a function

Mask utilities  (describe, build and simulate missingness)
    missing_entry_mask          cell-level ``NaN`` mask, per view
    missing_view_mask           sample x view mask of missing views
    complete_case_mask          samples that have every view
    missing_rate                fraction of samples missing each view
    mark_missing_views          turn an availability mask into ``NaN`` rows
    simulate_missing_views      drop views at random, for benchmarking

All three imputers follow the usual ``fit`` / ``transform`` contract, take a
list of views or a :class:`~polyview.dataset.multiviewdataset.MultiViewDataset`,
and return ``NaN``-free views, so they can be placed first in a
:class:`~polyview.pipeline.polypipeline.PolyPipeline`.

Examples
--------
>>> import numpy as np
>>> import polyview as pv
>>> from polyview.imputation import KNNViewImputer, simulate_missing_views
>>> mvd = pv.make_multiview_gaussian(n_samples=200, n_views=3, random_state=0)
>>> incomplete, mask = simulate_missing_views(mvd, 0.2, random_state=0)
>>> complete = KNNViewImputer(n_neighbors=5).fit_transform(incomplete)
>>> labels = pv.MultiViewKMeans(n_clusters=3, random_state=0).fit_predict(complete)
"""

from polyview.imputation._base import BaseViewImputer
from polyview.imputation.cross_view import CrossViewRegressionImputer
from polyview.imputation.drop import DropIncompleteSamples, drop_missing_views
from polyview.imputation.knn import KNNViewImputer
from polyview.imputation.mask import (
    complete_case_mask,
    mark_missing_views,
    missing_entry_mask,
    missing_rate,
    missing_view_mask,
    simulate_missing_views,
)
from polyview.imputation.simple import SimpleViewImputer

__all__ = [
    # base
    "BaseViewImputer",
    # imputers
    "SimpleViewImputer",
    "KNNViewImputer",
    "CrossViewRegressionImputer",
    # complete-case analysis
    "DropIncompleteSamples",
    "drop_missing_views",
    # mask utilities
    "missing_entry_mask",
    "missing_view_mask",
    "complete_case_mask",
    "missing_rate",
    "mark_missing_views",
    "simulate_missing_views",
]
