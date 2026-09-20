Imputation
==========

Real multi-view datasets are rarely complete: a sensor fails, a modality is
too expensive to acquire for every subject, a questionnaire goes unanswered.
This module encodes a missing view as a row of ``NaN`` and offers three ways
of dealing with it: marginal imputation, nearest-neighbour imputation, and
cross-view regression, plus complete-case analysis for when imputing is not
appropriate.

All imputers follow the usual ``fit`` / ``transform`` contract, accept a list
of views or a
:class:`~polyview.dataset.multiviewdataset.MultiViewDataset`, and return
``NaN``-free views, so they can be placed first in a
:class:`~polyview.pipeline.polypipeline.PolyPipeline`.

.. toctree::
   :maxdepth: 1

   simple
   knn
   cross_view
   drop
   mask

Choosing a strategy
-------------------

:class:`~polyview.imputation.SimpleViewImputer` ignores the relationships
between views, so it is the baseline any other strategy should beat.
:class:`~polyview.imputation.KNNViewImputer` and
:class:`~polyview.imputation.CrossViewRegressionImputer` both exploit the
fact that the views of one sample are correlated: the first copies real
samples, the second learns an explicit view-to-view mapping and is usually
stronger when the views are close to linearly related.
:func:`~polyview.imputation.simulate_missing_views` lets you compare them on
your own data by removing views you actually have.

See also
--------

- :doc:`../base/index`
- :doc:`../pipeline/index`
