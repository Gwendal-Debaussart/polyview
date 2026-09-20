Simple imputation
=================

Marginal imputation replaces every missing entry by a statistic of its own
column — the mean, the median, or a fixed constant — computed over the
samples for which the view is observed. It ignores the correlations between
views, which makes it the natural baseline: a strategy that cannot beat it is
not exploiting the multi-view structure at all.

.. automodule:: polyview.imputation.simple
   :members:
   :show-inheritance:
   :undoc-members:
