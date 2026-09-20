Cross-view regression imputation
================================

One multi-output regressor is fitted per view, mapping the concatenation of
all other views onto it, using the samples for which the target view is
observed. Where nearest-neighbour imputation copies real samples, this learns
an explicit view-to-view mapping, which is usually the stronger choice when
the views are close to linearly related and few complete samples are
available.

.. automodule:: polyview.imputation.cross_view
   :members:
   :show-inheritance:
   :undoc-members:
