View splitter
=============

:class:`~polyview.augmentation.view_splitter.ViewSplitter` cuts a single
matrix into consecutive column blocks, one per view. It is the inverse of
:class:`~polyview.fusion.early.ConcatFusion` and of
:meth:`~polyview.dataset.multiviewdataset.MultiViewDataset.to_numpy`.

Placed first in a
:class:`~polyview.pipeline.polypipeline.PolyPipeline`, it lets a multi-view
workflow receive its views as one 2-D array, which is what scikit-learn's
model-selection tools expect, since they split samples along the first axis.
This makes :class:`~sklearn.model_selection.GridSearchCV` and
:func:`~sklearn.model_selection.cross_val_score` usable on a multi-view
pipeline.

``ViewSplitter.n_views`` reports how many views ``n_features`` describes, or
0 when it is unset. It is kept out of the listing below: ``n_views`` is the
word every other docstring uses inside shape descriptions such as ``ndarray
of shape (n_views,)``, and a documented attribute of that name turns each of
them into a link to this page.

.. automodule:: polyview.augmentation.view_splitter
   :members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: n_views
