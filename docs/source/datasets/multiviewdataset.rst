Multi-view dataset container
============================

:class:`~polyview.dataset.multiviewdataset.MultiViewDataset` is the core data
structure of the library: an ordered collection of views sharing one sample
axis, together with optional sample labels and view names. Every estimator in
polyview accepts it interchangeably with a plain list of arrays, and it adds
the bookkeeping a list cannot carry — naming views, subsetting samples or
views while keeping labels aligned, splitting into train and test, and saving
to disk.

.. automodule:: polyview.dataset.multiviewdataset
   :members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: views, labels, view_names, n_views, n_samples, n_features
