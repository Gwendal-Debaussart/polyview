Nearest-neighbour imputation
============================

For a sample whose view is missing, neighbours are searched among the samples
that do have it, using the other views as the search space; the missing block
is then the average of those neighbours. Distances are computed per view on
standardised features and averaged over the views two samples have in common,
so samples sharing a single view remain comparable to samples sharing
several.

.. automodule:: polyview.imputation.knn
   :members:
   :show-inheritance:
   :undoc-members:
