Pipeline Workflows
==================

This page demonstrates common `PolyPipeline` patterns. It includes examples of typical multiview workflows, including per-view sklearn steps, fusion flows, and native multiview methods.

MV → MV (per-view sklearn)
-----------------------------

Use a single-view sklearn estimator directly in multiview mode. The
pipeline adapts it per view and returns one output per branch.

.. code-block:: python

   from polyview.pipeline.polypipeline import PolyPipeline
   from polyview.dataset.make_multiview_gaussian import make_multiview_gaussian
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import KMeans as SKKMeans

   mvd = make_multiview_gaussian(n_samples=120, n_features=10, n_views=3, random_state=0)

   pipe = PolyPipeline(
       steps=[
           ("scale", StandardScaler()),
           ("kmeans", SKKMeans(n_clusters=3, random_state=0)),
       ]
   )

   labels_by_view = pipe.fit_predict(mvd.views)
   print(len(labels_by_view))  # 3

MV → SV (fusion)
--------------------

Fuse per-view transformed features into a single representation for a single-view model.
Alternatives for the fusion step include elements from the `fusion` module, and some of the `embed` module's methods.

.. code-block:: python

   from polyview.pipeline.polypipeline import PolyPipeline
   from polyview.fusion.early import ConcatFusion
   from sklearn.decomposition import PCA
   from sklearn.cluster import KMeans

   pipe = PolyPipeline(
       steps=[
           ("pca", PCA(n_components=5)),
           ("fusion", ConcatFusion()),
           ("kmeans", KMeans(n_clusters=3, random_state=0)),
       ]
   )

   labels = pipe.fit_predict(mvd.views)
   print(labels.shape)

MV preprocessing → MV clustering
-----------------------------------

Apply per-view dimensionality reduction first, then cluster with a native multiview method (`MultiViewNMF`) that learns a consensus representation.

.. code-block:: python

   from polyview.pipeline.polypipeline import PolyPipeline
   from polyview.cluster.mv_nmf import MultiViewNMF
   from sklearn.decomposition import PCA

   pipe = PolyPipeline(
       steps=[
           ("pca", PCA(n_components=6)),
           ("mvnmf", MultiViewNMF(n_components=3, n_init=5, random_state=0)),
       ]
   )

   labels = pipe.fit_predict(mvd.views)
   print(labels.shape)


Late fusion (inside and outside pipeline)
-----------------------------------------

Late fusion can be either part of the same pipeline or applied as a
separate post-processing step.

.. code-block:: python

   from polyview.pipeline.polypipeline import PolyPipeline
   from polyview.fusion.late import MajorityVote
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import KMeans as SKKMeans

   # Inside the same pipeline: per-view clustering -> MajorityVote
   fused_inside = PolyPipeline(
       steps=[
           ("scale", StandardScaler()),
           ("kmeans", SKKMeans(n_clusters=3, random_state=0)),
           ("majority_vote", MajorityVote(weights=[0.2, 0.5, 0.3])),
       ]
   ).fit_predict(mvd.views)

   # Outside the pipeline: keep manual control over the fusion stage
   labels_by_view = PolyPipeline(
       steps=[
           ("scale", StandardScaler()),
           ("kmeans", SKKMeans(n_clusters=3, random_state=0)),
       ]
   ).fit_predict(mvd.views)

   fused_outside = MajorityVote(tie_break="first").fit_predict(labels_by_view)

SV → MV
-------

Start from one feature matrix, generate multiple projected views, then continue with a standard multiview clustering step.

.. code-block:: python

   import numpy as np
   from polyview.pipeline.polypipeline import PolyPipeline
   from polyview.augmentation.random_projections import RandomProjectionViews
   from sklearn.preprocessing import StandardScaler
   from polyview.cluster.mv_kmeans import MultiViewKMeans

   X = np.random.RandomState(0).randn(100, 40)

   pipe = PolyPipeline(
       steps=[
           ("rp", RandomProjectionViews(n_views=3, n_components=[10, 12, 14], random_state=0)),
           ("scale", StandardScaler()),
           ("cluster", MultiViewKMeans(n_clusters=3, random_state=0)),
       ]
   )

   labels = pipe.fit_predict(X)
   print(labels.shape)  # (100,)
   pipe.draw_diagram()

Pipeline diagram introspection
-------------------------------

`draw_diagram()` prints a branch-aware text diagram of step flow.

.. code-block:: python

   pipe.draw_diagram()           # before fit
   _ = pipe.fit_predict(mvd.views)
   pipe.draw_diagram()           # after fit
