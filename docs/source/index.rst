polyview overview
=================

*polyview* is a sklearn-compatible multi-view learning toolkit for data where
each sample is represented by several complementary views, such as audio and
video, multiple sensor streams, or heterogeneous feature extractors.
Examples and workflow walkthroughs are available in :doc:`examples/index`.

The library provides:

- native multi-view clustering algorithms such as multi-view K-means,
  co-training spectral clustering, co-regularized spectral clustering, and
  multi-view NMF,
- embedding methods including GCCA and MCCA,
- embedding methods including GCCA, MCCA, and MvMDS,
- fusion utilities for early fusion, late fusion, and kernel fusion,
- data augmentation tools that turn a single matrix into multiple views via
  random projections, random subspaces, or multiple kernels,
- a pipeline layer that moves between single-view and multi-view stages in a
  consistent sklearn-style API.

The documentation is organized by algorithm family so you can move quickly
from the overview to the relevant API reference or example workflow.

This library is inspired by `mvlearn <https://mvlearn.github.io/>`_, which has not been updated since 2020.

.. _mvlearn: https://mvlearn.github.io/

.. toctree::
  :maxdepth: 1

  cluster/index
  datasets/index
  embedd/index
  base/index
  augmentation/index
  fusion/index
  pipeline/index
  examples/index
  misc/index
