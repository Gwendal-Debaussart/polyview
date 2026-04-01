# TODO -- polyview

This file tracks todos for the project. It is not meant to be backlog of todos but rather a place to track 'todos' that are not finished, or recently completed. It is organized by part of the codebase.

## cluster

## fusion

- CCA and classical variants of CCA (e.g. kernel CCA, deep CCA) for multi-view embedding
- Late fustion methods, e.g. ensemble clustering, co-training, etc.

## embedd

- NMF-based multi-view embedding methods
- Multi-view MDS

## metrics

- consensus scores

## Pipelines

- A pipeline that takes in multiple views of data, applies appropriate preprocessing steps (e.g. normalization, dimensionality reduction), and then applies a fusion method to produce a final embedding or clustering result.

## Partial multiview methods

- Implement partial multiview methods that can handle missing views, e.g. matrix completion-based methods, or methods that can learn from incomplete data.

## viz

Implement the backbone for multiview visualization, e.g. a function that takes in multiple views and produces a 2D or 3D visualization of the data, possibly using dimensionality reduction techniques like t-SNE or UMAP.