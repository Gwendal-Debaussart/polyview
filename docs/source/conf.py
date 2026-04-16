# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import tomllib

sys.path.insert(0, os.path.abspath("../.."))

project = "polyview"
copyright = "2026, Gwendal Debaussart-Joniec"
author = "Gwendal Debaussart-Joniec"
with open(os.path.abspath("../../pyproject.toml"), "rb") as f:
    release = tomllib.load(f)["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "sklearn",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/polyview.svg"
html_favicon = "_static/polyview.svg"

html_css_files = [
    "styling.css",
]
html_js_files = [
    "force-light.js",
]

html_show_sourcelink = False
