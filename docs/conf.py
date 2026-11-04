"""Sphinx configuration for FAMEX documentation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from famex import __version__  # noqa: E402

project = "FAMEX"
copyright = "2026, FAMEX Development Team"
author = "FAMEX Development Team"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_heading_anchors = 4

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / napoleon --------------------------------------------------
add_module_names = False
toc_object_entries_show_parents = "hide"
autodoc_mock_imports = [
    "fairchem",
    "mace",
    "so3lr",
    "orb_models",
    "tblite",
    "upet",
    "torch",
    "torch_cluster",
    "e3nn",
    "pyscf",
    "rdkit",
]
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

# -- HTML output ----------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/FAMEX_logo.svg"
html_theme_options = {
    "logo_only": True,
}
