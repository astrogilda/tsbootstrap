import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsbootstrap"
current_year = datetime.now().year
copyright = f"2023 - {current_year} (MIT License), Sankalp Gilda"
author = "Sankalp Gilda"
release = "0.7.1"  # x-release-please-version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_gallery.load_style",  # thumbnail-grid CSS for the notebook gallery
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["**.ipynb_checkpoints"]
suppress_warnings = ["ref.undefined", "ref.footnote"]

# -- Options for nbsphinx (notebook gallery) ---------------------------------
# Notebooks ship with committed outputs and are executed as a separate CI gate
# (nbmake), so the docs build renders them without re-running cells. This keeps a
# transient cell warning from breaking the warnings-as-errors (-W) docs build.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# Per-notebook "run this" badges injected above each rendered tutorial.
nbsphinx_prolog = r"""
.. raw:: html

   <div class="admonition note">
     <p><strong>Run this tutorial:</strong>
     <a href="https://colab.research.google.com/github/astrogilda/tsbootstrap/blob/main/docs/source/{{ env.docname }}.ipynb">Open in Colab</a>
     |
     <a href="https://mybinder.org/v2/gh/astrogilda/tsbootstrap/HEAD?labpath=docs/source/{{ env.docname }}.ipynb">Launch Binder</a></p>
   </div>
"""

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#module-sphinx.ext.intersphinx
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "arch": ("https://arch.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "navigation_with_keys": False,
}

# html_theme = "furo"
html_static_path = []

# -- Options for autodoc -----------------------------------------------------
# Skip Pydantic internal attributes that cause issues with defer_build=True
autodoc_default_options = {
    "exclude-members": "__pydantic_serializer__, __pydantic_validator__, __pydantic_extra__",
}
