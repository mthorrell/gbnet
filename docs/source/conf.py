# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GBNet"
copyright = "2025, Michael Horrell"
author = "Michael Horrell"
release = "v0.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = []

# Add these settings to control autodoc behavior
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "no-inherited-members": True,
}

# This will prevent duplicate parameter documentation
autodoc_typehints = "description"

# This will prevent the init method from being documented separately
autodoc_docstring_signature = True
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autoapi_dirs = ["../../gbnet"]
autoapi_ignore = ["*/test_*.py", "*/tests/*", "*_test.py"]

# Control how parameters are displayed
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# Control the order of documentation
autodoc_member_order = "bysource"
