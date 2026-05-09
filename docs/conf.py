# Configuration file for the Sphinx documentation builder.

import os
import sys

repo_root = os.path.abspath("..")
sys.path.insert(0, repo_root)

import JenpyROQ

project = "JenpyROQ"
copyright = "2022 onwards, Gregorio Carullo, Sebastiano Bernuzzi, Matteo Breschi, Jacopo Tissino"
author = "Gregorio Carullo, Sebastiano Bernuzzi, Matteo Breschi, Jacopo Tissino"

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst"]
master_doc = "index"

pygments_style = "friendly"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 2,
    "sticky_navigation": True,
    "titles_only": True,
}
html_title = "JenpyROQ documentation"
html_logo = "JenpyROQ_docs_image.svg"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
htmlhelp_basename = "JenpyROQdocs"

version = JenpyROQ.__version__
release = JenpyROQ.__version__

mathjax_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js"
    "?config=TeX-AMS_SVG"
)
mathjax_config = {
    "messageStyle": "none",
    "SVG": {
        "font": "STIX-Web",
        "scale": 96,
        "linebreaks": {"automatic": True},
    },
    "TeX": {
        "equationNumbers": {"autoNumber": "none"},
    },
}
