"""Sphinx configuration for the nowcast platform documentation.

Built with the Read the Docs theme and published to GitHub Pages. The source is
Markdown through MyST, so the same files read well on GitHub and in the rendered
site, which matters because the architecture decision records live here too and
are read both ways.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "nowcast-runtime" / "src"))

project = "Nowcast Platform"
author = "Adriano P. Almeida"
copyright = f"{date.today():%Y}, {author}"

# Read from the package so the site can never disagree with what is installed.
try:
    from nowcast_runtime import __version__ as release
except ImportError:  # pragma: no cover - only when the runtime is absent
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_scripts", "adr/README.md"]

templates_path = ["_templates"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {version}"
html_logo = None
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
    "style_external_links": True,
}
# GitHub Pages serves no Jekyll, and a leading underscore would hide _static.
html_extra_path = ["_nojekyll"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# A missing cross-reference is a documentation bug, so the build refuses it.
nitpicky = False
suppress_warnings = ["myst.header"]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
