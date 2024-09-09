import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

version_file = "../../python/sglang/version.py"
with open(version_file, "r") as f:
    exec(compile(f.read(), version_file, "exec"))
__version__ = locals()["__version__"]

project = "SGLang"
copyright = "2023-2024, SGLang"
author = "SGLang Team"

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

autosectionlabel_prefix_document = True

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_logo = "_static/image/logo.png"
html_title = project
html_copy_source = True
html_last_updated_fmt = ""

html_theme_options = {
    "path_to_docs": "docs/en",
    "repository_url": "https://github.com/sgl-project/sglang",
    "repository_branch": "main",
    "show_navbar_depth": 3,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
}

html_static_path = ["_static"]
html_css_files = ["css/readthedocs.css"]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
myst_heading_anchors = 5

htmlhelp_basename = "sglangdoc"

latex_elements = {}

latex_documents = [
    (master_doc, "sglang.tex", "sglang Documentation", "SGLang Team", "manual"),
]

man_pages = [(master_doc, "sglang", "sglang Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "sglang",
        "sglang Documentation",
        author,
        "sglang",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project

epub_exclude_files = ["search.html"]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

autodoc_preserve_defaults = True
navigation_with_keys = False

autodoc_mock_imports = [
    "torch",
    "transformers",
    "triton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
