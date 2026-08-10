# -*- coding: utf-8 -*-
"""IreneRewrite documentation build configuration file."""

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.imgconverter',
]

# Autosectionlabel prefix setting — prevents duplicate key warnings
autosectionlabel_prefix_document = True

# Read the Docs builders may not provide optional optimization backends.
# Mock them so autodoc can import modules and render API docs.
autodoc_mock_imports = [
    'cvxopt',
    'cvxpy',
    'gpkit',
    'gpkit.constraints',
    'gpkit.constraints.bounded',
]

templates_path = ['_templates']
source_suffix = '.rst'
root_doc = 'index'

# -- Project information --------------------------------------------------

project = 'IreneRewrite'
copyright = '2016-2026, Mehdi Ghasemi'
author = 'Mehdi Ghasemi'
version = '1.3'
release = '1.3.1'
language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

html_theme = 'furo'
html_logo = './images/IreneLogo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    'preamble': r'''
\usepackage{mathrsfs}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage{upquote}
''',
}

latex_documents = [
    (root_doc, 'IreneRewrite.tex', u'IreneRewrite Documentation',
     u'Mehdi Ghasemi', 'manual'),
]

latex_logo = './images/IreneLogoSmall.png'


# -- Options for manual page output ---------------------------------------

man_pages = [
    (root_doc, 'irenerewrite', u'IreneRewrite Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (root_doc, 'IreneRewrite', u'IreneRewrite Documentation',
     author, 'IreneRewrite',
     'Polynomial optimization via SDP, SONC, and mean polynomial hierarchies.',
     'Optimization'),
]


# -- Intersphinx ----------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
}
