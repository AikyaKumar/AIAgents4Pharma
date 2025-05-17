#!/usr/bin/env python3
"""
This package provides modules for fetching and downloading academic papers from arXiv.
"""

# Import modules
from . import download_arxiv_input, download_medrxiv_input, download_biorxiv_input

__all__ = [
    "download_arxiv_input",
    "download_medrxiv_input",
    "download_biorxiv_input",
]
