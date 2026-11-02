"""sglang build hook for this research checkout.

The upstream Rust extensions (radix-tree core, gRPC, Rust HTTP server,
multimodal preprocessing) were pruned with ../rust; the Python fallbacks they
replace are what this checkout runs, so the package builds with setuptools only.
"""

from setuptools import setup

setup()
