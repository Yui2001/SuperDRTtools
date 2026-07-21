"""Numerical algorithms for DRT, BHT, sampling, and peak analysis."""

from importlib import import_module


__all__ = [
    "BHT",
    "HMC",
    "basics",
    "fGP",
    "nearest_PD",
    "parameter_selection",
    "peak_analysis",
    "runs",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
