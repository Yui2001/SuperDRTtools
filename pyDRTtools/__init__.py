"""SuperDRTtools package with lazily loaded feature namespaces."""

from importlib import import_module


__authors__ = "Francesco Ciucci, Adeleke Maradesa"
__date__ = "29th June, 2024"

_MODULES = {
    "algorithms": "algorithms",
    "app": "app",
    "controllers": "controllers",
    "infrastructure": "infrastructure",
    "services": "services",
    "ui": "ui",
    # Package-level aliases retain the common ``from pyDRTtools import ...`` API
    # without putting compatibility files back in the package root.
    "basics": "algorithms.basics",
    "BHT": "algorithms.BHT",
    "HMC": "algorithms.HMC",
    "fGP": "algorithms.fGP",
    "nearest_PD": "algorithms.nearest_PD",
    "parameter_selection": "algorithms.parameter_selection",
    "peak_analysis": "algorithms.peak_analysis",
    "runs": "algorithms.runs",
}

__all__ = list(_MODULES)


def __getattr__(name):
    target = _MODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{target}")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
