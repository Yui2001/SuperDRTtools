# -*- coding: utf-8 -*-
"""Peak-analysis orchestration and user-facing result extraction."""

from __future__ import annotations

import copy
import math

import numpy as np
from scipy.signal import find_peaks, savgol_filter


DEFAULT_MAX_AUTO_PEAKS = 12


def estimate_peak_count(entry, max_peaks=DEFAULT_MAX_AUTO_PEAKS):
    """Estimate a stable peak count from an existing DRT curve."""
    gamma = np.asarray(getattr(entry, 'gamma', []), dtype=float).reshape(-1)
    if gamma.size < 3 or not np.any(np.isfinite(gamma)):
        return max(1, int(max_peaks))

    gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
    window = max(5, (gamma.size // 25) | 1)
    window = min(window, gamma.size if gamma.size % 2 else gamma.size - 1)
    if window >= 5:
        smooth = savgol_filter(gamma, window_length=window, polyorder=2, mode='interp')
    else:
        smooth = gamma

    amplitude = max(0.0, float(np.max(smooth) - np.min(smooth)))
    noise = 1.4826 * float(np.median(np.abs(gamma - smooth)))
    prominence = max(0.02 * amplitude, 3.0 * noise, np.finfo(float).eps)
    height = max(0.0, float(np.min(smooth) + 0.01 * amplitude))
    distance = max(3, gamma.size // 50)
    indices, _ = find_peaks(smooth, height=height, prominence=prominence, distance=distance)
    return max(1, min(int(max_peaks), int(indices.size or 1)))


def _component_metrics(tau, component):
    tau = np.asarray(tau, dtype=float).reshape(-1)
    component = np.asarray(component, dtype=float).reshape(-1)
    if tau.size != component.size or tau.size == 0:
        raise ValueError('Peak component and relaxation-time vectors do not match.')

    valid = np.isfinite(tau) & (tau > 0) & np.isfinite(component)
    if not np.any(valid):
        raise ValueError('Peak component contains no finite data.')
    tau = tau[valid]
    component = component[valid]
    order = np.argsort(tau)
    tau = tau[order]
    component = component[order]

    peak_index = int(np.argmax(component))
    tau_peak = float(tau[peak_index])
    height = float(component[peak_index])
    resistance = float(np.trapz(component, x=np.log(tau)))

    half = 0.5 * height
    above = np.flatnonzero(component >= half)
    if above.size >= 2:
        fwhm_decades = float(np.log10(tau[above[-1]] / tau[above[0]]))
    else:
        fwhm_decades = float('nan')

    return {
        'tau_s': tau_peak,
        'frequency_hz': float(1.0 / (2.0 * math.pi * tau_peak)),
        'height_ohm': height,
        'resistance_ohm': resistance,
        'fwhm_decades': fwhm_decades,
    }


def attach_peak_results(entry, components, peak_method='separate', count_mode='Auto'):
    """Sort fitted components and attach serializable peak information."""
    tau = np.asarray(entry.out_tau_vec, dtype=float).reshape(-1)
    component_arrays = [np.asarray(component, dtype=float).reshape(-1) for component in components]
    metrics = [_component_metrics(tau, component) for component in component_arrays]
    order = sorted(range(len(metrics)), key=lambda index: metrics[index]['tau_s'])
    component_arrays = [component_arrays[index] for index in order]
    metrics = [metrics[index] for index in order]

    total_resistance = float(sum(max(0.0, item['resistance_ohm']) for item in metrics))
    results = []
    for index, item in enumerate(metrics):
        result = dict(item)
        result['name'] = f'Peak {index + 1}'
        result['component_index'] = index
        result['fraction_percent'] = (
            100.0 * max(0.0, item['resistance_ohm']) / total_resistance
            if total_resistance > 0 else 0.0
        )
        results.append(result)

    entry.N_peaks = len(results)
    entry.peak_components = component_arrays
    entry.peak_results = results
    entry.peak_method = str(peak_method)
    entry.peak_count_mode = str(count_mode)
    entry.peak_analysis_complete = True
    entry.peak_total_resistance_ohm = total_resistance
    if str(peak_method).lower() == 'combine':
        entry.out_gamma_fit = np.asarray(entry.gamma_fit_tot, dtype=float)
    else:
        entry.out_gamma_fit = component_arrays
    return entry


def analyze_peak_entry(entry, params):
    """Run the established peak fit and add non-algorithmic result metadata."""
    from ..algorithms.runs import peak_analysis as run_peak_analysis

    params = dict(params or {})
    working = copy.deepcopy(entry)
    count_mode = str(params.get('peak_count', 'Auto'))
    if count_mode.lower() == 'auto':
        requested_peaks = estimate_peak_count(
            working, int(params.get('max_auto_peaks', DEFAULT_MAX_AUTO_PEAKS))
        )
    else:
        requested_peaks = max(1, int(count_mode))

    analyzed = run_peak_analysis(
        working,
        rbf_type=params['rbf_type'],
        data_used=params['data_used'],
        induct_used=params['induct_used'],
        der_used=params['der_used'],
        cv_type=params['cv_type'],
        reg_param=params['reg_param'],
        shape_control=params['shape_control'],
        coeff=params['coeff'],
        peak_method='separate',
        N_peaks=requested_peaks,
    )
    raw_components = getattr(analyzed, 'out_gamma_fit', None)
    components = [] if raw_components is None else list(raw_components)
    if not components:
        raise ValueError('No DRT peaks were identified.')
    return attach_peak_results(
        analyzed,
        components,
        peak_method=params.get('peak_method', 'separate'),
        count_mode=count_mode,
    )
