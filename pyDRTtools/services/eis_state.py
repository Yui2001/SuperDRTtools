# -*- coding: utf-8 -*-
"""Shared raw/visible EIS arrays, mask state, and Lin-KK state helpers."""

import hashlib
from math import log10

import numpy as np
from impedance.validation import linKK


def _get_entry_raw_arrays(entry):
    """Return immutable raw impedance arrays for an entry; create *_0 attrs if missing."""
    if entry is None:
        raise ValueError('Empty entry')

    if not hasattr(entry, 'freq_0'):
        entry.freq_0 = np.asarray(getattr(entry, 'freq', []), dtype=float).reshape(-1).copy()
    if not hasattr(entry, 'Z_prime_0'):
        entry.Z_prime_0 = np.asarray(getattr(entry, 'Z_prime', []), dtype=float).reshape(-1).copy()
    if not hasattr(entry, 'Z_double_prime_0'):
        entry.Z_double_prime_0 = np.asarray(getattr(entry, 'Z_double_prime', []), dtype=float).reshape(-1).copy()
    if not hasattr(entry, 'Z_exp_0'):
        entry.Z_exp_0 = np.asarray(getattr(entry, 'Z_exp', entry.Z_prime_0 + 1j * entry.Z_double_prime_0),
                                   dtype=complex).reshape(-1).copy()

    freq0 = np.asarray(entry.freq_0, dtype=float).reshape(-1).copy()
    zre0 = np.asarray(entry.Z_prime_0, dtype=float).reshape(-1).copy()
    zim0 = np.asarray(entry.Z_double_prime_0, dtype=float).reshape(-1).copy()
    zexp0 = np.asarray(entry.Z_exp_0, dtype=complex).reshape(-1).copy()
    return freq0, zre0, zim0, zexp0

def _ensure_mask_state(entry):
    """Ensure per-point raw mask state exists and matches raw data length."""
    if entry is None:
        return entry
    freq0, _, _, _ = _get_entry_raw_arrays(entry)
    n = int(freq0.size)

    def _ensure_bool_attr(name: str):
        arr = np.asarray(getattr(entry, name, np.zeros(n, dtype=bool)), dtype=bool).reshape(-1)
        if arr.size != n:
            fixed = np.zeros(n, dtype=bool)
            m = min(n, arr.size)
            if m > 0:
                fixed[:m] = arr[:m]
            arr = fixed
        setattr(entry, name, arr.copy())

    _ensure_bool_attr('mask_manual_raw')
    _ensure_bool_attr('mask_auto_raw')
    entry.mask_total_raw = np.asarray(entry.mask_manual_raw | entry.mask_auto_raw, dtype=bool)
    if not hasattr(entry, 'mask_settings') or not isinstance(getattr(entry, 'mask_settings', None), dict):
        entry.mask_settings = {'freq_min': None, 'freq_max': None, 'kk_threshold': None}
    return entry

def _rebuild_visible_data_from_raw(entry, induct_choice_index: int = 0):
    """Rebuild working impedance arrays from raw arrays after inductance/mask updates."""
    if entry is None:
        raise ValueError('Empty entry')

    entry = _ensure_mask_state(entry)
    freq0, zre0, zim0, zexp0 = _get_entry_raw_arrays(entry)

    visible_keep = np.ones(freq0.size, dtype=bool)
    if int(induct_choice_index) == 2:  # discard inductive data
        visible_keep = (-zim0) > 0

    mask_total = np.asarray(getattr(entry, 'mask_manual_raw', np.zeros(freq0.size, dtype=bool)), dtype=bool) | \
                 np.asarray(getattr(entry, 'mask_auto_raw', np.zeros(freq0.size, dtype=bool)), dtype=bool)

    active_keep = visible_keep & (~mask_total)
    if not np.any(active_keep):
        raise ValueError('No impedance points remain after masking/filtering.')

    entry.visible_keep_raw = visible_keep.copy()
    entry.mask_total_raw = mask_total.copy()
    entry.active_raw_indices = np.where(active_keep)[0].astype(int)

    entry.freq = freq0[active_keep].copy()
    entry.Z_prime = zre0[active_keep].copy()
    entry.Z_double_prime = zim0[active_keep].copy()
    entry.Z_exp = zexp0[active_keep].copy()

    entry.tau = 1.0 / entry.freq
    entry.tau_fine = np.logspace(log10(entry.tau.min()) - 0.5,
                                 log10(entry.tau.max()) + 0.5,
                                 10 * entry.freq.shape[0])
    return entry

def _clear_kk_results(entry):
    """Remove cached Kramers-Kronig / lin-KK results from an entry."""
    if entry is None:
        return entry
    for attr in (
            'kk_valid', 'kk_c', 'kk_max_m', 'kk_fit_type', 'kk_selected_m', 'kk_mu', 'kk_tau',
            'kk_R0', 'kk_R', 'kk_L', 'kk_Z_fit', 'kk_res_re_pct', 'kk_res_im_pct',
            'kk_res_re_raw_pct', 'kk_res_im_raw_pct', 'kk_Z_fit_raw', 'kk_signature'
    ):
        try:
            if hasattr(entry, attr):
                delattr(entry, attr)
        except Exception:
            pass
    return entry

def _run_lin_kk(entry, c: float = 0.85, max_m: int = 50, fit_type: str = 'complex'):
    """Run K-K validation using impedance.py linKK and cache the result on the entry."""
    if entry is None:
        raise ValueError('No selected impedance data.')

    if linKK is None:
        raise ImportError(
            "impedance.py is not installed. Please install it first, e.g. pip install impedance"
        )

    c = float(c)
    max_m = int(max_m)
    if max_m < 1:
        raise ValueError('MAX Elements must be at least 1.')

    freq = np.asarray(entry.freq, dtype=float).reshape(-1)
    Z_exp = np.asarray(entry.Z_exp, dtype=complex).reshape(-1)

    if freq.size < 3:
        raise ValueError('Need at least 3 impedance points for K-K analysis.')
    if freq.size != Z_exp.size:
        raise ValueError('Frequency and impedance arrays must have the same length.')

    mask = np.isfinite(freq) & np.isfinite(Z_exp.real) & np.isfinite(Z_exp.imag) & (freq > 0)
    raw_active_idx = np.asarray(getattr(entry, 'active_raw_indices', np.arange(freq.size)), dtype=int).reshape(-1)
    if raw_active_idx.size != freq.size:
        raw_active_idx = np.arange(freq.size, dtype=int)
    freq = freq[mask]
    Z_exp = Z_exp[mask]
    raw_active_idx = raw_active_idx[mask]

    if freq.size < 3:
        raise ValueError('Too few valid impedance points for K-K analysis.')

    fit_type = str(fit_type).strip().lower()
    if fit_type not in ('real', 'imag', 'complex'):
        raise ValueError(f'Unsupported fit_type: {fit_type}')

    # impedance.py official linKK
    # returns: M, mu, Z_fit, resids_real, resids_imag
    M, mu, Z_fit, resids_real, resids_imag = linKK(
        freq,
        Z_exp,
        c=c,
        max_M=max_m,
        fit_type=fit_type,
        add_cap=False
    )

    # impedance.py residuals are fractional, convert to percent for your UI
    res_re_pct = 100.0 * np.asarray(resids_real, dtype=float)
    res_im_pct = 100.0 * np.asarray(resids_imag, dtype=float)

    entry.kk_valid = True
    entry.kk_c = c
    entry.kk_max_m = max_m
    entry.kk_fit_type = fit_type
    entry.kk_signature = _build_kk_signature(entry, c=c, max_m=max_m, fit_type=fit_type)
    entry.kk_selected_m = int(M)
    entry.kk_mu = float(mu)
    entry.kk_Z_fit = np.asarray(Z_fit, dtype=complex)
    entry.kk_res_re_pct = res_re_pct
    entry.kk_res_im_pct = res_im_pct

    try:
        freq0, _, _, _ = _get_entry_raw_arrays(entry)
        n_raw = int(freq0.size)
        kk_res_re_raw = np.full(n_raw, np.nan, dtype=float)
        kk_res_im_raw = np.full(n_raw, np.nan, dtype=float)
        kk_z_fit_raw = np.full(n_raw, np.nan + 1j * np.nan, dtype=complex)
        if raw_active_idx.size == res_re_pct.size:
            kk_res_re_raw[raw_active_idx] = res_re_pct
            kk_res_im_raw[raw_active_idx] = res_im_pct
            kk_z_fit_raw[raw_active_idx] = np.asarray(Z_fit, dtype=complex)
        entry.kk_res_re_raw_pct = kk_res_re_raw
        entry.kk_res_im_raw_pct = kk_res_im_raw
        entry.kk_Z_fit_raw = kk_z_fit_raw
    except Exception:
        pass

    # impedance.py linKK does not directly return R0/R/L/tau in its public API
    # so keep these optional / compatibility-safe
    entry.kk_tau = None
    entry.kk_R0 = None
    entry.kk_R = None
    entry.kk_L = None

    return entry

def _refresh_kk_current_arrays(entry):
    """Refresh current-view K-K arrays from cached raw K-K results after masking."""
    if entry is None or not bool(getattr(entry, 'kk_valid', False)):
        return entry
    try:
        active_idx = np.asarray(getattr(entry, 'active_raw_indices', []), dtype=int).reshape(-1)
        kk_re_raw = np.asarray(getattr(entry, 'kk_res_re_raw_pct', []), dtype=float).reshape(-1)
        kk_im_raw = np.asarray(getattr(entry, 'kk_res_im_raw_pct', []), dtype=float).reshape(-1)
        kk_fit_raw = np.asarray(getattr(entry, 'kk_Z_fit_raw', []), dtype=complex).reshape(-1)
        if active_idx.size == 0 or kk_re_raw.size == 0 or kk_im_raw.size == 0 or kk_fit_raw.size == 0:
            return entry
        valid = (active_idx >= 0) & (active_idx < kk_re_raw.size) & (active_idx < kk_im_raw.size) & (active_idx < kk_fit_raw.size)
        idx = active_idx[valid]
        entry.kk_res_re_pct = kk_re_raw[idx].copy()
        entry.kk_res_im_pct = kk_im_raw[idx].copy()
        entry.kk_Z_fit = kk_fit_raw[idx].copy()
    except Exception:
        pass
    return entry

def _array_signature_digest(values, dtype=None):
    """Return a compact, stable digest for a NumPy-compatible array."""
    try:
        arr = np.asarray(values, dtype=dtype).reshape(-1)
        arr = np.ascontiguousarray(arr)
        h = hashlib.sha1()
        h.update(str(arr.shape).encode('utf-8'))
        h.update(str(arr.dtype).encode('utf-8'))
        h.update(arr.view(np.uint8))
        return arr.size, h.hexdigest()
    except Exception:
        return 0, 'invalid'

def _build_kk_signature(entry, c=None, max_m=None, fit_type=None):
    """Build a signature for the exact data/settings used by K-K validation.

    The signature changes when the active impedance points change, e.g. after
    manual/auto masking, changing the inductance handling, or changing K-K
    parameters. It lets Run KKR skip files that have already been validated
    without invalidating their DRT fit status.
    """
    if entry is None:
        return None
    try:
        freq = np.asarray(getattr(entry, 'freq', []), dtype=float).reshape(-1)
        z_exp = np.asarray(getattr(entry, 'Z_exp', []), dtype=complex).reshape(-1)
        active_idx = np.asarray(getattr(entry, 'active_raw_indices', np.arange(freq.size)), dtype=int).reshape(-1)
        visible_keep = np.asarray(getattr(entry, 'visible_keep_raw', []), dtype=bool).reshape(-1)
        mask_total = np.asarray(getattr(entry, 'mask_total_raw', []), dtype=bool).reshape(-1)

        if active_idx.size != freq.size:
            active_idx = np.arange(freq.size, dtype=int)

        sig = (
            'kk-v2',
            'freq', _array_signature_digest(freq, dtype=float),
            'zre', _array_signature_digest(z_exp.real, dtype=float),
            'zim', _array_signature_digest(z_exp.imag, dtype=float),
            'active_raw_indices', _array_signature_digest(active_idx, dtype=int),
            'visible_keep_raw', _array_signature_digest(visible_keep, dtype=bool),
            'mask_total_raw', _array_signature_digest(mask_total, dtype=bool),
        )
        if c is not None and max_m is not None and fit_type is not None:
            sig = sig + ('c', float(c), 'max_m', int(max_m), 'fit_type', str(fit_type).strip().lower())
        return sig
    except Exception:
        return None

def _safe_kk_residual_max(re_res, im_res, n_points: int):
    """Return max(|Re residual|, |Im residual|) without all-NaN warnings."""
    n_points = int(n_points or 0)
    if n_points <= 0:
        return np.array([], dtype=float)

    re_arr = np.asarray(re_res, dtype=float).reshape(-1)
    im_arr = np.asarray(im_res, dtype=float).reshape(-1)
    if re_arr.size != n_points:
        re_arr = np.full(n_points, np.nan, dtype=float)
    if im_arr.size != n_points:
        im_arr = np.full(n_points, np.nan, dtype=float)

    kk_abs = np.vstack([np.abs(re_arr), np.abs(im_arr)])
    kk_max = np.full(n_points, np.nan, dtype=float)
    valid_cols = np.any(np.isfinite(kk_abs), axis=0)
    if np.any(valid_cols):
        kk_max[valid_cols] = np.nanmax(kk_abs[:, valid_cols], axis=0)
    return kk_max

def _entry_has_valid_kk(entry) -> bool:
    """Return True when K-K results match the current active EIS data."""
    if entry is None:
        return False
    try:
        if not bool(getattr(entry, 'kk_valid', False)):
            return False
        z_fit = np.asarray(getattr(entry, 'kk_Z_fit'))
        freq = np.asarray(getattr(entry, 'freq'))
        if z_fit.size != freq.size or z_fit.size <= 0:
            return False

        stored_sig = getattr(entry, 'kk_signature', None)
        if stored_sig is None:
            # Backward compatibility for very old in-memory entries created
            # before kk_signature existed.
            return True

        current_sig = _build_kk_signature(
            entry,
            c=getattr(entry, 'kk_c', None),
            max_m=getattr(entry, 'kk_max_m', None),
            fit_type=getattr(entry, 'kk_fit_type', None),
        )
        return stored_sig == current_sig
    except Exception:
        return False
