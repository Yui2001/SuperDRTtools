# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Baptiste Py, Ting Hei Wan, Adeleke Maradesa, DongXu Ye'

__date__ = '4th October 2024'

import csv
import os
import copy
import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

from numpy import absolute, angle
from PyQt5 import QtGui, QtWidgets, QtCore

# --- Parallel fitting performance notes ---
# Many numerical solvers (OpenBLAS/MKL/OpenMP) will use multiple CPU threads
# *inside a single fit*. If each fit already uses most/all cores, running
# multiple fits concurrently (Python threads) won't reduce total wall time.
# We therefore (optionally) limit native threadpools per fit during "Fit All"
# so that multiple fits can truly run side-by-side across cores.
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None


def _native_thread_limit_ctx(n_threads: int):
    """Context manager to limit BLAS/OMP thread pools (best-effort)."""
    if _threadpool_limits is None:
        return nullcontext()
    try:
        return _threadpool_limits(limits=int(max(1, n_threads)))
    except Exception:
        return nullcontext()


def apply_flat_theme(app: QtWidgets.QApplication) -> None:
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    font = QtGui.QFont("Arial")
    font.setPointSize(11)
    app.setFont(font)

    qss = """
    QWidget {
        background: #F5F5F7;
        color: #1D1D1F;
        font-family: Arial;
        font-size: 11pt;
    }

    /* Make combo boxes show a clear down arrow */
    QComboBox {
        padding-right: 28px;              /* 给箭头留位置 */
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border: 0px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 10px;
    }

    QComboBox::down-arrow {
        width: 0px;
        height: 0px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 2px solid #6E6E73;    /* 灰色倒三角 */
        margin-right: 10px;
    }

    QGroupBox {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 12px;
        margin-top: 16px;
        padding: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: #1D1D1F;
        font-weight: 600;
    }

    QPushButton {
        background: #FFFFFF;
        border: 1px solid #D2D2D7;
        border-radius: 10px;
        padding: 6px 12px;
        min-height: 28px;
    }
    QPushButton:hover { background: #F2F2F7; border-color: #C7C7CC; }
    QPushButton:pressed { background: #EAEAEE; border-color: #BDBDC2; }
    QPushButton:checked { background: #E9E9EE; border-color: #BDBDC2; }

    QLineEdit, QComboBox {
        background: #FFFFFF;
        border: 1px solid #D2D2D7;
        border-radius: 10px;
        padding: 6px 10px;
        min-height: 28px;
        selection-background-color: #0A84FF;
    }
    QLineEdit:focus, QComboBox:focus { border: 1px solid #0A84FF; }

    QComboBox::drop-down { border: 0px; width: 24px; }
    QComboBox QAbstractItemView {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 10px;
        selection-background-color: #0A84FF;
        selection-color: white;
        padding: 4px;
    }

    /* QTabWidget (top navigation) */
    QTabWidget::pane { border: 0px; background: transparent; }
    QTabBar::tab {
        background: #FFFFFF;
        border: 0px solid #D2D2D7;
        padding: 6px 14px;
        margin-right: 8px;
        border-radius: 10px;
        min-height: 28px;
    }
    QTabBar::tab:selected { background: #E9E9EE; border-color: #BDBDC2; }
    QTabBar::tab:hover { background: #F2F2F7; }

    QScrollArea { border: 0px; background: transparent; }

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 6px 4px 6px 4px;
}
QScrollBar::handle:vertical {
    background: rgba(60, 60, 67, 0.35);
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: rgba(60, 60, 67, 0.50); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 4px 6px 4px 6px;
}
QScrollBar::handle:horizontal {
    background: rgba(60, 60, 67, 0.35);
    border-radius: 5px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: rgba(60, 60, 67, 0.50); }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

/* ComboBox arrow + popup */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 26px;
    border: 0px;
}
QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 10px;
    selection-background-color: #0A84FF;
    selection-color: white;
    padding: 4px;
    outline: 0;
}


/* ComboBox popup visibility fix (Qt sometimes uses QTreeView/QTableView under the hood) */
QAbstractItemView {
    background: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid #E5E5EA;
    border-radius: 10px;
    outline: 0;
}
QAbstractItemView::item {
    color: #1D1D1F;
    padding: 6px 10px;
    min-height: 28px;
}
QAbstractItemView::item:selected {
    background: #0A84FF;
    color: #FFFFFF;
    border-radius: 8px;
}
QTreeView, QTableView, QListView {
    background: #FFFFFF;
    color: #1D1D1F;
}

/* Make QComboBox popup items clearly visible */
QListView {
    background: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid #E5E5EA;
    border-radius: 10px;
    outline: 0;
}
QListView::item {
    color: #1D1D1F;
    padding: 6px 10px;
    min-height: 28px;
}
QListView::item:selected {
    background: #0A84FF;
    color: #FFFFFF;
}

/* Kill unexpected frame borders/lines around the top bar (DO NOT style all QFrame, it breaks combo popups) */
QFrame#show_layout {
    border: 0px;
    background: transparent;
}

/* ComboBox popup - force readable items (some Qt styles use QFrame/QAbstractItemView internally) */
QComboBox QAbstractItemView, QComboBox QListView, QListView, QTreeView, QTableView {
    background: #FFFFFF;
    color: #1D1D1F;
}
QComboBox QAbstractItemView::item, QListView::item, QTreeView::item, QTableView::item {
    color: #1D1D1F;
}
QComboBox QAbstractItemView::item:selected, QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {
    background: #0A84FF;
    color: #FFFFFF;
}
"""
    app.setStyleSheet(qss)


from PyQt5.QtWidgets import QFileDialog
from . import layout
#
from .runs import *
#
from impedance.validation import linKK
import matplotlib as mpl

mpl.use("Qt5Agg")
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


class FileListDelegate(QtWidgets.QStyledItemDelegate):
    """Draw filename (left) and fit status (right, gray) in the Files list."""

    STATUS_ROLE = QtCore.Qt.UserRole + 1

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        # draw background/selection without text
        text = opt.text
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        painter.save()
        r = option.rect.adjusted(8, 0, -8, 0)

        filename = text
        status = index.data(self.STATUS_ROLE) or ""

        fm = opt.fontMetrics
        status_w = fm.horizontalAdvance(status) + 6 if status else 0

        left_rect = QtCore.QRect(r.left(), r.top(), max(0, r.width() - status_w), r.height())
        status_rect = QtCore.QRect(r.right() - status_w + 1, r.top(), status_w, r.height())

        # filename color respects selection
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.setPen(option.palette.color(QtGui.QPalette.Active, QtGui.QPalette.HighlightedText))
        else:
            painter.setPen(option.palette.color(QtGui.QPalette.Active, QtGui.QPalette.Text))
        painter.drawText(left_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, filename)

        if status:
            # status color: gray when not selected, white when selected
            if option.state & QtWidgets.QStyle.State_Selected:
                painter.setPen(option.palette.color(QtGui.QPalette.Active, QtGui.QPalette.HighlightedText))
            else:
                painter.setPen(QtGui.QColor(142, 142, 147))  # iOS-like gray
            painter.drawText(status_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, status)

        painter.restore()


class ExternalFileDropFilter(QtCore.QObject):
    """Enable drag&drop import of EIS text files onto the Files list without breaking InternalMove."""

    def __init__(self, gui_window):
        super().__init__(gui_window)
        self._gui = gui_window

    @staticmethod
    def _extract_paths(event):
        try:
            md = event.mimeData()
            if md is None or not md.hasUrls():
                return []
            paths = []
            for u in md.urls():
                try:
                    if u.isLocalFile():
                        p = u.toLocalFile()
                        if p and os.path.isfile(p):
                            paths.append(p)
                except Exception:
                    continue
            return paths
        except Exception:
            return []

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QtCore.QEvent.DragEnter, QtCore.QEvent.DragMove):
            paths = self._extract_paths(event)
            if paths:
                event.acceptProposedAction()
                return True
        elif et == QtCore.QEvent.Drop:
            paths = self._extract_paths(event)
            if paths:
                try:
                    self._gui.import_files_from_paths(paths)
                except Exception:
                    pass
                event.acceptProposedAction()
                return True
        return super().eventFilter(obj, event)


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class _FitWorkerSignals(QtCore.QObject):
    """Signals for background fit worker."""
    finished = QtCore.pyqtSignal(str, object, object)  # key, fitted_entry, signature
    error = QtCore.pyqtSignal(str, str)  # key, error message


class _FitWorker(QtCore.QRunnable):
    """Run one fit task in a worker thread (no UI access)."""

    def __init__(self, key: str, entry, mode: str, params: dict, signature):
        super().__init__()
        self.key = key
        self.entry = entry
        self.mode = mode
        self.params = params or {}
        self.signature = signature
        self.signals = _FitWorkerSignals()
        try:
            # allow Qt to delete runnable automatically after execution
            self.setAutoDelete(True)
        except Exception:
            pass

    def run(self):
        try:
            # Limit native BLAS/OpenMP threads per fit so multiple fits can
            # actually run in parallel across CPU cores.
            n_native = int((self.params or {}).get('native_threads', 1) or 1)
            with _native_thread_limit_ctx(n_native):
                entry = self._run_fit(self.entry, self.mode, self.params)
            self.signals.finished.emit(self.key, entry, self.signature)
        except Exception as e:
            self.signals.error.emit(self.key, str(e))

    @staticmethod
    def _run_fit(entry, mode: str, params: dict):
        """Pure computation. Must NOT touch Qt/UI objects."""
        if entry is None:
            raise ValueError("Empty entry")

        if mode == 'simple':
            source_entry = entry
            entry, used_kk = _prepare_entry_for_drt_fit(entry)
            entry = simple_run(
                entry,
                rbf_type=params['rbf_type'],
                data_used=params['data_used'],
                induct_used=params['induct_used'],
                der_used=params['der_used'],
                cv_type=params['cv_type'],
                reg_param=params['reg_param'],
                shape_control=params['shape_control'],
                coeff=params['coeff'],
            )
            # compute lambda_value like original logic
            if params['cv_type'] == 'custom':
                entry.lambda_value = params['reg_param']
            else:
                entry.lambda_value = basics.optimal_lambda(
                    entry.A_re, entry.A_im, entry.b_re, entry.b_im,
                    entry.M, params['data_used'], params['induct_used'], -3, params['cv_type']
                )
            return _finalize_fitted_entry_for_display(entry, source_entry, used_kk)

        if mode == 'bayesian':
            source_entry = entry
            entry, used_kk = _prepare_entry_for_drt_fit(entry)
            entry = Bayesian_run(
                entry,
                rbf_type=params['rbf_type'],
                data_used=params['data_used'],
                induct_used=params['induct_used'],
                der_used=params['der_used'],
                cv_type=params['cv_type'],
                reg_param=params['reg_param'],
                shape_control=params['shape_control'],
                coeff=params['coeff'],
                NMC_sample=params['sample_number'],
            )
            return _finalize_fitted_entry_for_display(entry, source_entry, used_kk)

        if mode == 'BHT':
            entry = BHT_run(
                entry,
                params['rbf_type'],
                params['der_used'],
                params['shape_control'],
                params['coeff'],
            )
            return entry

        raise ValueError(f"Unknown mode: {mode}")


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


def _prepare_entry_for_drt_fit(entry):
    """Return a deep-copied entry for DRT fitting using the current raw/masked data only."""
    if entry is None:
        raise ValueError('Empty entry')
    fit_entry = copy.deepcopy(entry)
    fit_entry.fit_data_source = 'raw'
    return fit_entry, False


def _finalize_fitted_entry_for_display(fitted_entry, source_entry, used_kk: bool):
    """Keep the current raw/masked EIS data for display after DRT fitting."""
    if fitted_entry is None:
        return fitted_entry
    try:
        fitted_entry.fit_data_source = 'raw'
    except Exception:
        pass
    if source_entry is None:
        return fitted_entry
    try:
        fitted_entry.freq = np.asarray(source_entry.freq, dtype=float).copy()
        fitted_entry.Z_prime = np.asarray(source_entry.Z_prime, dtype=float).copy()
        fitted_entry.Z_double_prime = np.asarray(source_entry.Z_double_prime, dtype=float).copy()
        fitted_entry.Z_exp = np.asarray(source_entry.Z_exp, dtype=complex).copy()
        fitted_entry.tau = np.asarray(source_entry.tau, dtype=float).copy()
        fitted_entry.tau_fine = np.asarray(source_entry.tau_fine, dtype=float).copy()
    except Exception:
        pass
    for attr in ('freq_0', 'Z_prime_0', 'Z_double_prime_0', 'Z_exp_0',
                 'mask_manual_raw', 'mask_auto_raw', 'mask_total_raw', 'mask_settings',
                 'visible_keep_raw', 'active_raw_indices',
                 'kk_valid', 'kk_c', 'kk_max_m', 'kk_fit_type', 'kk_selected_m', 'kk_mu', 'kk_tau',
                 'kk_R0', 'kk_R', 'kk_L', 'kk_Z_fit', 'kk_res_re_pct', 'kk_res_im_pct',
                 'kk_res_re_raw_pct', 'kk_res_im_raw_pct', 'kk_Z_fit_raw', 'kk_signature'):
        try:
            if hasattr(source_entry, attr):
                setattr(fitted_entry, attr, copy.deepcopy(getattr(source_entry, attr)))
        except Exception:
            pass
    return fitted_entry


class GUI(QtWidgets.QMainWindow):
    def __init__(self):

        # Initalize parent
        QtWidgets.QMainWindow.__init__(self)

        # Initalize GUI layout
        self.ui = layout.Ui_MainWindow()  # layout
        self.ui.setupUi(self)

        # setting data to be initially none
        self.data = None

        # multi-file stores (batch import + file list)
        self.data_store = {}  # working objects keyed by full path
        self.data_store_raw = {}  # raw objects keyed by full path
        self.current_file_key = None
        self.current_plot_option = 'EIS_data'
        self._eis_selected_raw_indices = []

        # project state (no UI/layout changes)
        self.current_project_path = None
        self._project_dirty = False
        self._project_loading = False
        self._project_io_busy = False

        # fit status tracking
        self.file_meta = {}  # keyed by file path: {fitted: bool, signature: tuple|None}

        # selected run mode (simple/bayesian/BHT)
        self.selected_run_mode = 'simple'

        # DRT comparison settings
        self.drt_comp_settings = {
            'ymin': None,
            'ymax': None,
            'start_color': '#08306B',
            'end_color': '#DEEBF7',
        }
        # DRT map (2D heatmap) settings
        self.drt_map_settings = {
            'vmin': None,  # color scale min (auto if None)
            'vmax': None,  # color scale max (auto if None)
            'major_levels': 10,  # number of major levels (Origin-style)
            'minor_levels': 10,  # number of minor levels between majors (controls smoothness)
            'fill_mode': 'contour',  # 'contour' (Fill to Contour Lines) or 'grid' (Fill to Grid Lines)
        }
        self._drt_map_hover_last = None
        self._suppress_file_select = False

        # DRT comparison selection highlight
        self._drt_comp_lines = []  # list of matplotlib Line2D
        self._drt_comp_selected = None  # selected Line2D or None
        self._drt_comp_hover_last = None  # (key, idx) for hover tooltip

        # Kramers-Kronig (lin-KK) settings cache
        self.kk_settings = {
            'c': 0.85,
            'max_m': 50,
            'fit_type': 'complex',
        }

        # cache default styles for run-select buttons (so we can restore when not selected)
        self._run_btn_default_style = {
            'simple': self.ui.simple_run_button.styleSheet(),
            'bayesian': self.ui.bayesian_button.styleSheet(),
            'BHT': self.ui.HT_button.styleSheet(),
        }

        # linking buttons with various functions
        # import button
        self.ui.import_button.clicked.connect(self.import_files)
        self.ui.induct_choice.currentIndexChanged.connect(self.inductance_callback)  # activated when item change

        # files list interactions (if present in layout)
        if hasattr(self.ui, "files_list"):
            self.ui.files_list.currentRowChanged.connect(self._on_file_selected)
            self.ui.files_list.model().rowsMoved.connect(self._on_files_reordered)
            self.ui.files_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.ui.files_list.customContextMenuRequested.connect(self._show_files_context_menu)
            self.ui.files_list.setItemDelegate(FileListDelegate(self.ui.files_list))

            # enable drag & drop import onto Files list, keep internal move reorder
            try:
                self.ui.files_list.setAcceptDrops(True)
                self.ui.files_list.viewport().setAcceptDrops(True)
                self.ui.files_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
            except Exception:
                pass
            try:
                self._files_drop_filter = ExternalFileDropFilter(self)
                self.ui.files_list.installEventFilter(self._files_drop_filter)
                self.ui.files_list.viewport().installEventFilter(self._files_drop_filter)
            except Exception:
                pass

        # also accept external file drops onto the main window
        try:
            self.setAcceptDrops(True)
        except Exception:
            pass

        # show buttons
        self.ui.show_EIS.clicked.connect(lambda: self.plotting_callback('EIS_data'))
        if hasattr(self.ui, 'show_KK_res'):
            self.ui.show_KK_res.clicked.connect(lambda: self.plotting_callback('KK_residual'))
        self.ui.show_mag.clicked.connect(lambda: self.plotting_callback('Magnitude'))
        self.ui.show_phase.clicked.connect(lambda: self.plotting_callback('Phase'))
        self.ui.show_re.clicked.connect(lambda: self.plotting_callback('Re_data'))
        self.ui.show_im.clicked.connect(lambda: self.plotting_callback('Im_data'))
        self.ui.show_re_res.setText('DRT Residual')
        self.ui.show_re_res.clicked.connect(lambda: self.plotting_callback('DRT_residual'))
        try:
            self.ui.show_im_res.hide()
        except Exception:
            pass
        self.ui.show_DRT.clicked.connect(lambda: self.plotting_callback('DRT_data'))
        self.ui.show_score.clicked.connect(lambda: self.plotting_callback('Score'))
        if hasattr(self.ui, 'show_DRT_comp'):
            self.ui.show_DRT_comp.clicked.connect(lambda: self.plotting_callback('DRT_comparison'))
        if hasattr(self.ui, 'show_DRT_map'):
            self.ui.show_DRT_map.clicked.connect(lambda: self.plotting_callback('DRT_map'))

        # run mode select buttons
        self.ui.simple_run_button.clicked.connect(lambda: self._select_run_mode('simple'))
        self.ui.bayesian_button.clicked.connect(lambda: self._select_run_mode('bayesian'))
        self.ui.HT_button.clicked.connect(lambda: self._select_run_mode('BHT'))

        # fit buttons (added below the run mode selection)
        if hasattr(self.ui, 'fit_one_button'):
            self.ui.fit_one_button.clicked.connect(self.fit_selected_callback)
        if hasattr(self.ui, 'fit_all_button'):
            self.ui.fit_all_button.clicked.connect(self.fit_all_callback)
        if hasattr(self.ui, 'run_kkr_button'):
            self.ui.run_kkr_button.clicked.connect(self.kk_run_callback)

        # thread pool for parallel fitting (Fit All)
        self._fit_pool = QtCore.QThreadPool.globalInstance()
        try:
            # Use all available logical cores (leave 1 core for UI responsiveness).
            max_workers = int(os.cpu_count() or 1)
            self._fit_pool.setMaxThreadCount(max(1, max_workers - 1))
        except Exception:
            pass
        self._fit_all_active = False
        self._fit_all_pending = 0
        self._fit_all_done = 0
        self._fit_all_skipped = 0
        self._fit_all_errors = 0
        self._fit_all_restore = {}

        self.ui.peak_decon_button.clicked.connect(self.peak_analysis_run_callback)

        # default: simple selected
        self._select_run_mode('simple')

        # export result buttons
        self.ui.export_DRT_button.clicked.connect(self.export_DRT)
        self.ui.export_EIS_button.clicked.connect(self.export_EIS)
        self.ui.export_fig_button.clicked.connect(self.export_fig)

        # Existing project controls: connect logic only; do not change button layout/style.
        if hasattr(self.ui, 'open_project_button'):
            self.ui.open_project_button.clicked.connect(self.open_project_callback)
        if hasattr(self.ui, 'save_project_button'):
            self.ui.save_project_button.clicked.connect(self.save_project_callback)
            # Right-click on the existing Save button provides Save As without changing the button.
            self.ui.save_project_button.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.ui.save_project_button.customContextMenuRequested.connect(
                self._show_save_project_context_menu
            )
        if hasattr(self.ui, 'save_project_as_button'):
            self.ui.save_project_as_button.clicked.connect(self.save_project_as_callback)
        self._project_buttons_connected = True

        self._install_project_dirty_tracking()

    def _project_entry_attributes(self):
        # Persist imported data and user-visible analysis results only.
        # Large solver matrices and live C-extension objects are intentionally excluded.
        return (
            'method', 'fit_data_source', 'freq', 'freq_0', 'Z_prime', 'Z_prime_0',
            'Z_double_prime', 'Z_double_prime_0', 'Z_exp', 'Z_exp_0', 'tau', 'tau_fine',
            'visible_keep_raw', 'active_raw_indices', 'mask_manual_raw', 'mask_auto_raw',
            'mask_total_raw', 'mask_settings',
            'kk_valid', 'kk_c', 'kk_max_m', 'kk_fit_type', 'kk_selected_m', 'kk_mu',
            'kk_tau', 'kk_R0', 'kk_R', 'kk_L', 'kk_Z_fit', 'kk_Z_fit_raw',
            'kk_res_re_pct', 'kk_res_im_pct', 'kk_res_re_raw_pct', 'kk_res_im_raw_pct',
            'kk_signature',
            'lambda_value', 'L', 'R', 'out_tau_vec', 'gamma', 'mean', 'upper_bound',
            'lower_bound', 'mu_L_0', 'mu_R_inf', 'mu_gamma_fine_re',
            'mu_gamma_fine_im', 'mu_Z_re', 'mu_Z_im', 'mu_Z_H_re_agm',
            'mu_Z_H_im_agm', 'band_re_agm', 'band_im_agm', 'res_re', 'res_im',
            'res_H_re', 'res_H_im', 'out_scores', 'N_peaks', 'out_gamma_fit', 'df'
        )

    def _entry_to_project_record(self, entry):
        if entry is None:
            return None
        attrs = {}
        for name in self._project_entry_attributes():
            if not hasattr(entry, name):
                continue
            value = getattr(entry, name)
            # Keep only values handled by project_io; no Qt/Matplotlib/solver objects.
            attrs[name] = value
        return {'attrs': attrs}

    def _entry_from_project_record(self, record):
        if not isinstance(record, dict):
            raise ValueError('Invalid EIS entry in project file.')
        try:
            entry = EIS_object.__new__(EIS_object)
        except Exception:
            entry = SimpleNamespace()
        for name, value in (record.get('attrs', {}) or {}).items():
            setattr(entry, name, value)
        if not hasattr(entry, 'method'):
            entry.method = 'none'
        _ensure_mask_state(entry)
        return entry

    def _capture_project_ui_state(self):
        combo_names = (
            'discre_choice', 'data_used_choice', 'induct_choice', 'der_choice',
            'lambda_choice', 'shape_control_choice', 'fit_type_choice', 'peak_method_choice'
        )
        line_names = (
            'reg_param_entry', 'reg_param_entry_2', 'sample_no_entry', 'FWHM_entry',
            'cutoff_entry', 'max_elements_entry', 'peak_num_entry'
        )
        return {
            'combo_indices': {
                name: int(getattr(self.ui, name).currentIndex())
                for name in combo_names if hasattr(self.ui, name)
            },
            'line_texts': {
                name: str(getattr(self.ui, name).text())
                for name in line_names if hasattr(self.ui, name)
            },
        }

    def _restore_project_ui_state(self, ui_state):
        ui_state = ui_state or {}
        for name, index in (ui_state.get('combo_indices', {}) or {}).items():
            widget = getattr(self.ui, name, None)
            if widget is None:
                continue
            blocker = QtCore.QSignalBlocker(widget)
            widget.setCurrentIndex(int(index))
            del blocker
        for name, text in (ui_state.get('line_texts', {}) or {}).items():
            widget = getattr(self.ui, name, None)
            if widget is None:
                continue
            blocker = QtCore.QSignalBlocker(widget)
            widget.setText(str(text))
            del blocker

    def _build_project_state(self):
        file_order = self._get_file_keys_in_ui_order()
        return {
            'data_store': {
                key: self._entry_to_project_record(self.data_store[key])
                for key in file_order if key in self.data_store
            },
            'file_meta': copy.deepcopy(self.file_meta),
            'file_order': list(file_order),
            'current_file_key': self.current_file_key,
            'current_plot_option': self.current_plot_option,
            'selected_run_mode': self.selected_run_mode,
            'drt_comp_settings': copy.deepcopy(self.drt_comp_settings),
            'drt_map_settings': copy.deepcopy(self.drt_map_settings),
            'kk_settings': copy.deepcopy(self.kk_settings),
            'ui_state': self._capture_project_ui_state(),
        }

    def _restore_project_state(self, state):
        if not isinstance(state, dict):
            raise ValueError('Invalid project state.')

        self._project_loading = True
        try:
            records = state.get('data_store', {}) or {}
            restored = {key: self._entry_from_project_record(rec) for key, rec in records.items()}
            self.data_store = restored
            # data_store_raw is only used as an imported-path membership cache in this GUI.
            self.data_store_raw = {key: None for key in restored}
            self.file_meta = copy.deepcopy(state.get('file_meta', {}) or {})
            for key, entry in restored.items():
                self.file_meta.setdefault(
                    key,
                    {'fitted': getattr(entry, 'method', 'none') != 'none', 'signature': None}
                )

            self.drt_comp_settings = copy.deepcopy(
                state.get('drt_comp_settings', self.drt_comp_settings) or self.drt_comp_settings
            )
            self.drt_map_settings = copy.deepcopy(
                state.get('drt_map_settings', self.drt_map_settings) or self.drt_map_settings
            )
            self.kk_settings = copy.deepcopy(
                state.get('kk_settings', self.kk_settings) or self.kk_settings
            )
            self.selected_run_mode = state.get('selected_run_mode', 'simple')
            self.current_plot_option = state.get('current_plot_option', 'EIS_data')
            self._restore_project_ui_state(state.get('ui_state', {}))

            order = [key for key in (state.get('file_order', []) or []) if key in restored]
            order.extend(key for key in restored if key not in order)
            target = state.get('current_file_key')
            if target not in restored:
                target = order[0] if order else None

            self.current_file_key = target
            self.data = restored.get(target) if target is not None else None
            self._eis_selected_raw_indices = []

            if hasattr(self.ui, 'files_list'):
                blocker = QtCore.QSignalBlocker(self.ui.files_list)
                self.ui.files_list.clear()
                selected_row = -1
                for key in order:
                    item = QtWidgets.QListWidgetItem(os.path.basename(key))
                    item.setToolTip(key)
                    item.setData(QtCore.Qt.UserRole, key)
                    fitted = bool(self.file_meta.get(key, {}).get('fitted', False))
                    item.setData(FileListDelegate.STATUS_ROLE, 'Fitted' if fitted else 'Unfitted')
                    self.ui.files_list.addItem(item)
                    if key == target:
                        selected_row = self.ui.files_list.count() - 1
                if selected_row >= 0:
                    self.ui.files_list.setCurrentRow(selected_row)
                del blocker

            self._select_run_mode(self.selected_run_mode)
        finally:
            self._project_loading = False

        self.plotting_callback(self.current_plot_option)

    def _install_project_dirty_tracking(self):
        combo_names = (
            'discre_choice', 'data_used_choice', 'induct_choice', 'der_choice',
            'lambda_choice', 'shape_control_choice', 'fit_type_choice', 'peak_method_choice'
        )
        line_names = (
            'reg_param_entry', 'sample_no_entry', 'FWHM_entry',
            'cutoff_entry', 'max_elements_entry', 'peak_num_entry'
        )
        for name in combo_names:
            widget = getattr(self.ui, name, None)
            if widget is not None:
                widget.currentIndexChanged.connect(self._mark_project_dirty)
        for name in line_names:
            widget = getattr(self.ui, name, None)
            if widget is not None:
                widget.textEdited.connect(self._mark_project_dirty)
        for name in ('simple_run_button', 'bayesian_button', 'HT_button'):
            button = getattr(self.ui, name, None)
            if button is not None:
                button.clicked.connect(self._mark_project_dirty)

    def _mark_project_dirty(self, *args):
        if self._project_loading:
            return
        if not self.data_store and not self.current_project_path:
            return
        self._project_dirty = True
        self._update_project_window_title()

    def _set_project_clean(self):
        self._project_dirty = False
        self._update_project_window_title()

    def _show_save_project_context_menu(self, pos):
        # Save As is available without changing the existing Save button.
        menu = QtWidgets.QMenu(self)
        action = menu.addAction('Save Project As...')
        action.triggered.connect(self.save_project_as_callback)
        button = getattr(self.ui, 'save_project_button', None)
        if button is not None:
            menu.exec_(button.mapToGlobal(pos))

    def _import_paths_into_store(self, paths):
        """Import a list of EIS text file paths into the multi-file stores."""
        newly_added = []
        for path in paths:
            if path in self.data_store_raw:
                continue

            try:
                raw_obj = EIS_object.from_file(path)
            except Exception as e:
                try:
                    self.statusBar().showMessage(f'Failed to import: {path} ({e})', 3000)
                except Exception:
                    pass
                continue

            self.data_store_raw[path] = raw_obj
            self.data_store[path] = copy.deepcopy(raw_obj)
            newly_added.append(path)
            self.file_meta[path] = {'fitted': False, 'signature': None}

            if hasattr(self.ui, "files_list"):
                item = QtWidgets.QListWidgetItem(os.path.basename(path))
                item.setToolTip(path)
                item.setData(QtCore.Qt.UserRole, path)
                item.setData(FileListDelegate.STATUS_ROLE, 'Unfitted')
                self.ui.files_list.addItem(item)

        if newly_added:
            self._mark_project_dirty()
        return newly_added

    def import_file(self):
        """Backward-compatible single-file import (calls batch importer)."""
        self.import_files(single=True)

    def import_files(self, single: bool = False):
        """Batch import of EIS text files, tracked in the right-side Files list."""
        file_filter = "All supported EIS files (*);;CSV files (*.csv);;TXT files (*.txt);;All Files (*)"

        if single:
            path, _ = QFileDialog.getOpenFileName(None, "Please choose a file", "", file_filter)
            paths = [path] if path else []
        else:
            paths, _ = QFileDialog.getOpenFileNames(None, "Please choose file(s)", "", file_filter)

        # keep local files; parser handles csv/txt/extensionless EIS text files.
        paths = [p for p in paths if p and os.path.isfile(p)]
        if not paths:
            return

        newly_added = self._import_paths_into_store(paths)

        if not newly_added:
            return

        # apply inductance choice to newly imported data
        for key in newly_added:
            self._apply_inductance_to_key(key, reset_method=True)

        # select first file if none selected yet
        if self.current_file_key is None:
            self._set_current_file(newly_added[0])
            if hasattr(self.ui, "files_list"):
                # select matching row
                for i in range(self.ui.files_list.count()):
                    it = self.ui.files_list.item(i)
                    if it.data(QtCore.Qt.UserRole) == newly_added[0]:
                        self.ui.files_list.setCurrentRow(i)
                        break

        self.statusBar().showMessage('Imported: %d file(s)' % len(newly_added), 1500)

    def import_files_from_paths(self, paths):
        """Import files from an explicit list of paths (used for drag & drop)."""
        # keep local files; parser handles csv/txt/extensionless EIS text files.
        paths = [p for p in (paths or []) if p and os.path.isfile(p)]
        if not paths:
            return

        newly_added = self._import_paths_into_store(paths)

        if not newly_added:
            return

        # apply inductance choice to newly imported data
        for key in newly_added:
            self._apply_inductance_to_key(key, reset_method=True)

        # select first file if none selected yet
        if self.current_file_key is None:
            self._set_current_file(newly_added[0])
            if hasattr(self.ui, "files_list"):
                for i in range(self.ui.files_list.count()):
                    it = self.ui.files_list.item(i)
                    if it.data(QtCore.Qt.UserRole) == newly_added[0]:
                        self.ui.files_list.setCurrentRow(i)
                        break

        try:
            self.statusBar().showMessage('Imported: %d file(s)' % len(newly_added), 1500)
        except Exception:
            pass

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Accept dropping EIS text files onto the main window."""
        try:
            md = event.mimeData()
            if md is not None and md.hasUrls():
                for u in md.urls():
                    if u.isLocalFile():
                        p = u.toLocalFile()
                        if p and os.path.isfile(p):
                            event.acceptProposedAction()
                            return
        except Exception:
            pass
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        """Handle dropped EIS text files (append to Files list)."""
        paths = []
        try:
            md = event.mimeData()
            if md is not None and md.hasUrls():
                for u in md.urls():
                    if u.isLocalFile():
                        p = u.toLocalFile()
                        if p and os.path.isfile(p):
                            paths.append(p)
        except Exception:
            paths = []

        if paths:
            try:
                self.import_files_from_paths(paths)
            except Exception:
                pass
            event.acceptProposedAction()
        else:
            event.ignore()

    def _apply_inductance_to_key(self, key: str, reset_method: bool = True, clear_kk: bool = True):
        """Rebuild one file's working data after inductance/mask changes."""
        if key not in self.data_store:
            return

        obj = self.data_store[key]
        _ensure_mask_state(obj)
        obj = _rebuild_visible_data_from_raw(obj, int(self.ui.induct_choice.currentIndex()))

        if reset_method:
            obj.method = 'none'
        if clear_kk:
            _clear_kk_results(obj)
        else:
            _refresh_kk_current_arrays(obj)

        if reset_method and key in getattr(self, 'file_meta', {}):
            self.file_meta[key] = {'fitted': False, 'signature': None}
            self._update_file_status(key, fitted=False)

        self.data_store[key] = obj

    def _rebuild_after_mask_change(self, key: str, refresh_plot: bool = True):
        """Rebuild one dataset after mask changes and optionally refresh the EIS plot."""
        if key not in getattr(self, 'data_store', {}):
            return
        self._apply_inductance_to_key(key, reset_method=True, clear_kk=False)
        if key == getattr(self, 'current_file_key', None):
            self.data = self.data_store[key]
        if refresh_plot:
            self.plotting_callback('EIS_data')
        self._mark_project_dirty()

    def _toggle_mask_point(self, key: str, raw_index: int, masked: bool):
        entry = self.data_store.get(key)
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        if 0 <= int(raw_index) < entry.mask_manual_raw.size:
            entry.mask_manual_raw[int(raw_index)] = bool(masked)
        self.data_store[key] = entry
        self._eis_selected_raw_indices = []
        self._rebuild_after_mask_change(key)

    def _set_mask_for_raw_indices(self, key: str, raw_indices, masked: bool):
        entry = self.data_store.get(key)
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        idx = np.asarray(list(raw_indices or []), dtype=int).reshape(-1)
        if idx.size:
            idx = idx[(idx >= 0) & (idx < entry.mask_manual_raw.size)]
            if idx.size:
                entry.mask_manual_raw[idx] = bool(masked)
        self.data_store[key] = entry
        self._eis_selected_raw_indices = []
        self._rebuild_after_mask_change(key)

    def _clear_masks_for_key(self, key: str, clear_manual: bool = True, clear_auto: bool = True):
        entry = self.data_store.get(key)
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        if clear_manual:
            entry.mask_manual_raw[:] = False
        if clear_auto:
            entry.mask_auto_raw[:] = False
        self.data_store[key] = entry
        self._eis_selected_raw_indices = []
        self._rebuild_after_mask_change(key)

    @staticmethod
    def _build_auto_mask_from_settings(entry, freq_min=None, freq_max=None, kk_threshold=None):
        """Build an automatic mask from frequency and/or K-K residual criteria."""
        entry = _ensure_mask_state(entry)
        freq0, _, _, _ = _get_entry_raw_arrays(entry)
        auto_mask = np.zeros(freq0.size, dtype=bool)

        if freq_min is not None or freq_max is not None:
            f_lo = float(freq_min) if freq_min is not None else None
            f_hi = float(freq_max) if freq_max is not None else None
            if f_lo is not None and f_hi is not None and f_lo > f_hi:
                f_lo, f_hi = f_hi, f_lo
            freq_mask = np.ones(freq0.size, dtype=bool)
            if f_lo is not None:
                freq_mask &= freq0 >= f_lo
            if f_hi is not None:
                freq_mask &= freq0 <= f_hi
            auto_mask |= freq_mask

        if kk_threshold is not None:
            kk_re = np.asarray(getattr(entry, 'kk_res_re_raw_pct', np.full(freq0.size, np.nan)), dtype=float)
            kk_im = np.asarray(getattr(entry, 'kk_res_im_raw_pct', np.full(freq0.size, np.nan)), dtype=float)
            if kk_re.size != freq0.size or kk_im.size != freq0.size:
                raise ValueError('Current K-K residuals are unavailable for threshold masking.')
            kk_max = _safe_kk_residual_max(kk_re, kk_im, freq0.size)
            auto_mask |= np.isfinite(kk_max) & (kk_max > float(kk_threshold))

        return auto_mask

    def _apply_mask_settings_to_keys(self, keys, freq_min=None, freq_max=None, kk_threshold=None):
        """Apply automatic mask settings to one or more files."""
        valid_keys = [k for k in (keys or []) if k in self.data_store]
        if not valid_keys:
            return

        for key in valid_keys:
            entry = self.data_store.get(key)
            if entry is None:
                continue
            entry = _ensure_mask_state(entry)
            entry.mask_auto_raw = self._build_auto_mask_from_settings(
                entry, freq_min=freq_min, freq_max=freq_max, kk_threshold=kk_threshold
            )
            entry.mask_settings = {'freq_min': freq_min, 'freq_max': freq_max, 'kk_threshold': kk_threshold}
            self.data_store[key] = entry
            self._rebuild_after_mask_change(key, refresh_plot=False)

        current_key = getattr(self, 'current_file_key', None)
        if current_key in valid_keys:
            self.data = self.data_store.get(current_key)
            self.plotting_callback('EIS_data')

    def _clear_auto_mask_for_keys(self, keys):
        """Clear automatic masks for one or more files."""
        valid_keys = [k for k in (keys or []) if k in self.data_store]
        for key in valid_keys:
            entry = self.data_store.get(key)
            if entry is None:
                continue
            entry = _ensure_mask_state(entry)
            entry.mask_auto_raw[:] = False
            entry.mask_settings = {'freq_min': None, 'freq_max': None, 'kk_threshold': None}
            self.data_store[key] = entry
            self._rebuild_after_mask_change(key, refresh_plot=False)

        current_key = getattr(self, 'current_file_key', None)
        if current_key in valid_keys:
            self.data = self.data_store.get(current_key)
            self.plotting_callback('EIS_data')

    def _apply_mask_settings_to_current(self, freq_min=None, freq_max=None, kk_threshold=None):
        """Backward-compatible wrapper: apply mask settings only to the current file."""
        key = getattr(self, 'current_file_key', None)
        if key is None:
            return
        self._apply_mask_settings_to_keys([key], freq_min=freq_min, freq_max=freq_max, kk_threshold=kk_threshold)

    def _open_mask_setting_dialog(self):
        """Open a clearer mask-settings dialog for the current file or all files."""
        key = getattr(self, 'current_file_key', None)
        entry = self.data_store.get(key) if key else None
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        settings = getattr(entry, 'mask_settings', {}) or {}

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Mask Settings')
        dlg.setModal(True)
        dlg.resize(560, 260)

        root = QtWidgets.QVBoxLayout(dlg)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QtWidgets.QLabel('Mask points by frequency range and/or K-K residual threshold.', dlg)
        title.setWordWrap(True)
        title.setStyleSheet('font-weight: 600;')
        root.addWidget(title)

        subtitle = QtWidgets.QLabel(
            'Apply affects the current file only. Use "Apply to All Files" to apply the same criteria to every imported file.',
            dlg,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet('color: #6E6E73;')
        root.addWidget(subtitle)

        form_widget = QtWidgets.QWidget(dlg)
        form = QtWidgets.QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        fmin_edit = QtWidgets.QLineEdit(dlg)
        fmax_edit = QtWidgets.QLineEdit(dlg)
        thr_edit = QtWidgets.QLineEdit(dlg)
        for edit in (fmin_edit, fmax_edit, thr_edit):
            edit.setPlaceholderText('Optional')
            edit.setMinimumHeight(30)

        if settings.get('freq_min') is not None:
            fmin_edit.setText(str(settings.get('freq_min')))
        if settings.get('freq_max') is not None:
            fmax_edit.setText(str(settings.get('freq_max')))
        if settings.get('kk_threshold') is not None:
            thr_edit.setText(str(settings.get('kk_threshold')))

        form.addRow('Frequency min (Hz)', fmin_edit)
        form.addRow('Frequency max (Hz)', fmax_edit)
        form.addRow('K-K residual > (%)', thr_edit)
        root.addWidget(form_widget)

        button_grid = QtWidgets.QGridLayout()
        button_grid.setHorizontalSpacing(8)
        button_grid.setVerticalSpacing(8)

        clear_auto_btn = QtWidgets.QPushButton('Clear Current File Auto Mask', dlg)
        clear_auto_all_btn = QtWidgets.QPushButton('Clear All Files Auto Masks', dlg)
        apply_btn = QtWidgets.QPushButton('Apply to This File', dlg)
        apply_all_btn = QtWidgets.QPushButton('Apply to All Files', dlg)

        apply_btn.setDefault(True)
        apply_btn.setAutoDefault(True)

        # 让四个按钮宽度一致，上下更整齐
        for btn in (clear_auto_btn, clear_auto_all_btn, apply_btn, apply_all_btn):
            btn.setMinimumWidth(170)
            btn.setMinimumHeight(32)

        # 两排两列，上下对齐
        button_grid.addWidget(clear_auto_btn, 0, 0)
        button_grid.addWidget(clear_auto_all_btn, 0, 1)
        button_grid.addWidget(apply_btn, 1, 0)
        button_grid.addWidget(apply_all_btn, 1, 1)

        root.addLayout(button_grid)

        def _parse_optional(edit):
            s = (edit.text() or '').strip()
            return None if s == '' else float(s)

        def _read_settings(target_keys):
            try:
                fmin = _parse_optional(fmin_edit)
                fmax = _parse_optional(fmax_edit)
                thr = _parse_optional(thr_edit)
                if thr is not None:
                    missing = []
                    for target_key in target_keys:
                        target_entry = self.data_store.get(target_key)
                        if target_entry is None or not getattr(target_entry, 'kk_valid', False):
                            missing.append(os.path.basename(target_key))
                    if missing:
                        raise ValueError(
                            'Please run K-K validation first for: ' + ', '.join(missing[:5]) + (' ...' if len(missing) > 5 else '')
                        )
                return fmin, fmax, thr
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Invalid mask setting', str(e))
                return None

        def _apply_current():
            values = _read_settings([key])
            if values is None:
                return
            fmin, fmax, thr = values
            self._apply_mask_settings_to_current(freq_min=fmin, freq_max=fmax, kk_threshold=thr)
            dlg.accept()

        def _apply_all():
            target_keys = self._get_file_keys_in_ui_order()
            values = _read_settings(target_keys)
            if values is None:
                return
            fmin, fmax, thr = values
            self._apply_mask_settings_to_keys(target_keys, freq_min=fmin, freq_max=fmax, kk_threshold=thr)
            dlg.accept()

        def _clear_current():
            self._clear_auto_mask_for_keys([key])
            dlg.accept()

        def _clear_all():
            self._clear_auto_mask_for_keys(self._get_file_keys_in_ui_order())
            dlg.accept()

        apply_btn.clicked.connect(_apply_current)
        apply_all_btn.clicked.connect(_apply_all)
        clear_auto_btn.clicked.connect(_clear_current)
        clear_auto_all_btn.clicked.connect(_clear_all)
        dlg.exec_()

    def _find_eis_raw_index_from_event(self, fig, event, threshold_px: float = 10.0):
        payload = getattr(fig, '_eis_plot_payload', None)
        if not payload or getattr(event, 'inaxes', None) is not fig.axes:
            return None
        xs = np.asarray(payload.get('x', []), dtype=float)
        ys = np.asarray(payload.get('y', []), dtype=float)
        raw_indices = np.asarray(payload.get('raw_indices', []), dtype=int)
        if xs.size == 0 or ys.size == 0 or raw_indices.size != xs.size:
            return None
        try:
            pts = fig.axes.transData.transform(np.column_stack([xs, ys]))
        except Exception:
            return None
        dist2 = (pts[:, 0] - event.x) ** 2 + (pts[:, 1] - event.y) ** 2
        j = int(np.argmin(dist2))
        if float(np.sqrt(dist2[j])) > float(threshold_px):
            return None
        return int(raw_indices[j])

    def _apply_eis_selection_highlight(self, fig):
        if fig is None:
            return

        ax = fig.axes
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
        old_autoscale = ax.get_autoscale_on()

        for artist in getattr(fig, '_eis_selection_artists', []):
            try:
                artist.remove()
            except Exception:
                pass
        fig._eis_selection_artists = []

        payload = getattr(fig, '_eis_plot_payload', None)
        if payload:
            selected = np.asarray(getattr(self, '_eis_selected_raw_indices', []), dtype=int).reshape(-1)
            raw_indices = np.asarray(payload.get('raw_indices', []), dtype=int).reshape(-1)
            xs = np.asarray(payload.get('x', []), dtype=float).reshape(-1)
            ys = np.asarray(payload.get('y', []), dtype=float).reshape(-1)

            if selected.size > 0 and raw_indices.size > 0 and xs.size == raw_indices.size and ys.size == raw_indices.size:
                keep = np.isin(raw_indices, selected)
                if np.any(keep):
                    artist = ax.scatter(
                        xs[keep], ys[keep],
                        s=95, facecolors='none',
                        edgecolors='#00B5FF',
                        linewidths=1.5, zorder=30
                    )
                    fig._eis_selection_artists = [artist]

        ax.set_autoscale_on(False)
        ax.set_xlim(old_xlim)
        ax.set_ylim(old_ylim)

        try:
            fig.draw_idle()
        except Exception:
            fig.draw()

        ax.set_autoscale_on(old_autoscale)

    def _on_eis_box_select(self, eclick, erelease):
        fig = getattr(self, '_eis_canvas', None)
        if fig is None or getattr(fig, '_eis_plot_payload', None) is None:
            return
        if eclick is None or erelease is None:
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return

        payload = fig._eis_plot_payload
        raw_indices = np.asarray(payload.get('raw_indices', []), dtype=int).reshape(-1)
        xs = np.asarray(payload.get('x', []), dtype=float).reshape(-1)
        ys = np.asarray(payload.get('y', []), dtype=float).reshape(-1)
        if raw_indices.size == 0 or xs.size != raw_indices.size or ys.size != raw_indices.size:
            return

        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        self._eis_selected_raw_indices = raw_indices[keep].astype(int).tolist()
        self._apply_eis_selection_highlight(fig)

    def _clear_eis_selection(self):
        self._eis_selected_raw_indices = []
        fig = getattr(self, '_eis_canvas', None)
        if fig is not None:
            self._apply_eis_selection_highlight(fig)

    def _show_eis_context_menu(self, fig, event):
        key = getattr(self, 'current_file_key', None)
        entry = self.data_store.get(key) if key else None
        if entry is None:
            return
        entry = _ensure_mask_state(entry)

        raw_index = self._find_eis_raw_index_from_event(fig, event)
        menu = QtWidgets.QMenu(self)

        selected = np.asarray(getattr(self, '_eis_selected_raw_indices', []), dtype=int).reshape(-1)
        if selected.size:
            selected = selected[(selected >= 0) & (selected < entry.mask_total_raw.size)]
            if selected.size:
                any_masked = bool(np.any(entry.mask_total_raw[selected]))
                any_unmasked = bool(np.any(~entry.mask_total_raw[selected]))

                if any_unmasked:
                    act_mask_sel = QtWidgets.QAction(f'Mask selected points ({selected.size})', self)
                    act_mask_sel.triggered.connect(lambda: self._set_mask_for_raw_indices(key, selected.tolist(), True))
                    menu.addAction(act_mask_sel)
                if any_masked:
                    act_unmask_sel = QtWidgets.QAction(f'Unmask selected points ({selected.size})', self)
                    act_unmask_sel.triggered.connect(lambda: self._set_mask_for_raw_indices(key, selected.tolist(), False))
                    menu.addAction(act_unmask_sel)

                act_clear_sel = QtWidgets.QAction('Clear selection', self)
                act_clear_sel.triggered.connect(self._clear_eis_selection)
                menu.addAction(act_clear_sel)
                menu.addSeparator()

        if raw_index is not None and selected.size == 0:
            is_masked = bool(entry.mask_total_raw[int(raw_index)])
            act_toggle = QtWidgets.QAction('Unmask point' if is_masked else 'Mask point', self)
            act_toggle.triggered.connect(lambda: self._toggle_mask_point(key, raw_index, not is_masked))
            menu.addAction(act_toggle)
            menu.addSeparator()

        act_setting = QtWidgets.QAction('Mask setting', self)
        act_setting.triggered.connect(self._open_mask_setting_dialog)
        menu.addAction(act_setting)

        act_clear = QtWidgets.QAction('Clear all masks', self)
        act_clear.triggered.connect(lambda: self._clear_masks_for_key(key, clear_manual=True, clear_auto=True))
        menu.addAction(act_clear)

        try:
            menu.exec_(QtGui.QCursor.pos())
        except Exception:
            menu.exec_()

    def _on_eis_button_press(self, event):
        fig = getattr(self, '_eis_canvas', None)
        if fig is None:
            return
        if getattr(event, 'button', None) == 3:
            if getattr(event, 'inaxes', None) is fig.axes:
                self._show_eis_context_menu(fig, event)
            return
        if getattr(event, 'button', None) == 1 and getattr(event, 'inaxes', None) is fig.axes:
            raw_index = self._find_eis_raw_index_from_event(fig, event)
            if raw_index is None and len(getattr(self, '_eis_selected_raw_indices', [])) > 0:
                self._clear_eis_selection()

    def _set_current_file(self, key: str):
        if key not in self.data_store:
            return
        self.current_file_key = key
        self.data = self.data_store[key]
        self._eis_selected_raw_indices = []
        # keep current plot selection across switching
        self.plotting_callback(self.current_plot_option)

    def _select_drt_comparison_line_by_key(self, key):
        selected_line = None
        for line in getattr(self, '_drt_comp_lines', []) or []:
            try:
                if line.get_gid() == key:
                    selected_line = line
                    break
            except Exception:
                continue
        if selected_line is None:
            self._clear_drt_comp_selection()
        else:
            self._apply_drt_comp_selection(selected_line)

    def _on_file_selected(self, row: int):
        if row < 0 or not hasattr(self.ui, "files_list"):
            return
        item = self.ui.files_list.item(row)
        if not item:
            return
        key = item.data(QtCore.Qt.UserRole)
        if key is None:
            return
        if getattr(self, '_suppress_file_select', False):
            # Selection triggered from DRT comparison line click: update current file without replot.
            self.current_file_key = key
            if key in getattr(self, 'data_store', {}):
                self.data = self.data_store[key]
            self._suppress_file_select = False
            return

        if getattr(self, 'current_plot_option', None) == 'DRT_comparison':
            self.current_file_key = key
            self.data = self.data_store.get(key)
            self._eis_selected_raw_indices = []
            self._select_drt_comparison_line_by_key(key)
            return

        self._set_current_file(key)

    def _on_files_reordered(self, *args, **kwargs):
        self._mark_project_dirty()
        # Order affects DRT comparison colors AND DRT map row order.
        if getattr(self, 'current_plot_option', None) in ('DRT_comparison', 'DRT_map'):
            self.plotting_callback(self.current_plot_option)
        return

    def _show_files_context_menu(self, pos: QtCore.QPoint):
        if not hasattr(self.ui, "files_list"):
            return

        menu = QtWidgets.QMenu(self)
        item = self.ui.files_list.itemAt(pos)

        act_clear_current = QtWidgets.QAction("Clear", self)
        act_clear_all = QtWidgets.QAction("Clear all files", self)

        if item is None:
            act_clear_current.setEnabled(False)

        menu.addAction(act_clear_current)
        menu.addSeparator()
        menu.addAction(act_clear_all)

        menu.setStyleSheet("""
            QMenu { background:#FFFFFF;color:#1D1D1F;border:0px solid #E5E5EA;border-radius:10px; }
            QMenu::item { color:#1D1D1F;padding:6px 10px;min-height:28px; }
        """)

        def _clear_current():
            if item is None:
                return
            key = item.data(QtCore.Qt.UserRole)
            self._remove_file_by_key(key)

        def _clear_all():
            self._remove_all_files()

        act_clear_current.triggered.connect(_clear_current)
        act_clear_all.triggered.connect(_clear_all)

        menu.exec_(self.ui.files_list.mapToGlobal(pos))

    def _remove_file_by_key(self, key: str):
        if not key:
            return

        # remove from stores
        self.data_store.pop(key, None)
        self.data_store_raw.pop(key, None)
        if hasattr(self, 'file_meta'):
            self.file_meta.pop(key, None)

        # remove from UI list
        if hasattr(self.ui, "files_list"):
            for i in range(self.ui.files_list.count()):
                it = self.ui.files_list.item(i)
                if it.data(QtCore.Qt.UserRole) == key:
                    self.ui.files_list.takeItem(i)
                    break

        # update current selection
        self._mark_project_dirty()

        if self.current_file_key == key:
            self.current_file_key = None
            self.data = None

            if hasattr(self.ui, "files_list") and self.ui.files_list.count() > 0:
                self.ui.files_list.setCurrentRow(0)  # triggers selection callback
            else:
                self.plotting_callback(self.current_plot_option)  # clears

    def _remove_all_files(self):
        self.data_store.clear()
        self.data_store_raw.clear()
        if hasattr(self, 'file_meta'):
            self.file_meta.clear()
        self.current_file_key = None
        self.data = None
        if hasattr(self.ui, "files_list"):
            self.ui.files_list.clear()
        self._mark_project_dirty()
        self.plotting_callback(self.current_plot_option)  # clears

    def _update_run_select_button_styles(self) -> None:
        """Blue background + white text for the selected run mode button."""
        selected_override = """
            QPushButton {
                background-color: #0A84FF;
                color: #FFFFFF;
            }
            QPushButton:hover { background-color: #0A84FF; }
            QPushButton:pressed { background-color: #0A84FF; }
        """

        # 先恢复默认样式
        self.ui.simple_run_button.setStyleSheet(self._run_btn_default_style.get('simple', ''))
        self.ui.bayesian_button.setStyleSheet(self._run_btn_default_style.get('bayesian', ''))
        self.ui.HT_button.setStyleSheet(self._run_btn_default_style.get('BHT', ''))

        height = 30
        # ✅ 恢复后立刻锁高度（避免恢复默认也触发变高）
        self.ui.simple_run_button.setFixedHeight(height)
        self.ui.bayesian_button.setFixedHeight(height)
        self.ui.HT_button.setFixedHeight(height)

        # 再给选中的叠加颜色（在默认基础上叠加，不覆盖其它）
        if self.selected_run_mode == 'simple':
            base = self._run_btn_default_style.get('simple', '')
            self.ui.simple_run_button.setStyleSheet(base + "\n" + selected_override)
            self.ui.simple_run_button.setFixedHeight(height)

        elif self.selected_run_mode == 'bayesian':
            base = self._run_btn_default_style.get('bayesian', '')
            self.ui.bayesian_button.setStyleSheet(base + "\n" + selected_override)
            self.ui.bayesian_button.setFixedHeight(height)

        elif self.selected_run_mode == 'BHT':
            base = self._run_btn_default_style.get('BHT', '')
            self.ui.HT_button.setStyleSheet(base + "\n" + selected_override)
            self.ui.HT_button.setFixedHeight(height)

    def _select_run_mode(self, mode: str):
        """Select which run mode will be used by Fit One / Fit All."""
        self.selected_run_mode = mode

        # Update button texts (keep size/alignment unchanged)
        try:
            self.ui.simple_run_button.setText('Selected' if mode == 'simple' else 'Select')
            self.ui.bayesian_button.setText('Selected' if mode == 'bayesian' else 'Select')
            self.ui.HT_button.setText('Selected' if mode == 'BHT' else 'Select')
        except Exception:
            pass

        self._update_run_select_button_styles()

    def _confirm_raw_fit_if_needed(self, keys, mode: str) -> bool:
        """Allow fitting without showing a K-K validation warning dialog.

        DRT fitting uses the current raw/masked experimental EIS data, not the
        K-K fitted curve, so K-K validation is treated as an optional quality
        check rather than a mandatory gate for fitting.
        """
        return True

    def _current_fit_signature(self, mode: str):
        """Build a signature of the current UI options + selected mode to support skip logic."""
        induct_used = int(self.ui.induct_choice.currentIndex())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())

        if mode == 'simple' or mode == 'bayesian':
            rbf_type = str(self.ui.discre_choice.currentText())
            data_used = str(self.ui.data_used_choice.currentText())
            der_used = str(self.ui.der_choice.currentText())
            cv_type = str(self.ui.lambda_choice.currentText())
            reg_param = float(self.ui.reg_param_entry.text())
            sig = ('mode', mode,
                   'rbf', rbf_type,
                   'data_used', data_used,
                   'induct', induct_used,
                   'der', der_used,
                   'cv', cv_type,
                   'reg', reg_param,
                   'shape', shape_control,
                   'coeff', coeff)
            if mode == 'bayesian':
                sample_number = int(self.ui.sample_no_entry.text())
                sig = sig + ('samples', sample_number)
            return sig

        if mode == 'BHT':
            rbf_type = str(self.ui.discre_choice.currentText())
            der_used = str(self.ui.der_choice.currentText())
            return ('mode', mode,
                    'rbf', rbf_type,
                    'induct', induct_used,
                    'der', der_used,
                    'shape', shape_control,
                    'coeff', coeff)

        return ('mode', mode)

    def _update_file_status(self, key: str, fitted: bool):
        """Update the right-side status label in the Files list."""
        if not hasattr(self.ui, 'files_list') or not key:
            return
        status_text = 'Fitted' if fitted else 'Unfitted'
        for i in range(self.ui.files_list.count()):
            it = self.ui.files_list.item(i)
            if it and it.data(QtCore.Qt.UserRole) == key:
                it.setData(FileListDelegate.STATUS_ROLE, status_text)
                # repaint without disturbing selection
                self.ui.files_list.viewport().update()
                break

    def _fit_key(self, key: str, mode: str, signature=None, update_ui: bool = False):
        """Run the selected mode on a specific file (by key) and store results."""
        if key not in self.data_store:
            return

        entry = self.data_store[key]
        signature = signature if signature is not None else self._current_fit_signature(mode)

        if mode == 'simple':
            source_entry = entry
            entry, used_kk = _prepare_entry_for_drt_fit(entry)
            rbf_type = str(self.ui.discre_choice.currentText())
            data_used = str(self.ui.data_used_choice.currentText())
            induct_used = int(self.ui.induct_choice.currentIndex())
            der_used = str(self.ui.der_choice.currentText())
            cv_type = str(self.ui.lambda_choice.currentText())
            reg_param = float(self.ui.reg_param_entry.text())
            shape_control = str(self.ui.shape_control_choice.currentText())
            coeff = float(self.ui.FWHM_entry.text())

            entry = simple_run(entry, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                               der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                               shape_control=shape_control, coeff=coeff)

            if cv_type == 'custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im,
                                                           entry.M, data_used, induct_used, -3, cv_type)
            entry = _finalize_fitted_entry_for_display(entry, source_entry, used_kk)
            if update_ui:
                self.ui.reg_param_entry_2.setText(str(entry.lambda_value))

        elif mode == 'bayesian':
            source_entry = entry
            entry, used_kk = _prepare_entry_for_drt_fit(entry)
            rbf_type = str(self.ui.discre_choice.currentText())
            data_used = str(self.ui.data_used_choice.currentText())
            induct_used = int(self.ui.induct_choice.currentIndex())
            der_used = str(self.ui.der_choice.currentText())
            cv_type = str(self.ui.lambda_choice.currentText())
            reg_param = float(self.ui.reg_param_entry.text())
            shape_control = str(self.ui.shape_control_choice.currentText())
            coeff = float(self.ui.FWHM_entry.text())
            sample_number = int(self.ui.sample_no_entry.text())

            entry = Bayesian_run(entry, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                 der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                                 shape_control=shape_control, coeff=coeff, NMC_sample=sample_number)
            entry = _finalize_fitted_entry_for_display(entry, source_entry, used_kk)

        elif mode == 'BHT':
            rbf_type = str(self.ui.discre_choice.currentText())
            der_used = str(self.ui.der_choice.currentText())
            shape_control = str(self.ui.shape_control_choice.currentText())
            coeff = float(self.ui.FWHM_entry.text())

            entry = BHT_run(entry, rbf_type, der_used, shape_control, coeff)

        else:
            return

        self.data_store[key] = entry
        if key == getattr(self, 'current_file_key', None):
            self.data = entry

        if key in getattr(self, 'file_meta', {}):
            self.file_meta[key] = {'fitted': True, 'signature': signature}
        self._update_file_status(key, fitted=True)
        self._mark_project_dirty()

    def fit_selected_callback(self):
        """Fit only the currently selected file."""
        if getattr(self, 'current_file_key', None) is None or self.data is None:
            return

        mode = getattr(self, 'selected_run_mode', 'simple')
        if not self._confirm_raw_fit_if_needed([self.current_file_key], mode):
            return
        self._fit_key(self.current_file_key, mode, update_ui=True)
        self.plotting_callback(
            self.current_plot_option if getattr(self, 'current_plot_option', None) in ('DRT_comparison',
                                                                                       'DRT_map') else 'DRT_data')

    def fit_all_callback(self):
        """Fit all files in the list in parallel; skip those already fitted with the same signature."""
        if not getattr(self, 'data_store', None):
            return

        # prevent overlapping Fit All runs
        if getattr(self, '_fit_all_active', False):
            try:
                self.statusBar().showMessage('Fit all is running...', 1500)
            except Exception:
                pass
            return

        mode = getattr(self, 'selected_run_mode', 'simple')

        # snapshot signature + parameters once (UI must only be read in main thread)
        try:
            signature = self._current_fit_signature(mode)
            params = self._build_fit_params(mode)
        except Exception as e:
            try:
                QtWidgets.QMessageBox.warning(self, "Fit parameters error", str(e))
            except Exception:
                pass
            return

        # preserve current view
        self._fit_all_restore = {
            'current_key': getattr(self, 'current_file_key', None),
            'current_plot': getattr(self, 'current_plot_option', 'EIS_data'),
        }

        # iterate in UI order if possible
        keys = []
        if hasattr(self.ui, 'files_list'):
            for i in range(self.ui.files_list.count()):
                it = self.ui.files_list.item(i)
                if it:
                    k = it.data(QtCore.Qt.UserRole)
                    if k:
                        keys.append(k)
        if not keys:
            keys = list(self.data_store.keys())

        n_skip = 0
        to_fit = []
        for key in keys:
            meta = getattr(self, 'file_meta', {}).get(key)
            if meta and meta.get('fitted') and meta.get('signature') == signature:
                n_skip += 1
                continue
            if key in self.data_store:
                to_fit.append(key)

        if not to_fit:
            try:
                self.statusBar().showMessage(f'Fit all done: 0 fitted, {n_skip} skipped', 2000)
            except Exception:
                pass
            return

        if not self._confirm_raw_fit_if_needed(to_fit, mode):
            try:
                self.statusBar().showMessage('Fit all cancelled.', 1500)
            except Exception:
                pass
            return

        self._fit_all_active = True
        self._fit_all_pending = len(to_fit)
        self._fit_all_done = 0
        self._fit_all_skipped = n_skip
        self._fit_all_errors = 0

        # schedule workers
        for key in to_fit:
            try:
                entry_copy = copy.deepcopy(self.data_store.get(key))
            except Exception:
                entry_copy = self.data_store.get(key)

            worker = _FitWorker(key=key, entry=entry_copy, mode=mode, params=params, signature=signature)
            worker.signals.finished.connect(self._on_fit_worker_finished)
            worker.signals.error.connect(self._on_fit_worker_error)
            self._fit_pool.start(worker)

        try:
            self.statusBar().showMessage(f'Fit all started: {len(to_fit)} running, {n_skip} skipped', 2000)
        except Exception:
            pass

    def _build_fit_params(self, mode: str) -> dict:
        """Read all fitting parameters from UI once; safe to pass into worker threads."""
        params = {}
        # For parallel Fit All, limit each task to 1 native BLAS/OMP thread so
        # multiple fits can run concurrently without oversubscribing the CPU.
        params['native_threads'] = 1
        params['rbf_type'] = str(self.ui.discre_choice.currentText())
        params['induct_used'] = int(self.ui.induct_choice.currentIndex())
        params['der_used'] = str(self.ui.der_choice.currentText())
        params['shape_control'] = str(self.ui.shape_control_choice.currentText())
        params['coeff'] = float(self.ui.FWHM_entry.text())

        if mode in ('simple', 'bayesian'):
            params['data_used'] = str(self.ui.data_used_choice.currentText())
            params['cv_type'] = str(self.ui.lambda_choice.currentText())
            params['reg_param'] = float(self.ui.reg_param_entry.text())

        if mode == 'bayesian':
            params['sample_number'] = int(float(self.ui.sample_no_entry.text()))

        return params

    @QtCore.pyqtSlot(str, object, object)
    def _on_fit_worker_finished(self, key: str, fitted_entry, signature):
        """Main-thread slot: store results + update status."""
        try:
            self.data_store[key] = fitted_entry
            if key == getattr(self, 'current_file_key', None):
                self.data = fitted_entry

            if key in getattr(self, 'file_meta', {}):
                self.file_meta[key] = {'fitted': True, 'signature': signature}
            self._update_file_status(key, fitted=True)
            self._mark_project_dirty()
        except Exception:
            pass

        try:
            self._fit_all_done += 1
            self._fit_all_pending -= 1
        except Exception:
            return

        if self._fit_all_pending <= 0:
            self._finish_fit_all_parallel()

    @QtCore.pyqtSlot(str, str)
    def _on_fit_worker_error(self, key: str, msg: str):
        try:
            self._fit_all_errors += 1
            self._fit_all_pending -= 1
        except Exception:
            return

        try:
            if key in getattr(self, 'file_meta', {}):
                self.file_meta[key] = {'fitted': False, 'signature': None}
            self._update_file_status(key, fitted=False)
        except Exception:
            pass

        if self._fit_all_pending <= 0:
            self._finish_fit_all_parallel()

    def _finish_fit_all_parallel(self):
        """Finalize Fit All after all workers complete."""
        self._fit_all_active = False

        try:
            current_key = self._fit_all_restore.get('current_key')
            current_plot = self._fit_all_restore.get('current_plot', 'EIS_data')
        except Exception:
            current_key = getattr(self, 'current_file_key', None)
            current_plot = getattr(self, 'current_plot_option', 'EIS_data')

        if current_key in getattr(self, 'data_store', {}):
            self.current_file_key = current_key
            self.data = self.data_store[current_key]

        self.current_plot_option = current_plot
        try:
            self.plotting_callback(current_plot)
        except Exception:
            pass

        try:
            self.statusBar().showMessage(
                f'Fit all done: {self._fit_all_done} fitted, {self._fit_all_skipped} skipped, {self._fit_all_errors} errors',
                2500
            )
        except Exception:
            pass

    def inductance_callback(self):

        keys = self._get_file_keys_in_ui_order()
        if not keys:
            if self.data is None:
                return
            keys = [getattr(self, 'current_file_key', None)] if getattr(self, 'current_file_key', None) else []
            if not keys:
                return

        errors = []
        for key in keys:
            try:
                self._apply_inductance_to_key(key, reset_method=True)
            except Exception as e:
                errors.append((key, str(e)))

        current_key = getattr(self, 'current_file_key', None)
        if current_key in getattr(self, 'data_store', {}):
            self.data = self.data_store[current_key]
        elif keys:
            first_ok = next((k for k in keys if k in getattr(self, 'data_store', {})), None)
            if first_ok is not None:
                self._set_current_file(first_ok)

        # keep current plot selection (do not force EIS)
        self.plotting_callback(self.current_plot_option)

        if errors:
            try:
                msg = '; '.join(f"{os.path.basename(k) if k else 'Unknown'}: {err}" for k, err in errors[:3])
                if len(errors) > 3:
                    msg += f"; +{len(errors) - 3} more"
                QtWidgets.QMessageBox.warning(self, 'Inductance option warning', msg)
            except Exception:
                pass

    def kk_run_callback(self):
        """Run K-K validation only for files whose data/settings changed."""
        keys = self._get_file_keys_in_ui_order()
        if not keys:
            if self.data is None:
                return
            keys = [getattr(self, 'current_file_key', None)] if getattr(self, 'current_file_key', None) else []
            if not keys:
                return

        try:
            c = float(self.ui.cutoff_entry.text())
            max_m = int(float(self.ui.max_elements_entry.text()))
            fit_type = str(self.ui.fit_type_choice.currentText()).strip().lower()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'K-K parameters error', f'Invalid K-K parameters: {e}')
            return

        self.kk_settings = {'c': c, 'max_m': max_m, 'fit_type': fit_type}

        done = 0
        skipped = 0
        errors = 0
        last_error = None
        for key in keys:
            if key is None:
                continue
            entry = self.data_store.get(key, self.data if key == getattr(self, 'current_file_key', None) else None)
            if entry is None:
                continue

            try:
                current_signature = _build_kk_signature(entry, c=c, max_m=max_m, fit_type=fit_type)
                if bool(getattr(entry, 'kk_valid', False)) and getattr(entry, 'kk_signature', None) == current_signature:
                    _refresh_kk_current_arrays(entry)
                    if key in self.data_store:
                        self.data_store[key] = entry
                    if key == getattr(self, 'current_file_key', None):
                        self.data = entry
                    skipped += 1
                    continue

                _clear_kk_results(entry)
                entry = _run_lin_kk(entry, c=c, max_m=max_m, fit_type=fit_type)
                if key in self.data_store:
                    self.data_store[key] = entry
                if key == getattr(self, 'current_file_key', None):
                    self.data = entry

                # K-K validation does not change the DRT input data. Therefore,
                # it must not reset the existing Fit status/signature.
                done += 1
            except Exception as e:
                errors += 1
                last_error = str(e)
                try:
                    _clear_kk_results(entry)
                except Exception:
                    pass

        if getattr(self, 'current_file_key', None) in self.data_store:
            self.data = self.data_store[self.current_file_key]

        if getattr(self, 'current_plot_option', None) in ('EIS_data', 'KK_residual'):
            self.plotting_callback(self.current_plot_option)

        if done > 0:
            self._mark_project_dirty()

        msg = f'K-K analysis done: {done} processed, {skipped} skipped'
        if errors:
            msg += f', {errors} error(s)'
        try:
            self.statusBar().showMessage(msg, 3000)
        except Exception:
            pass

        if errors and last_error:
            try:
                QtWidgets.QMessageBox.warning(self, 'K-K analysis warning', last_error)
            except Exception:
                pass

    def simple_run_callback(self):  # callback for simple ridge regularization

        if self.data is None:
            return

        # we first read the parameters chosen by the users
        rbf_type = str(self.ui.discre_choice.currentText())
        data_used = str(self.ui.data_used_choice.currentText())
        induct_used = int(self.ui.induct_choice.currentIndex())
        der_used = str(self.ui.der_choice.currentText())
        cv_type = str(self.ui.lambda_choice.currentText())
        reg_param = float(self.ui.reg_param_entry.text())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())

        # we perform the computation
        self.data = simple_run(self.data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                               der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control,
                               coeff=coeff)

        entry = self.data
        # optimally select the regularization level
        if cv_type == 'custom':
            entry.lambda_value = reg_param
        else:
            entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                       data_used, induct_used, -3, cv_type)

        # Update the QLineEdit with the computed regularization parameter
        self.ui.reg_param_entry_2.setText(str(entry.lambda_value))

        self._mark_project_dirty()
        self.plotting_callback('DRT_data')

    def bayesian_run_callback(self):  # callback for Bayesian regularization

        if self.data is None:
            return

        # we first read the parameters chosen by the users
        rbf_type = str(self.ui.discre_choice.currentText())
        data_used = str(self.ui.data_used_choice.currentText())
        induct_used = int(self.ui.induct_choice.currentIndex())
        der_used = str(self.ui.der_choice.currentText())
        cv_type = str(self.ui.lambda_choice.currentText())
        reg_param = float(self.ui.reg_param_entry.text())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())
        sample_number = int(self.ui.sample_no_entry.text())

        # we perform the computation
        self.data = Bayesian_run(self.data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                 der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control,
                                 coeff=coeff, NMC_sample=sample_number)
        self._mark_project_dirty()
        self.plotting_callback('DRT_data')

    def BHT_run_callback(self):  # callback for Hilbert transform run

        if self.data is None:
            return

        # we first read the parameters chosen by the users
        rbf_type = str(self.ui.discre_choice.currentText())
        der_used = str(self.ui.der_choice.currentText())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())

        # we perform the computation
        self.data = BHT_run(self.data, rbf_type, der_used, shape_control, coeff)
        ##
        self._mark_project_dirty()
        self.plotting_callback('DRT_data')

    def peak_analysis_run_callback(self):  # callback for peak analysis

        if self.data is None:
            return

        # we first read the parameters chosen by the users
        rbf_type = str(self.ui.discre_choice.currentText())
        data_used = str(self.ui.data_used_choice.currentText())
        induct_used = int(self.ui.induct_choice.currentIndex())
        der_used = str(self.ui.der_choice.currentText())
        cv_type = str(self.ui.lambda_choice.currentText())
        reg_param = float(self.ui.reg_param_entry.text())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())
        peak_method = str(self.ui.peak_method_choice.currentText())
        N_peaks = float(self.ui.peak_num_entry.text())

        # we perform the computation
        self.data = peak_analysis(self.data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                  der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control,
                                  coeff=coeff, peak_method=peak_method, N_peaks=N_peaks)
        if self.current_file_key in self.data_store:
            self.data_store[self.current_file_key] = self.data
        self._mark_project_dirty()
        self.plotting_callback('DRT_data')

    def plotting_callback(self, plot_to_show):

        if plot_to_show in ('Re_residual', 'Im_residual'):
            plot_to_show = 'DRT_residual'

        # remember current plot selection so file switching keeps the same view
        self.current_plot_option = plot_to_show

        if plot_to_show == 'DRT_comparison':
            self._plot_drt_comparison()
            return

        if plot_to_show == 'DRT_map':
            self._plot_drt_map()
            return

        # not plotting if no data has been imported
        if self.data is None:
            # clear panel
            try:
                self.ui.plot_panel.setScene(QtWidgets.QGraphicsScene())
            except Exception:
                pass
            return

        # initalize the figure object
        fig = Figure_Canvas()

        # rendering the figure object on to the gui
        fig_to_plot = getattr(fig, plot_to_show)
        fig_to_plot(self.data)
        if plot_to_show == 'EIS_data':
            self._eis_canvas = fig
            try:
                fig.mpl_connect('button_press_event', self._on_eis_button_press)
            except Exception:
                pass
            try:
                self._eis_rect_selector = RectangleSelector(
                    fig.axes,
                    self._on_eis_box_select,
                    useblit=False,
                    button=[1],
                    interactive=False,
                    spancoords='data',
                    minspanx=0.0,
                    minspany=0.0,
                )
                fig._eis_rect_selector = self._eis_rect_selector
            except Exception:
                self._eis_rect_selector = None
            try:
                self._apply_eis_selection_highlight(fig)
            except Exception:
                pass
        self._show_canvas_in_plot_panel(fig)
        self.ui.plot_panel.show()

    def _show_canvas_in_plot_panel(self, fig):
        view = self.ui.plot_panel
        scene = QtWidgets.QGraphicsScene(view)
        proxy = scene.addWidget(fig)

        view.setScene(scene)
        view.setAlignment(QtCore.Qt.AlignCenter)

        margin = 8
        vw = max(200, view.viewport().width() - 2 * margin)
        vh = max(200, view.viewport().height() - 2 * margin)

        dpi = fig.figure.get_dpi()
        fig.figure.set_size_inches(vw / dpi, vh / dpi, forward=True)
        fig.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        fig.resize(vw, vh)

        proxy.setPos(margin, margin)
        scene.setSceneRect(0, 0, vw + 2 * margin, vh + 2 * margin)

        self._current_canvas = fig
        self._current_scene = scene
        self._current_proxy = proxy

        try:
            fig.draw_idle()
        except Exception:
            fig.draw()

    def _refresh_current_plot_size(self):
        if getattr(self, 'current_plot_option', None):
            try:
                self.plotting_callback(self.current_plot_option)
            except Exception:
                pass

    def _get_file_keys_in_ui_order(self):
        keys = []
        if hasattr(self.ui, 'files_list'):
            for i in range(self.ui.files_list.count()):
                it = self.ui.files_list.item(i)
                if it is None:
                    continue
                k = it.data(QtCore.Qt.UserRole)
                if k:
                    keys.append(k)
        if not keys:
            keys = list(getattr(self, 'data_store', {}).keys())
        return keys

    @staticmethod
    def _hex_to_rgb(hex_color: str):
        h = (hex_color or '').strip()
        if not h.startswith('#'):
            h = '#' + h
        if not re.fullmatch(r"#([0-9a-fA-F]{6})", h):
            raise ValueError(f"Invalid hex color: {hex_color}")
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    @staticmethod
    def _rgb_to_hex(rgb):
        return '#%02X%02X%02X' % tuple(int(max(0, min(255, c))) for c in rgb)

    def _compute_gradient_colors(self, n: int, start_hex: str, end_hex: str):
        if n <= 0:
            return []
        if n == 1:
            return [start_hex]
        s = self._hex_to_rgb(start_hex)
        e = self._hex_to_rgb(end_hex)
        cols = []
        for i in range(n):
            t = i / (n - 1)
            rgb = (
                int(round(s[0] + (e[0] - s[0]) * t)),
                int(round(s[1] + (e[1] - s[1]) * t)),
                int(round(s[2] + (e[2] - s[2]) * t)),
            )
            cols.append(self._rgb_to_hex(rgb))
        return cols

    def _on_drt_comp_pick(self, event):
        # When user clicks a line, select the corresponding file in the Files list.
        # Only react to left-click picks. (On some backends/Qt embeddings, non-left
        # interactions can still generate pick events.)
        try:
            me = getattr(event, 'mouseevent', None)
            if me is not None and getattr(me, 'button', None) != 1:
                return
        except Exception:
            pass

        artist = getattr(event, 'artist', None)
        if artist is None or not hasattr(self.ui, 'files_list'):
            return
        key = None
        try:
            key = artist.get_gid()
        except Exception:
            key = None
        if not key:
            return

        for i in range(self.ui.files_list.count()):
            it = self.ui.files_list.item(i)
            if it and it.data(QtCore.Qt.UserRole) == key:
                self._suppress_file_select = True
                self.ui.files_list.setCurrentRow(i)
                break

        # highlight: selected line stays opaque; others become 80% transparent
        try:
            self._apply_drt_comp_selection(artist)
        except Exception:
            pass

    def _drt_comp_force_redraw(self):
        """Force an immediate repaint of the DRT comparison plot.

        Because the Matplotlib canvas is embedded inside a QGraphicsView via
        QGraphicsProxyWidget, canvas.draw()/draw_idle() alone may not flush to
        the screen immediately. Wheel/resize events often trigger a repaint,
        which is why alpha changes can appear only after scrolling.

        This method forces BOTH the Matplotlib canvas redraw AND the Qt
        QGraphicsView/scene repaint.
        """
        canvas = getattr(self, '_drt_comp_canvas', None)
        if canvas is None:
            return

        view = getattr(self.ui, 'plot_panel', None)
        scene = getattr(self, '_drt_comp_scene', None)
        proxy = getattr(self, '_drt_comp_proxy', None)

        def _do():
            # 1) redraw matplotlib
            try:
                canvas.draw()
            except Exception:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass

            # 2) force Qt repaint of the embedded widget + scene + viewport
            try:
                if proxy is not None:
                    proxy.update()
            except Exception:
                pass
            try:
                if scene is not None:
                    scene.invalidate(scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                    scene.update()
            except Exception:
                pass
            try:
                if view is not None:
                    view.viewport().update()
                    view.viewport().repaint()
                    view.update()
                    view.repaint()
            except Exception:
                pass

        # Queue repaint to the next Qt event loop tick (important for mpl callbacks)
        try:
            QtCore.QTimer.singleShot(0, _do)
        except Exception:
            _do()

    def _on_drt_comp_button_press(self, event):
        """DRT comparison interactions (robust in QGraphicsView embedding).

        - Right click in axes -> context menu with "Setting".
        - Left click on a curve -> select that curve, dim others (alpha=0.2), bring selected curve to top.
        - Left click on blank/outside axes -> clear selection (all opaque).
        """
        ax = getattr(event, 'inaxes', None)

        # click outside axes
        if ax is None:
            self._clear_drt_comp_selection()
            return

        btn = getattr(event, 'button', None)

        # right click -> menu
        if btn == 3:
            self._show_drt_comp_context_menu()
            return

        # only handle left click for selection
        if btn != 1:
            return

        # manual hit-test for lines (pick_event is unreliable under QGraphicsProxyWidget)
        clicked_line = None
        try:
            for ln in (self._drt_comp_lines or []):
                ok, _ = ln.contains(event)
                if ok:
                    clicked_line = ln
                    break
        except Exception:
            clicked_line = None

        if clicked_line is None:
            # blank click -> clear selection
            self._clear_drt_comp_selection()
            return

        # select corresponding file in Files list
        key = None
        try:
            key = clicked_line.get_gid()
        except Exception:
            key = None

        if key and hasattr(self.ui, 'files_list'):
            for i in range(self.ui.files_list.count()):
                it = self.ui.files_list.item(i)
                if it and it.data(QtCore.Qt.UserRole) == key:
                    self._suppress_file_select = True
                    self.ui.files_list.setCurrentRow(i)
                    break

        self._apply_drt_comp_selection(clicked_line)

    def _on_drt_comp_motion(self, event):
        """Hover tooltip on DRT comparison curves (file, tau, gamma)."""
        ax = getattr(event, 'inaxes', None)
        canvas = getattr(self, '_drt_comp_canvas', None)
        if ax is None or canvas is None:
            self._drt_comp_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        if getattr(event, 'xdata', None) is None or getattr(event, 'ydata', None) is None:
            self._drt_comp_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        ex = getattr(event, 'x', None)
        ey = getattr(event, 'y', None)
        if ex is None or ey is None:
            return

        best = None  # (dist2, line, idx)

        lines_to_check = (self._drt_comp_lines or [])
        # If a curve is selected, only show hover info for the selected curve
        if getattr(self, '_drt_comp_selected', None) is not None:
            lines_to_check = [self._drt_comp_selected]

        try:
            for ln in lines_to_check:
                xdata = np.asarray(ln.get_xdata(), dtype=float)
                ydata = np.asarray(ln.get_ydata(), dtype=float)
                if xdata.size == 0:
                    continue
                pts = ax.transData.transform(np.column_stack([xdata, ydata]))
                dx = pts[:, 0] - ex
                dy = pts[:, 1] - ey
                dist2 = dx * dx + dy * dy
                j = int(np.argmin(dist2))
                d2 = float(dist2[j])
                if best is None or d2 < best[0]:
                    best = (d2, ln, j)
        except Exception:
            best = None

        if best is None:
            return

        d2, ln, idx = best
        # threshold ~ 8px
        if d2 > (8.0 * 8.0):
            if self._drt_comp_hover_last is not None:
                self._drt_comp_hover_last = None
                try:
                    QtWidgets.QToolTip.hideText()
                except Exception:
                    pass
            return

        key = None
        try:
            key = ln.get_gid()
        except Exception:
            key = None
        fname = os.path.basename(key) if key else "Curve"

        try:
            tau_val = float(ln.get_xdata()[idx])
            gam_val = float(ln.get_ydata()[idx])
        except Exception:
            return

        tag = (key, idx)
        if self._drt_comp_hover_last == tag:
            return
        self._drt_comp_hover_last = tag

        text = f"File: {fname}\nτ: {tau_val:.3e} s\nγ: {gam_val:.4g} Ω"
        try:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
        except Exception:
            pass

    def _show_drt_comp_context_menu(self):
        """Show right-click context menu on DRT comparison plot."""
        menu = QtWidgets.QMenu(self)
        act = QtWidgets.QAction("Setting", self)
        menu.addAction(act)
        act.triggered.connect(self._open_drt_comparison_settings)
        try:
            menu.exec_(QtGui.QCursor.pos())
        except Exception:
            menu.exec_()

    def _apply_drt_comp_selection(self, selected_line):
        """Dim all DRT comparison curves except the selected one; bring selected curve to top."""
        if not self._drt_comp_lines:
            return

        for ln in self._drt_comp_lines:
            try:
                if ln is selected_line:
                    ln.set_alpha(1.0)
                    ln.set_zorder(10)
                else:
                    ln.set_alpha(0.2)
                    ln.set_zorder(1)
            except Exception:
                pass

        self._drt_comp_selected = selected_line
        self._drt_comp_force_redraw()

    def _clear_drt_comp_selection(self):
        """Restore all DRT comparison curves to fully opaque."""
        if not self._drt_comp_lines:
            self._drt_comp_selected = None
            return

        if self._drt_comp_selected is None:
            return

        for ln in self._drt_comp_lines:
            try:
                ln.set_alpha(1.0)
                ln.set_zorder(1)
            except Exception:
                pass

        self._drt_comp_selected = None
        self._drt_comp_force_redraw()

    def _open_drt_comparison_settings(self):
        """Dialog: set y-limits and gradient colors for DRT comparison."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DRT comparison settings")
        dlg.setModal(True)

        form = QtWidgets.QFormLayout(dlg)

        ymin_edit = QtWidgets.QLineEdit(dlg)
        ymax_edit = QtWidgets.QLineEdit(dlg)
        ymin_edit.setPlaceholderText("auto")
        ymax_edit.setPlaceholderText("auto")

        if self.drt_comp_settings.get('ymin') is not None:
            ymin_edit.setText(str(self.drt_comp_settings.get('ymin')))
        if self.drt_comp_settings.get('ymax') is not None:
            ymax_edit.setText(str(self.drt_comp_settings.get('ymax')))

        start_edit = QtWidgets.QLineEdit(dlg)
        end_edit = QtWidgets.QLineEdit(dlg)
        start_edit.setText(self.drt_comp_settings.get('start_color', '#08306B'))
        end_edit.setText(self.drt_comp_settings.get('end_color', '#DEEBF7'))

        def _pick_color(target_edit: QtWidgets.QLineEdit):
            try:
                current = QtGui.QColor(target_edit.text().strip())
            except Exception:
                current = QtGui.QColor("#08306B")
            col = QtWidgets.QColorDialog.getColor(current, dlg)
            if col.isValid():
                target_edit.setText(col.name().upper())

        pick_start = QtWidgets.QPushButton("Pick", dlg)
        pick_end = QtWidgets.QPushButton("Pick", dlg)
        pick_start.clicked.connect(lambda: _pick_color(start_edit))
        pick_end.clicked.connect(lambda: _pick_color(end_edit))

        start_row = QtWidgets.QHBoxLayout()
        start_row.addWidget(start_edit)
        start_row.addWidget(pick_start)

        end_row = QtWidgets.QHBoxLayout()
        end_row.addWidget(end_edit)
        end_row.addWidget(pick_end)

        form.addRow("Y min", ymin_edit)
        form.addRow("Y max", ymax_edit)
        form.addRow("Gradient start", start_row)
        form.addRow("Gradient end", end_row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dlg
        )
        form.addRow(btns)

        def _parse_float_or_none(s):
            s = (s or '').strip()
            if s == '':
                return None
            return float(s)

        def _apply():
            try:
                ymin = _parse_float_or_none(ymin_edit.text())
                ymax = _parse_float_or_none(ymax_edit.text())
                if ymin is not None and ymax is not None and ymin >= ymax:
                    raise ValueError("Y min must be smaller than Y max.")
                start_hex = start_edit.text().strip()
                end_hex = end_edit.text().strip()
                self._hex_to_rgb(start_hex)
                self._hex_to_rgb(end_hex)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Invalid settings", str(e))
                return

            self.drt_comp_settings['ymin'] = ymin
            self.drt_comp_settings['ymax'] = ymax
            self.drt_comp_settings['start_color'] = start_hex.upper()
            self.drt_comp_settings['end_color'] = end_hex.upper()
            self._mark_project_dirty()
            dlg.accept()

        btns.accepted.connect(_apply)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            if getattr(self, 'current_plot_option', None) == 'DRT_comparison':
                self.plotting_callback('DRT_comparison')

    def _plot_drt_comparison(self):
        # Build comparison plot from all fitted entries, using UI order and gradient colors.
        keys = self._get_file_keys_in_ui_order()
        if not keys or not getattr(self, 'data_store', None):
            try:
                self.ui.plot_panel.setScene(QtWidgets.QGraphicsScene())
            except Exception:
                pass
            return

        # collect entries that have DRT data
        entries = []
        entry_keys = []
        for k in keys:
            entry = self.data_store.get(k)
            if entry is None:
                continue
            if getattr(entry, 'method', 'none') == 'none':
                continue
            if not hasattr(entry, 'out_tau_vec'):
                continue
            entries.append(entry)
            entry_keys.append(k)

        if not entries:
            try:
                self.ui.plot_panel.setScene(QtWidgets.QGraphicsScene())
            except Exception:
                pass
            return

        cols = self._compute_gradient_colors(
            len(entries),
            self.drt_comp_settings.get('start_color', '#08306B'),
            self.drt_comp_settings.get('end_color', '#DEEBF7'),
        )

        y_lim = None
        if self.drt_comp_settings.get('ymin') is not None or self.drt_comp_settings.get('ymax') is not None:
            y_lim = (self.drt_comp_settings.get('ymin'), self.drt_comp_settings.get('ymax'))

        fig = Figure_Canvas()
        fig.DRT_comparison(entries, cols, y_lim=y_lim, keys=entry_keys)

        # store lines for highlighting/dimming
        self._drt_comp_canvas = fig
        self._drt_comp_lines = list(getattr(fig, '_drt_comp_lines', []) or [])
        self._drt_comp_selected = None

        # interactivity
        try:
            fig.mpl_connect('button_press_event', self._on_drt_comp_button_press)
            fig.mpl_connect('motion_notify_event', self._on_drt_comp_motion)
        except Exception:
            pass

        self._show_canvas_in_plot_panel(fig)
        self._drt_comp_scene = self._current_scene
        self._drt_comp_proxy = self._current_proxy
        self.ui.plot_panel.show()
        self._select_drt_comparison_line_by_key(self.current_file_key)

    # ---------------- DRT Map (2D heatmap) ----------------

    def _show_drt_map_context_menu(self):
        menu = QtWidgets.QMenu(self)
        act = QtWidgets.QAction("Setting", self)
        menu.addAction(act)
        act.triggered.connect(self._open_drt_map_settings)
        try:
            menu.exec_(QtGui.QCursor.pos())
        except Exception:
            menu.exec_()

    def _open_drt_map_settings(self):
        """Configure DRT Map color scaling and Origin-style level settings."""
        # Determine current auto min/max if needed (based on current Z matrix)
        auto_min = None
        auto_max = None
        try:
            z = getattr(self, '_drt_map_Z', None)
            if z is not None and hasattr(z, 'size') and z.size:
                auto_min = float(np.nanmin(z))
                auto_max = float(np.nanmax(z))
        except Exception:
            pass

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DRT Map settings")
        dlg.setModal(True)
        form = QtWidgets.QFormLayout(dlg)

        from_edit = QtWidgets.QLineEdit(dlg)
        to_edit = QtWidgets.QLineEdit(dlg)
        major_edit = QtWidgets.QLineEdit(dlg)
        minor_edit = QtWidgets.QLineEdit(dlg)

        # placeholders / defaults
        vmin = self.drt_map_settings.get('vmin')
        vmax = self.drt_map_settings.get('vmax')
        if vmin is None and auto_min is not None:
            vmin = auto_min
        if vmax is None and auto_max is not None:
            vmax = auto_max

        if vmin is not None:
            from_edit.setText(str(vmin))
        if vmax is not None:
            to_edit.setText(str(vmax))

        major_edit.setText(str(self.drt_map_settings.get('major_levels', 10)))
        minor_edit.setText(str(self.drt_map_settings.get('minor_levels', 10)))

        fill_choice = QtWidgets.QComboBox(dlg)
        fill_choice.addItems(["Fill to Contour Lines", "Fill to Grid Lines"])
        fill_mode = (self.drt_map_settings.get('fill_mode') or 'contour').lower()
        fill_choice.setCurrentIndex(0 if fill_mode == 'contour' else 1)

        form.addRow("From", from_edit)
        form.addRow("To", to_edit)
        form.addRow("Major levels", major_edit)
        form.addRow("Minor levels", minor_edit)
        form.addRow("Fill mode", fill_choice)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dlg
        )
        form.addRow(btns)

        def _apply():
            try:
                vmin2 = float(from_edit.text().strip())
                vmax2 = float(to_edit.text().strip())
                if vmin2 >= vmax2:
                    raise ValueError("From must be smaller than To.")

                maj = int(float(major_edit.text().strip()))
                minu = int(float(minor_edit.text().strip()))
                if maj < 2:
                    raise ValueError("Major levels must be >= 2.")
                if minu < 2:
                    raise ValueError("Minor levels must be >= 2.")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Invalid settings", str(e))
                return

            self.drt_map_settings['vmin'] = vmin2
            self.drt_map_settings['vmax'] = vmax2
            self.drt_map_settings['major_levels'] = maj
            self.drt_map_settings['minor_levels'] = minu
            self.drt_map_settings['fill_mode'] = 'contour' if fill_choice.currentIndex() == 0 else 'grid'
            self._mark_project_dirty()
            dlg.accept()

        btns.accepted.connect(_apply)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            if getattr(self, 'current_plot_option', None) == 'DRT_map':
                self.plotting_callback('DRT_map')

    @staticmethod
    def _logspace_edges(x):
        """Compute pcolormesh-friendly bin edges for a strictly positive x grid."""
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            return np.array([x[0] * 0.9, x[0] * 1.1], dtype=float)
        edges = np.empty(x.size + 1, dtype=float)
        # internal edges at geometric mean
        edges[1:-1] = np.sqrt(x[:-1] * x[1:])
        # extrapolate first/last
        edges[0] = x[0] * (x[0] / edges[1])
        edges[-1] = x[-1] * (x[-1] / edges[-2])
        return edges

    def _on_drt_map_button_press(self, event):
        # right-click -> settings menu
        if getattr(event, 'button', None) == 3:
            self._show_drt_map_context_menu()
            return

    def _on_drt_map_motion(self, event):
        # show tooltip for hovered point (only on the heatmap axes, not on the colorbar)
        ax = getattr(event, 'inaxes', None)
        main_ax = getattr(self, '_drt_map_ax', None)
        if ax is None or main_ax is None or ax is not main_ax:
            self._drt_map_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        x = getattr(event, 'xdata', None)
        y = getattr(event, 'ydata', None)
        if x is None or y is None:
            self._drt_map_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        xgrid = getattr(self, '_drt_map_x', None)
        zmat = getattr(self, '_drt_map_Z', None)
        keys = getattr(self, '_drt_map_keys', None)
        if xgrid is None or zmat is None or keys is None:
            return

        n_files = int(zmat.shape[0])
        # y is 0..n_files (we invert y-axis), so row is int(y) clipped
        row = int(np.clip(int(y), 0, n_files - 1))

        # nearest x index
        try:
            xi = int(np.searchsorted(xgrid, x))
            if xi <= 0:
                col = 0
            elif xi >= len(xgrid):
                col = len(xgrid) - 1
            else:
                col = xi - 1 if abs(xgrid[xi - 1] - x) <= abs(xgrid[xi] - x) else xi
        except Exception:
            col = 0

        last = getattr(self, '_drt_map_hover_last', None)
        if last == (row, col):
            return
        self._drt_map_hover_last = (row, col)

        try:
            val = float(zmat[row, col])
        except Exception:
            return

        key = keys[row] if row < len(keys) else None
        fname = os.path.basename(key) if key else f"Row {row + 1}"

        try:
            tau_val = float(xgrid[col])
        except Exception:
            tau_val = float(x)

        text = f"File: {fname}\nτ: {tau_val:.3e} s\nγ: {val:.4g} Ω"
        try:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
        except Exception:
            pass

    def _plot_drt_map(self):
        # Build 2D heatmap from all fitted entries, ordered by Files list (top->bottom).
        keys = self._get_file_keys_in_ui_order()
        if not keys or not getattr(self, 'data_store', None):
            try:
                self.ui.plot_panel.setScene(QtWidgets.QGraphicsScene())
            except Exception:
                pass
            return

        curves = []
        entry_keys = []

        for k in keys:
            entry = self.data_store.get(k)
            if entry is None:
                continue
            if getattr(entry, 'method', 'none') == 'none':
                continue
            if not hasattr(entry, 'out_tau_vec'):
                continue

            x = np.asarray(entry.out_tau_vec, dtype=float)
            y = None
            if entry.method in ('simple', 'peak'):
                y = getattr(entry, 'gamma', None)
            elif entry.method == 'credit':
                y = getattr(entry, 'gamma', None)  # MAP
            elif entry.method == 'BHT':
                y = getattr(entry, 'mu_gamma_fine_re', None)
            else:
                y = getattr(entry, 'gamma', None)

            if y is None:
                continue
            y = np.asarray(y, dtype=float)
            if x.size != y.size:
                continue

            curves.append((x, y))
            entry_keys.append(k)

        if not curves:
            try:
                self.ui.plot_panel.setScene(QtWidgets.QGraphicsScene())
            except Exception:
                pass
            return

        # ---- Common X grid (log-x) ----
        xs = [c[0] for c in curves]
        same_grid = True
        for a in xs[1:]:
            if a.shape != xs[0].shape or not np.allclose(a, xs[0], rtol=0, atol=0):
                same_grid = False
                break

        if same_grid:
            xgrid = xs[0]
        else:
            xmin = min(float(np.min(x)) for x in xs)
            xmax = max(float(np.max(x)) for x in xs)
            npts = max(len(x) for x in xs)
            xgrid = np.logspace(np.log10(xmin), np.log10(xmax), npts)

        # ---- Interpolate all curves onto xgrid (log-x) ----
        zmat = np.zeros((len(curves), len(xgrid)), dtype=float)
        lxg = np.log10(xgrid)
        for i, (x, y) in enumerate(curves):
            lx = np.log10(x)
            if np.any(np.diff(lx) <= 0):
                order = np.argsort(lx)
                lx = lx[order]
                y2 = y[order]
            else:
                y2 = y
            zmat[i, :] = np.interp(lxg, lx, y2)

        # ---- Settings: From/To and Origin-style levels ----
        vmin = self.drt_map_settings.get('vmin')
        vmax = self.drt_map_settings.get('vmax')
        if vmin is None:
            vmin = float(np.nanmin(zmat))
        if vmax is None:
            vmax = float(np.nanmax(zmat))
        if vmin == vmax:
            vmax = vmin + 1e-12

        major_levels = int(self.drt_map_settings.get('major_levels', 10) or 10)
        minor_levels = int(self.drt_map_settings.get('minor_levels', 10) or 10)
        total_levels = max(2, major_levels * minor_levels)

        fill_mode = (self.drt_map_settings.get('fill_mode') or 'contour').lower()
        # Origin-like warming palette (blue -> white -> red)
        cmap = mpl.cm.get_cmap('RdBu_r', total_levels)

        fig = Figure_Canvas()
        ax = fig.axes

        # ---- Layout: heatmap big; vertical short colorbar on the left ----
        heat_top = 0.90
        heat_bottom = 0.12

        cbar_left = -0.005
        cbar_w = 0.028

        gap_x = 0.030
        heat_left = 0.05 + cbar_w + gap_x
        heat_right = 0.92

        ax.set_position([heat_left, heat_bottom, heat_right - heat_left, heat_top - heat_bottom])

        # short vertical colorbar: about 1/4 of heatmap height
        heat_h = (heat_top - heat_bottom)
        cbar_h = heat_h * 0.25
        cbar_bottom = heat_bottom
        cax = fig.figure.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])

        n_files = zmat.shape[0]

        # ---- Plot heatmap ----
        if fill_mode == 'grid':
            x_edges = self._logspace_edges(xgrid)
            y_edges = np.linspace(0, n_files, n_files + 1)
            m = ax.pcolormesh(
                x_edges, y_edges, zmat,
                cmap=cmap,
                shading='auto',
                vmin=vmin,
                vmax=vmax
            )
            ax.invert_yaxis()  # first file at top
        else:
            # Fill to Contour Lines (smooth, no blank band):
            # use y-edges (0..n_files) and a z-matrix with (n_files+1) rows
            y_edges = np.arange(n_files + 1, dtype=float)

            if n_files == 1:
                z_edges = np.vstack([zmat[0], zmat[0]])
            else:
                z_edges = np.empty((n_files + 1, zmat.shape[1]), dtype=float)
                z_edges[0, :] = zmat[0, :]
                z_edges[-1, :] = zmat[-1, :]
                z_edges[1:-1, :] = 0.5 * (zmat[:-1, :] + zmat[1:, :])

            levels = np.linspace(vmin, vmax, total_levels)
            m = ax.contourf(
                xgrid, y_edges, z_edges,
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            ax.set_ylim(n_files, 0)  # first file at top

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau/s$')
        ax.set_ylabel('Files')
        ax.set_xlim([float(np.min(xgrid)), float(np.max(xgrid))])
        # 每个文件之间一个刻度（边界线）
        ax.set_yticks(range(0, n_files + 1))
        # 隐藏 y 轴数字，但保留刻度线
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='major', left=True, labelleft=False)

        # ---- Colorbar: only 3 ticks (min/mid/max), no extra configuration ----
        cbar = fig.figure.colorbar(m, cax=cax, orientation='vertical')
        try:
            ticks = [vmin, (vmin + vmax) / 2.0, vmax]
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=8)
            cbar.outline.set_linewidth(0.6)
            cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        except Exception:
            pass

        try:
            ticks = [vmin, (vmin + vmax) / 2.0, vmax]
            cbar.set_ticks(ticks)
            cbar.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        except Exception:
            pass

        # store for hover + axis filtering
        self._drt_map_canvas = fig
        self._drt_map_ax = ax
        self._drt_map_cbar_ax = cbar.ax
        self._drt_map_x = xgrid
        self._drt_map_Z = zmat
        self._drt_map_keys = entry_keys
        self._drt_map_hover_last = None

        try:
            fig.mpl_connect('button_press_event', self._on_drt_map_button_press)
        except Exception:
            pass
        try:
            fig.mpl_connect('motion_notify_event', self._on_drt_map_motion)
        except Exception:
            pass

        self._show_canvas_in_plot_panel(fig)
        self._drt_map_scene = self._current_scene
        self._drt_map_proxy = self._current_proxy
        self.ui.plot_panel.show()

    def export_DRT(self):  # callback for exporting the DRT results
        """Export DRT with options: selected, all separate, merged (matrix / 3-cols)."""

        # --- helpers ---
        def _is_fitted_entry(entry) -> bool:
            try:
                return entry is not None and getattr(entry, 'method', 'none') != 'none' and hasattr(entry,
                                                                                                    'out_tau_vec')
            except Exception:
                return False

        def _ui_order_keys():
            return self._get_file_keys_in_ui_order()

        def _get_gamma_for_export(entry):
            """Representative gamma curve consistent with DRT plot/comparison."""
            if entry is None:
                return None
            m = getattr(entry, 'method', 'none')
            if m in ('simple', 'peak'):
                return getattr(entry, 'gamma', None)
            if m == 'credit':
                return getattr(entry, 'gamma', None)  # MAP
            if m == 'BHT':
                return getattr(entry, 'mu_gamma_fine_re', None)
            return getattr(entry, 'gamma', None)

        def _write_single_drt(entry, path: str):
            """Write one file's DRT result (same format as legacy export_DRT)."""
            if not _is_fitted_entry(entry):
                return

            # ensure extension exists if user typed a name without one
            root, ext0 = os.path.splitext(path)
            if ext0 == '':
                path = root + '.csv'

            if entry.method == 'simple':
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.L])
                    writer.writerow(['R', entry.R])
                    writer.writerow(['tau', 'gamma'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n], entry.gamma[n]])

            elif entry.method == 'credit':
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.L])
                    writer.writerow(['R', entry.R])
                    writer.writerow(['tau', 'MAP', 'Mean', 'Upperbound', 'Lowerbound'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n], entry.gamma[n],
                                         entry.mean[n], entry.upper_bound[n], entry.lower_bound[n]])

            elif entry.method == 'BHT':
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.mu_L_0])
                    writer.writerow(['R', entry.mu_R_inf])
                    writer.writerow(['tau', 'gamma_Re', 'gamma_Im'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n],
                                         entry.mu_gamma_fine_re[n],
                                         entry.mu_gamma_fine_im[n]])

            elif entry.method == 'peak':
                # same as simple's main gamma export
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.L])
                    writer.writerow(['R', entry.R])
                    writer.writerow(['tau', 'gamma'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n], entry.gamma[n]])

            # peaks dataframe if present
            if hasattr(entry, 'df'):
                try:
                    df_path = os.path.splitext(path)[0] + '_peaks.csv'
                    entry.df.to_csv(df_path, index=False)
                except Exception:
                    pass

        # --- gather fitted entries ---
        keys_all = _ui_order_keys()
        fitted_keys = []
        fitted_entries = []
        for k in keys_all:
            e = self.data_store.get(k) if hasattr(self, 'data_store') else None
            if _is_fitted_entry(e):
                fitted_keys.append(k)
                fitted_entries.append(e)

        if not fitted_entries:
            return

        current_key = getattr(self, 'current_file_key', None)
        current_entry = self.data_store.get(current_key) if current_key else None
        has_selected = _is_fitted_entry(current_entry)

        # --- dialog UI ---
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export DRT")
        dlg.setModal(True)
        vbox = QtWidgets.QVBoxLayout(dlg)

        rb_selected = QtWidgets.QRadioButton("Export selected file", dlg)
        rb_all = QtWidgets.QRadioButton("Export all fitted files (separate)", dlg)
        rb_merged = QtWidgets.QRadioButton("Export merged (tau & gamma)", dlg)

        merged_type = QtWidgets.QComboBox(dlg)
        merged_type.addItems([
            "Matrix: tau, y1, y2, ...",
            "3 columns: name, tau, gamma"
        ])
        merged_row = QtWidgets.QHBoxLayout()
        merged_row.addWidget(QtWidgets.QLabel("Merged format", dlg))
        merged_row.addWidget(merged_type)

        if has_selected:
            rb_selected.setChecked(True)
        else:
            rb_all.setChecked(True)
            rb_selected.setEnabled(False)

        vbox.addWidget(rb_selected)
        vbox.addWidget(rb_all)
        vbox.addWidget(rb_merged)
        vbox.addLayout(merged_row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg
        )
        vbox.addWidget(btns)

        def _update_enabled():
            merged_type.setEnabled(rb_merged.isChecked())

        rb_selected.toggled.connect(_update_enabled)
        rb_all.toggled.connect(_update_enabled)
        rb_merged.toggled.connect(_update_enabled)
        _update_enabled()

        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # --- do export based on choice ---
        if rb_selected.isChecked():
            if not has_selected:
                return
            base = os.path.splitext(os.path.basename(current_key))[0] + "_fitted"
            ext = os.path.splitext(current_key)[1].lower()
            if ext not in ('.csv', '.txt'):
                ext = '.csv'
            default_path = os.path.join(os.path.dirname(current_key), base + ext)
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save DRT (selected file)",
                default_path,
                "CSV files (*.csv);; TXT files (*.txt)"
            )
            if not path:
                return
            _write_single_drt(current_entry, path)
            return

        if rb_all.isChecked():
            folder = QFileDialog.getExistingDirectory(self, "Select folder to export all fitted DRT files")
            if not folder:
                return
            for k, e in zip(fitted_keys, fitted_entries):
                base = os.path.splitext(os.path.basename(k))[0] + "_fitted"
                ext = os.path.splitext(k)[1].lower()
                if ext not in ('.csv', '.txt'):
                    ext = '.csv'
                out_path = os.path.join(folder, base + ext)
                try:
                    _write_single_drt(e, out_path)
                except Exception:
                    continue
            return

        if rb_merged.isChecked():
            # prepare common tau grid (use first fitted file, interpolate others if needed)
            tau0 = np.asarray(fitted_entries[0].out_tau_vec, dtype=float)
            if tau0.ndim != 1 or tau0.size < 2:
                return

            # X grid in log space for interpolation when needed
            lx0 = np.log10(tau0)

            names = [os.path.splitext(os.path.basename(k))[0] for k in fitted_keys]

            if merged_type.currentIndex() == 0:
                # Matrix: header + tau + each gamma column
                default_name = "DRT_merged_matrix.csv"
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save merged DRT (matrix)",
                    default_name,
                    "CSV files (*.csv)"
                )
                if not path:
                    return
                if not path.lower().endswith('.csv'):
                    path = path + '.csv'

                cols = []
                for e in fitted_entries:
                    g = _get_gamma_for_export(e)
                    if g is None:
                        cols.append(np.full_like(tau0, np.nan, dtype=float))
                        continue
                    g = np.asarray(g, dtype=float)
                    tau = np.asarray(e.out_tau_vec, dtype=float)
                    if tau.shape == tau0.shape and np.allclose(tau, tau0, rtol=0, atol=0):
                        cols.append(g)
                    else:
                        # interpolate onto tau0 in log-x
                        try:
                            lx = np.log10(tau)
                            order = np.argsort(lx)
                            lx = lx[order]
                            g2 = g[order]
                            cols.append(np.interp(lx0, lx, g2))
                        except Exception:
                            cols.append(np.full_like(tau0, np.nan, dtype=float))

                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['tau'] + names)
                    for i in range(tau0.size):
                        row = [tau0[i]] + [float(c[i]) for c in cols]
                        w.writerow(row)
                return

            else:
                # 3 columns: name, tau, gamma (NO log10 on tau, unlike merge_DRT_三列.py conversion)
                default_name = "DRT_merged_3cols.csv"
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save merged DRT (3 columns)",
                    default_name,
                    "CSV files (*.csv)"
                )
                if not path:
                    return
                if not path.lower().endswith('.csv'):
                    path = path + '.csv'

                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    # mimic merge_DRT_三列.py style: no header row
                    for name, e in zip(names, fitted_entries):
                        g = _get_gamma_for_export(e)
                        if g is None:
                            continue
                        tau = np.asarray(e.out_tau_vec, dtype=float)
                        g = np.asarray(g, dtype=float)
                        n = min(tau.size, g.size)
                        for i in range(n):
                            w.writerow([name, tau[i], g[i]])
                return

    def export_EIS(self):  # callback for exporting the EIS fitting results

        # return None if the users have not conducted any computation
        if self.data == None:
            return

        # select path to save the result
        path, ext = QFileDialog.getSaveFileName(None, "Please directory to save the EIS fitting result",
                                                "", "CSV files (*.csv);; TXT files (*.txt)")
        if path == "":  # Check if the path is empty
            return  # Exitthe function if no path is selected

        if self.data.method == 'BHT':  # save result for BHT
            with open(path, 'w', newline='') as save_file:
                writer = csv.writer(save_file)
                writer.writerow(['s_res_re', self.data.out_scores['s_res_re'][0],
                                 self.data.out_scores['s_res_re'][1],
                                 self.data.out_scores['s_res_re'][2]])
                writer.writerow(['s_res_im', self.data.out_scores['s_res_im'][0],
                                 self.data.out_scores['s_res_im'][1],
                                 self.data.out_scores['s_res_im'][2]])
                writer.writerow(['s_mu_re', self.data.out_scores['s_mu_re']])
                writer.writerow(['s_mu_im', self.data.out_scores['s_mu_im']])
                writer.writerow(['s_HD_re', self.data.out_scores['s_HD_re']])
                writer.writerow(['s_HD_im', self.data.out_scores['s_HD_im']])
                writer.writerow(['s_JSD_re', self.data.out_scores['s_JSD_re']])
                writer.writerow(['s_JSD_im', self.data.out_scores['s_JSD_im']])
                writer.writerow(['freq', 'mu_Z_re', 'mu_Z_im', 'mu_H_re', 'mu_H_im',
                                 'Z_H_re_band', 'Z_H_im_band', 'Z_H_re_res', 'Z_H_im_res'])
                # save frequency, the fitted impedance and the residual
                for n in range(self.data.freq.shape[0]):
                    writer.writerow([self.data.freq[n], self.data.mu_Z_re[n],
                                     self.data.mu_Z_im[n], self.data.mu_Z_H_im_agm[n],
                                     self.data.band_re_agm[n], self.data.band_im_agm[n],
                                     self.data.res_H_re[n], self.data.res_H_im[n]])

        else:  # save result for simple and bayesian run
            with open(path, 'w', newline='') as save_file:
                writer = csv.writer(save_file)
                writer.writerow(['freq', 'mu_Z_re', 'mu_Z_im', 'Z_re_res', 'Z_im_res'])
                # save frequency, the fitted impedance and the residual
                for n in range(self.data.freq.shape[0]):
                    writer.writerow([self.data.freq[n], self.data.mu_Z_re[n],
                                     self.data.mu_Z_im[n], self.data.res_re[n],
                                     self.data.res_im[n]])

    def export_fig(self):  # export the figures as png

        # return if users have not conducted any computation
        if self.data == None:
            return

        file_choices = "PNG (*.png)"
        path, ext = QFileDialog.getSaveFileName(self, 'Save file', '', file_choices)

        pixmap = QtGui.QPixmap(self.ui.plot_panel.viewport().size())
        self.ui.plot_panel.viewport().render(pixmap)
        pixmap.save(path)

        if path:
            self.statusBar().showMessage('Saved to %s' % path, 1000)

    def _choose_project_path(self, title='Save Project'):
        default_path = self.current_project_path or ''
        if not default_path and self.current_file_key:
            base = os.path.splitext(os.path.basename(self.current_file_key))[0] + '.sdrtp'
            default_path = os.path.join(os.path.dirname(self.current_file_key), base)
        path, _ = QFileDialog.getSaveFileName(
            self, title, default_path, 'SuperDRTtools Project (*.sdrtp)'
        )
        if not path:
            return None
        if not path.lower().endswith('.sdrtp'):
            path += '.sdrtp'
        return path

    def _run_project_io(self, operation, failure_title):
        if self._project_io_busy:
            return False
        self._project_io_busy = True
        self.setCursor(QtCore.Qt.WaitCursor)
        try:
            QtWidgets.QApplication.processEvents()
            operation()
            return True
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, failure_title, str(exc))
            return False
        finally:
            self.unsetCursor()
            self._project_io_busy = False

    def save_project_callback(self):
        # Existing project -> overwrite it. Normal imported workspace -> first save creates a project.
        path = self.current_project_path
        if not path:
            path = self._choose_project_path('Save Project')
        if not path:
            return False

        try:
            from .project_io import save_project_state
        except Exception:
            from project_io import save_project_state

        def _save():
            save_project_state(path, self._build_project_state())

        if not self._run_project_io(_save, 'Save Project Failed'):
            return False

        self.current_project_path = path
        self._set_project_clean()
        self.statusBar().showMessage(f'Project saved: {path}', 2000)
        return True

    def save_project_as_callback(self):
        path = self._choose_project_path('Save Project As')
        if not path:
            return False

        try:
            from .project_io import save_project_state
        except Exception:
            from project_io import save_project_state

        def _save_as():
            save_project_state(path, self._build_project_state())

        if not self._run_project_io(_save_as, 'Save Project As Failed'):
            return False

        self.current_project_path = path
        self._set_project_clean()
        self.statusBar().showMessage(f'Project saved: {path}', 2000)
        return True

    def _ask_unsaved_changes(self, action_text):
        return QtWidgets.QMessageBox.question(
            self,
            'Unsaved Project',
            f'The current workspace contains unsaved changes.\n\nSave before {action_text}?',
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )

    def open_project_callback(self):
        if self._project_io_busy:
            return False

        if self._project_dirty:
            reply = self._ask_unsaved_changes('opening another project')
            if reply == QtWidgets.QMessageBox.Cancel:
                return False
            if reply == QtWidgets.QMessageBox.Save and not self.save_project_callback():
                return False

        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', '', 'SuperDRTtools Project (*.sdrtp)'
        )
        if not path:
            return False

        try:
            from .project_io import load_project_state
        except Exception:
            from project_io import load_project_state

        loaded = {}
        def _load():
            loaded['state'] = load_project_state(path)

        if not self._run_project_io(_load, 'Open Project Failed'):
            return False

        try:
            self._restore_project_state(loaded['state'])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Open Project Failed', str(exc))
            return False

        self.current_project_path = path
        self._set_project_clean()
        self.statusBar().showMessage(f'Project loaded: {path}', 2000)
        return True

    def closeEvent(self, event):
        if not self._project_dirty:
            event.accept()
            return

        reply = self._ask_unsaved_changes('closing SuperDRTtools')
        if reply == QtWidgets.QMessageBox.Cancel:
            event.ignore()
            return
        if reply == QtWidgets.QMessageBox.Discard:
            event.accept()
            return
        if self.save_project_callback():
            event.accept()
        else:
            event.ignore()

    def _update_project_window_title(self):
        path = getattr(self, 'current_project_path', None)
        if path:
            title = f"SuperDRTtools {os.path.basename(path)}"
        else:
            title = 'SuperDRTtools'
        if self._project_dirty:
            title += ' *'
        self.setWindowTitle(title)


class Figure_Canvas(FigureCanvas):

    def __init__(self, parent=None, width=7.5, height=6.5, dpi=100):  # create Figure under matplotlib.pyplot

        # deactivate the popping of another figure panel
        plt.ioff()

        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=20)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.close('all')
        fig = plt.figure(figsize=(width, height), dpi=dpi)

        # initalizing parent
        FigureCanvas.__init__(self, fig)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()
        self.setParent(parent)
        self.axes = fig.add_subplot(111)  # using add_subplot method
        fig.tight_layout()
        fig.subplots_adjust(left=0.14, bottom=0.13, right=0.9, top=0.9)
        try:
            self.mpl_connect('scroll_event', self._on_scroll_zoom)
        except Exception:
            pass
        try:
            self.mpl_connect('motion_notify_event', self._on_general_motion)
            self.mpl_connect('figure_leave_event', self._on_general_leave)
        except Exception:
            pass
        self._kk_hover_payload = None
        self._kk_hover_annotation = None
        self._eis_hover_payload = None
        self._eis_plot_payload = None

    @staticmethod
    def _zoom_axis_limits(cur_min, cur_max, center, scale_factor, log_scale=False):
        if not np.isfinite(cur_min) or not np.isfinite(cur_max) or cur_min == cur_max:
            return cur_min, cur_max
        if center is None or not np.isfinite(center):
            center = (cur_min + cur_max) / 2.0
        if log_scale:
            if cur_min <= 0 or cur_max <= 0 or center <= 0:
                return cur_min, cur_max
            lmin, lmax, lc = np.log10(cur_min), np.log10(cur_max), np.log10(center)
            new_min = 10.0 ** (lc - (lc - lmin) * scale_factor)
            new_max = 10.0 ** (lc + (lmax - lc) * scale_factor)
            return new_min, new_max
        new_min = center - (center - cur_min) * scale_factor
        new_max = center + (cur_max - center) * scale_factor
        return new_min, new_max

    def _on_scroll_zoom(self, event):
        ax = getattr(event, 'inaxes', None)
        if ax is None:
            return
        base_scale = 1.2
        scale_factor = 1.0 / base_scale if getattr(event, 'button', None) == 'up' else base_scale

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_center = event.xdata if getattr(event, 'xdata', None) is not None else (x0 + x1) / 2.0
        y_center = event.ydata if getattr(event, 'ydata', None) is not None else (y0 + y1) / 2.0

        x_reversed = x0 > x1
        y_reversed = y0 > y1
        x_low, x_high = (x1, x0) if x_reversed else (x0, x1)
        y_low, y_high = (y1, y0) if y_reversed else (y0, y1)

        new_x_low, new_x_high = self._zoom_axis_limits(x_low, x_high, x_center, scale_factor,
                                                       log_scale=(ax.get_xscale() == 'log'))
        new_y_low, new_y_high = self._zoom_axis_limits(y_low, y_high, y_center, scale_factor,
                                                       log_scale=(ax.get_yscale() == 'log'))
        ax.set_xlim((new_x_high, new_x_low) if x_reversed else (new_x_low, new_x_high))
        ax.set_ylim((new_y_high, new_y_low) if y_reversed else (new_y_low, new_y_high))
        try:
            self.draw_idle()
        except Exception:
            self.draw()

    @staticmethod
    def _format_hover_frequency(value: float) -> str:
        try:
            v = float(value)
        except Exception:
            return str(value)
        if v == 0:
            return '0'
        if 1e-2 <= abs(v) < 1e4:
            out = f"{v:.3f}"
            out = out.rstrip('0').rstrip('.')
            return out
        return f"{v:.3e}"

    def _hide_kk_hover(self):
        try:
            QtWidgets.QToolTip.hideText()
        except Exception:
            pass

    def _on_general_motion(self, event):
        eis_payload = getattr(self, '_eis_hover_payload', None)
        if eis_payload and getattr(event, 'inaxes', None) is self.axes:
            xs = np.asarray(eis_payload.get('x', []), dtype=float)
            ys = np.asarray(eis_payload.get('y', []), dtype=float)
            if xs.size:
                pts = np.column_stack([xs, ys])
                try:
                    disp = self.axes.transData.transform(pts)
                except Exception:
                    disp = None
                if disp is not None and disp.size:
                    dist2 = (disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2
                    idx = int(np.argmin(dist2))
                    d = float(np.sqrt(dist2[idx]))
                    if d <= 12.0:
                        freq_val = float(np.asarray(eis_payload.get('freq', []), dtype=float)[idx])
                        x_val = float(xs[idx])
                        y_val = float(ys[idx])
                        re_res = np.asarray(eis_payload.get('res_re', []), dtype=float)
                        im_res = np.asarray(eis_payload.get('res_im', []), dtype=float)
                        masked = np.asarray(eis_payload.get('masked', []), dtype=bool)
                        re_txt = 'N/A' if idx >= re_res.size or not np.isfinite(re_res[idx]) else f"{re_res[idx]:.2f}%"
                        im_txt = 'N/A' if idx >= im_res.size or not np.isfinite(im_res[idx]) else f"{im_res[idx]:.2f}%"
                        masked_txt = 'Yes' if idx < masked.size and bool(masked[idx]) else 'No'
                        text = (
                            f"Frequency: {self._format_hover_frequency(freq_val)} Hz\n"
                            f"Z': {x_val:.4g} Ω\n"
                            f"-Z'': {y_val:.4g} Ω\n"
                            f"KK Re residual: {re_txt}\n"
                            f"KK Im residual: {im_txt}\n"
                            f"Masked: {masked_txt}"
                        )
                        try:
                            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
                        except Exception:
                            pass
                        return

        payload = getattr(self, '_kk_hover_payload', None)
        if not payload or getattr(event, 'inaxes', None) is not self.axes:
            self._hide_kk_hover()
            return

        best = None
        threshold_px = 15.0
        for label, xs, ys in payload:
            pts = np.column_stack([xs, ys])
            try:
                disp = self.axes.transData.transform(pts)
            except Exception:
                continue
            if disp.size == 0:
                continue
            dist2 = (disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2
            idx = int(np.argmin(dist2))
            d = float(np.sqrt(dist2[idx]))
            if best is None or d < best[0]:
                best = (d, label, float(xs[idx]), float(ys[idx]))

        if best is None or best[0] > threshold_px:
            self._hide_kk_hover()
            return

        _, label, x_val, y_val = best
        text = (
            f"Frequency: {self._format_hover_frequency(x_val)} Hz\n"
            f"{label}: {y_val:.2f}%"
        )
        try:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
        except Exception:
            pass

    def _on_general_leave(self, event):
        self._hide_kk_hover()

    def EIS_data(self, entry):  # plot the data
        legend_needed = False
        self._kk_hover_payload = None
        self._eis_hover_payload = None
        self._eis_plot_payload = None

        if entry.method == 'BHT':  # for BHT run
            self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im,
                           'k', label='$Z_\mu$(Regressed)', linewidth=3)
            self.axes.plot(entry.mu_Z_H_re_agm, -entry.mu_Z_H_im_agm,
                           'b', label='$Z_H$(Hilbert transform)', linewidth=3)
            legend_needed = True

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im, 'k', linewidth=2, label='DRT fit')
            legend_needed = True

        if getattr(entry, 'kk_valid', False) and hasattr(entry, 'kk_Z_fit'):
            self.axes.plot(entry.kk_Z_fit.real, -entry.kk_Z_fit.imag,
                           color='orange', linewidth=2.5, label='K-K fit')
            legend_needed = True

        freq0 = np.asarray(getattr(entry, 'freq_0', getattr(entry, 'freq', [])), dtype=float).reshape(-1)
        zre0 = np.asarray(getattr(entry, 'Z_prime_0', getattr(entry, 'Z_prime', [])), dtype=float).reshape(-1)
        zim0 = np.asarray(getattr(entry, 'Z_double_prime_0', getattr(entry, 'Z_double_prime', [])), dtype=float).reshape(-1)
        visible_keep = np.asarray(getattr(entry, 'visible_keep_raw', np.ones(freq0.size, dtype=bool)), dtype=bool).reshape(-1)
        if visible_keep.size != freq0.size:
            visible_keep = np.ones(freq0.size, dtype=bool)
        masked = np.asarray(getattr(entry, 'mask_total_raw', np.zeros(freq0.size, dtype=bool)), dtype=bool).reshape(-1)
        if masked.size != freq0.size:
            masked = np.zeros(freq0.size, dtype=bool)
        kk_re = np.asarray(getattr(entry, 'kk_res_re_raw_pct', np.full(freq0.size, np.nan)), dtype=float).reshape(-1)
        kk_im = np.asarray(getattr(entry, 'kk_res_im_raw_pct', np.full(freq0.size, np.nan)), dtype=float).reshape(-1)
        if kk_re.size != freq0.size:
            kk_re = np.full(freq0.size, np.nan, dtype=float)
        if kk_im.size != freq0.size:
            kk_im = np.full(freq0.size, np.nan, dtype=float)
        kk_max = _safe_kk_residual_max(kk_re, kk_im, freq0.size)

        disp_idx = np.where(visible_keep)[0].astype(int)
        if disp_idx.size:
            severe = np.zeros(disp_idx.size, dtype=int)
            kk_disp = kk_max[disp_idx]
            severe[(np.isfinite(kk_disp)) & (kk_disp > 1.0) & (kk_disp <= 5.0)] = 1
            severe[(np.isfinite(kk_disp)) & (kk_disp > 5.0)] = 2
            masked_disp = masked[disp_idx]

            labels = {0: 'Experimental data', 1: 'KK residual > 1%', 2: 'KK residual > 5%'}
            colors = {0: '#D62728', 1: '#FF8C00', 2: '#7A3DB8'}
            for sev in (0, 1, 2):
                idx_vis = disp_idx[(severe == sev) & (~masked_disp)]
                if idx_vis.size:
                    self.axes.scatter(zre0[idx_vis], -zim0[idx_vis], s=34, c=colors[sev], alpha=0.95,
                                      edgecolors='none', label=labels[sev])
                    legend_needed = True
            for sev in (0, 1, 2):
                idx_mask = disp_idx[(severe == sev) & masked_disp]
                if idx_mask.size:
                    self.axes.scatter(zre0[idx_mask], -zim0[idx_mask], s=34, c=colors[sev], alpha=0.25,
                                      edgecolors='none', label='Masked point' if sev == 0 else None)
                    legend_needed = True

            self._eis_hover_payload = {
                'raw_indices': disp_idx,
                'x': zre0[disp_idx],
                'y': -zim0[disp_idx],
                'freq': freq0[disp_idx],
                'res_re': kk_re[disp_idx],
                'res_im': kk_im[disp_idx],
                'masked': masked[disp_idx],
            }
            self._eis_plot_payload = self._eis_hover_payload

        if legend_needed:
            handles, labels = self.axes.get_legend_handles_labels()
            unique = {}
            for h, lab in zip(handles, labels):
                if lab and lab not in unique:
                    unique[lab] = h
            self.axes.legend(list(unique.values()), list(unique.keys()), frameon=False, fontsize=11, loc='best')

        self.axes.set_xlabel('$Z^{\prime}/\Omega$')
        self.axes.set_ylabel('-$Z^{\prime \prime}/\Omega$')
        self.axes.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        self.axes.axis('equal')

    def Magnitude(self, entry):  # plot the magnitude

        if entry.method == 'BHT':  # for BHT run
            self.axes.semilogx(entry.freq, absolute(entry.mu_Z_re + 1j * entry.mu_Z_im),
                               'k', label='$Z_\mu$(Regressed)', linewidth=2)
            self.axes.semilogx(entry.freq, absolute(entry.mu_Z_H_re_agm + 1j * entry.mu_Z_H_im_agm),
                               'b', label='$Z_H$(Hilbert transform)', linewidth=2)
            self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.semilogx(entry.freq, absolute(entry.mu_Z_re + 1j * entry.mu_Z_im), 'k', linewidth=3)
            self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('$|Z|/\Omega$')

    def Phase(self, entry):  # plot the phase

        if entry.method == 'BHT':  # for BHT run
            self.axes.semilogx(entry.freq, angle(entry.mu_Z_re + 1j * entry.mu_Z_im, deg=True),
                               'k', label='$Z_\mu$(Regressed)', linewidth=2)
            self.axes.semilogx(entry.freq, angle(entry.mu_Z_H_re_agm + 1j * entry.mu_Z_H_im_agm, deg=True),
                               'b', label='$Z_H$(Hilbert transform)', linewidth=2)
            self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg=True), 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.semilogx(entry.freq, angle(entry.mu_Z_re + 1j * entry.mu_Z_im, deg=True), 'k', linewidth=3)
            self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg=True), 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg=True), 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('$angle/^\circ$')

    def Re_data(self, entry):  # plot the real part

        if entry.method == 'BHT':  # for BHT run
            self.axes.fill_between(entry.freq, entry.mu_Z_H_re_agm - 3 * entry.band_re_agm,
                                   entry.mu_Z_H_re_agm + 3 * entry.band_re_agm, facecolor='lightgrey')
            self.axes.semilogx(entry.freq, entry.mu_Z_re, 'k', label='$Z_\mu$(Regressed)', linewidth=3)
            self.axes.semilogx(entry.freq, entry.mu_Z_H_re_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
            self.axes.semilogx(entry.freq, entry.Z_prime, 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.semilogx(entry.freq, entry.mu_Z_re, 'k', linewidth=3)
            self.axes.semilogx(entry.freq, entry.Z_prime, 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, entry.Z_prime, 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('$Z^{\prime}/\Omega$')

    def Im_data(self, entry):  # plot of the imaginary part

        if entry.method == 'BHT':  # for BHT run
            self.axes.fill_between(entry.freq, -entry.mu_Z_H_im_agm - 3 * entry.band_im_agm,
                                   -entry.mu_Z_H_im_agm + 3 * entry.band_im_agm, facecolor='lightgrey')
            self.axes.semilogx(entry.freq, -entry.mu_Z_im, 'k', label='$Z_\mu$(Regressed)', linewidth=3)
            self.axes.semilogx(entry.freq, -entry.mu_Z_H_im_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
            self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or bayesian run
            self.axes.semilogx(entry.freq, -entry.mu_Z_im, 'k')
            self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('-$Z^{\prime \prime}/\Omega$')

    def DRT_residual(self, entry):  # plot DRT residual analysis

        self._kk_hover_payload = None

        if entry.method == 'none':
            self.axes.text(0.5, 0.5, 'Run DRT fitting to view residuals.',
                           ha='center', va='center', transform=self.axes.transAxes)
            self.axes.set_axis_off()
            return

        freq = np.asarray(entry.freq, dtype=float)
        z_abs = np.asarray(np.abs(entry.Z_exp), dtype=float)
        if z_abs.size != freq.size:
            z_abs = np.ones_like(freq, dtype=float)
        z_abs_safe = np.where(np.isfinite(z_abs) & (z_abs > 0), z_abs, np.nan)

        if entry.method == 'BHT':
            res_re = np.asarray(entry.res_H_re, dtype=float)
            res_im = np.asarray(entry.res_H_im, dtype=float)
        else:
            res_re = np.asarray(entry.res_re, dtype=float)
            res_im = np.asarray(entry.res_im, dtype=float)

        res_re_pct = 100.0 * res_re / z_abs_safe
        res_im_pct = 100.0 * res_im / z_abs_safe

        self.axes.semilogx(freq, res_re_pct, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Re}}$')
        self.axes.semilogx(freq, res_im_pct, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Im}}$')

        y_max = float(np.nanmax(np.abs(np.concatenate([res_re_pct, res_im_pct]))))
        if (not np.isfinite(y_max)) or y_max <= 0:
            y_max = 1.0

        self.axes.set_xlim([float(np.max(freq)), float(np.min(freq))])
        self.axes.set_ylim([-1.15 * y_max, 1.15 * y_max])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel(r'$\Delta\,(\%)$')
        self.axes.axhline(0.0, color='0.5', linewidth=0.8)
        self.axes.grid(True, which='both', linestyle='--', alpha=0.35)
        self.axes.legend(frameon=False, fontsize=12, loc='best')

        self._kk_hover_payload = [
            ('Real Residual', freq, res_re_pct),
            ('Imag Residual', freq, res_im_pct),
        ]

    def Re_residual(self, entry):
        self.DRT_residual(entry)

    def Im_residual(self, entry):
        self.DRT_residual(entry)

    def KK_residual(self, entry):  # plot K-K residual analysis

        self._kk_hover_payload = None

        if not getattr(entry, 'kk_valid', False) or not hasattr(entry, 'kk_res_re_pct'):
            self.axes.text(0.5, 0.5, 'Run K-K analysis to view residuals.',
                           ha='center', va='center', transform=self.axes.transAxes)
            self.axes.set_axis_off()
            return

        freq = np.asarray(entry.freq, dtype=float)
        res_re = np.asarray(entry.kk_res_re_pct, dtype=float)
        res_im = np.asarray(entry.kk_res_im_pct, dtype=float)

        self.axes.semilogx(freq, res_re, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Re}}$')
        self.axes.semilogx(freq, res_im, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Im}}$')

        y_max = float(np.nanmax(np.abs(np.concatenate([res_re, res_im]))))
        if (not np.isfinite(y_max)) or y_max <= 0:
            y_max = 1.0

        self.axes.set_xlim([float(np.max(freq)), float(np.min(freq))])
        self.axes.set_ylim([-1.15 * y_max, 1.15 * y_max])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel(r'$\Delta\,(\%)$')
        self.axes.axhline(0.0, color='0.5', linewidth=0.8)
        self.axes.grid(True, which='both', linestyle='--', alpha=0.35)
        self.axes.legend(frameon=False, fontsize=12, loc='best')

        self._kk_hover_payload = [
            ('Real Residual', freq, res_re),
            ('Imag Residual', freq, res_im),
        ]

    def DRT_data(self, entry):  # plot the DRT

        if entry.method == 'none':
            return

        elif entry.method == 'simple':
            self.axes.semilogx(entry.out_tau_vec, entry.gamma, 'k', linewidth=3)
            y_min = 0
            y_max = max(entry.gamma)

        elif entry.method == 'credit':
            self.axes.fill_between(entry.out_tau_vec, entry.lower_bound, entry.upper_bound, facecolor='lightgrey')
            self.axes.semilogx(entry.out_tau_vec, entry.gamma, color='black', label='MAP', linewidth=3)
            self.axes.semilogx(entry.out_tau_vec, entry.mean, color='blue', label='mean', linewidth=3)
            self.axes.semilogx(entry.out_tau_vec, entry.lower_bound, color='black')
            self.axes.semilogx(entry.out_tau_vec, entry.upper_bound, color='black')
            self.axes.legend(frameon=False, fontsize=15)
            y_min = 0
            y_max = max(entry.upper_bound)

        elif entry.method == 'BHT':
            self.axes.semilogx(entry.out_tau_vec, entry.mu_gamma_fine_re, 'b', linewidth=3, label='$Mean Re$')
            self.axes.semilogx(entry.out_tau_vec, entry.mu_gamma_fine_im, 'k', linewidth=3, label='$Mean Im$')
            y_min = min(np.concatenate((entry.mu_gamma_fine_re, entry.mu_gamma_fine_im)))
            y_max = max(np.concatenate((entry.mu_gamma_fine_re, entry.mu_gamma_fine_im)))

        elif entry.method == 'peak':

            self.axes.semilogx(entry.out_tau_vec, entry.gamma, color='black', linewidth=3)
            color = ['red', 'green', 'cyan', 'yellow', 'orange', 'blue', 'grey', 'brown', 'coral', 'darkblue',
                     'darkgreen', 'gold']

            if len(entry.out_gamma_fit) == entry.N_peaks:  # separate fit of the peaks
                for i in range(entry.N_peaks):
                    self.axes.semilogx(entry.out_tau_vec, entry.out_gamma_fit[i], color=color[i], linewidth=3)

            else:  # combine fit of the DRT peaks
                self.axes.semilogx(entry.out_tau_vec, entry.out_gamma_fit, color='green', linewidth=3)
            y_min = 0
            y_max = max(entry.gamma)

        self.axes.set_xlabel(r'$\tau/s$')
        self.axes.set_ylabel(r'$\gamma( \tau)/\Omega$')
        self.axes.set_ylim([y_min, 1.1 * y_max])
        self.axes.set_xlim([min(entry.out_tau_vec), max(entry.out_tau_vec)])

    def DRT_comparison(self, entries, colors, y_lim=None, keys=None):
        """Overlay multiple DRT curves in the same axes (log-x)."""
        if not entries:
            return

        # expose plotted lines for GUI interactions (pick/highlight)
        self._drt_comp_lines = []

        if keys is None:
            keys = [None] * len(entries)

        xs_min = None
        xs_max = None
        ymins = []
        ymaxs = []

        for entry, color, key in zip(entries, colors, keys):
            if getattr(entry, 'method', 'none') == 'none':
                continue
            if not hasattr(entry, 'out_tau_vec'):
                continue

            x = entry.out_tau_vec

            # Choose a single representative curve per file (matching DRT plot intent)
            y = None
            if entry.method in ('simple', 'peak'):
                y = getattr(entry, 'gamma', None)
            elif entry.method == 'credit':
                y = getattr(entry, 'gamma', None)  # MAP curve
            elif entry.method == 'BHT':
                y = getattr(entry, 'mu_gamma_fine_re', None)
            else:
                y = getattr(entry, 'gamma', None)

            if y is None:
                continue

            line, = self.axes.semilogx(x, y, color=color, linewidth=2, picker=5)
            self._drt_comp_lines.append(line)
            if key is not None:
                try:
                    line.set_gid(key)
                except Exception:
                    pass

            xs_min = float(np.min(x)) if xs_min is None else min(xs_min, float(np.min(x)))
            xs_max = float(np.max(x)) if xs_max is None else max(xs_max, float(np.max(x)))
            ymins.append(float(np.min(y)))
            ymaxs.append(float(np.max(y)))

        if xs_min is None or xs_max is None or not ymins or not ymaxs:
            return

        self.axes.set_xlabel(r'$\tau/s$')
        self.axes.set_ylabel(r'$\gamma( \tau)/\Omega$')
        self.axes.set_xlim([xs_min, xs_max])

        auto_ymin = min(ymins)
        auto_ymax = max(ymaxs)
        # mimic single DRT plot: start at 0 if all positive
        if auto_ymin > 0:
            auto_ymin = 0.0

        if y_lim is not None:
            ymin, ymax = y_lim
            if ymin is None:
                ymin = auto_ymin
            if ymax is None:
                ymax = auto_ymax * 1.1
            self.axes.set_ylim([ymin, ymax])
        else:
            self.axes.set_ylim([auto_ymin, auto_ymax * 1.1])

    def Score(self, entry):  # plot the EIS score

        if entry.method != 'BHT':
            return

        else:  # for BHT method
            Re_part = np.array(
                [entry.out_scores['s_res_re'][0], entry.out_scores['s_res_re'][1], entry.out_scores['s_res_re'][2],
                 entry.out_scores['s_mu_re'], entry.out_scores['s_HD_re'], entry.out_scores['s_JSD_re']])
            Im_part = np.array(
                [entry.out_scores['s_res_im'][0], entry.out_scores['s_res_im'][1], entry.out_scores['s_res_im'][2],
                 entry.out_scores['s_mu_im'], entry.out_scores['s_HD_im'], entry.out_scores['s_JSD_im']])
            x = np.arange(6)  # the label locations
            width = 0.35  # the width of the bars

            self.axes.bar(x - width / 2, Re_part * 100, width, label='Re part', color='blue')
            self.axes.bar(x + width / 2, Im_part * 100, width, label='Im part', color='black')
            self.axes.plot([-0.5, 5.5], [100, 100], '--k')

            self.axes.legend(frameon=False, fontsize=15)
            self.axes.set_ylim([0, 125])
            self.axes.set_xlim([-0.5, 5.5])
            self.axes.set_xticks(x)
            self.axes.set_xticklabels(
                (r'$s_{1\sigma}$', r'$s_{2\sigma}$', r'$s_{3\sigma}$', r'$s_{\mu}$', r'$s_{\rm HD}$', r'$s_{\rm JSD}$'))
            self.axes.set_yticks([0, 50, 100])
            self.axes.set_ylabel(r'$\rm Scores (\%)$')


if __name__ == "__main__":  # starting the GUI when users run this file

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = GUI()
    MainWindow.show()
    sys.exit(app.exec_())

# ---------------- __SIDEBAR_TAB_REFACTOR__ ----------------
# Safe runtime refactor: does NOT modify GUI class source structure.
# We attach helper methods onto GUI after class definition, then call them from launch_gui.

_NOISE_TEXTS = ("Options for RBF", "Options for kBL", "Options for KBL")

# ---- Tuning knobs (you can change these numbers manually) ----
TOPBAR_GUTTER = 0  # kept for compatibility; top bar now uses layouts
PLOT_SHIFT_DX = 0  # kept for compatibility; plot area now uses layouts


def _set_label_style_clean(widget: QtWidgets.QWidget) -> None:
    for lab in widget.findChildren(QtWidgets.QLabel):
        try:
            lab.setStyleSheet("background: transparent;")
        except Exception:
            pass


def _text_contains_noise(t: str) -> bool:
    if not t:
        return False
    ts = t.strip()
    for n in _NOISE_TEXTS:
        if n in ts:
            return True
    return False


def _hide_noise_global(root: QtWidgets.QWidget, allow_groupboxes=None) -> None:
    """Hide ghost labels AND ghost groupboxes (titles) from old absolute layout."""
    allow_groupboxes = allow_groupboxes or set()

    for lab in root.findChildren(QtWidgets.QLabel):
        try:
            if _text_contains_noise(lab.text()):
                lab.hide()
        except Exception:
            pass

    for gb in root.findChildren(QtWidgets.QGroupBox):
        try:
            if gb in allow_groupboxes:
                continue
            if _text_contains_noise(gb.title()):
                gb.hide()
        except Exception:
            pass


def _hide_top_bar_frames(root: QtWidgets.QWidget) -> None:
    try:
        for fr in root.findChildren(QtWidgets.QFrame):
            if fr.frameShape() in (QtWidgets.QFrame.HLine, QtWidgets.QFrame.VLine, QtWidgets.QFrame.Box,
                                   QtWidgets.QFrame.Panel):
                fr.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass


def _clear_layout_widget(parent: QtWidgets.QWidget) -> None:
    lay = parent.layout()
    if lay is None:
        return
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(parent)
    QtWidgets.QWidget().setLayout(lay)


def _hide_unmanaged_children(container: QtWidgets.QWidget, keep: set) -> None:
    for ch in container.findChildren((QtWidgets.QLabel, QtWidgets.QFrame, QtWidgets.QGroupBox)):
        if ch in keep:
            continue
        try:
            if isinstance(ch, QtWidgets.QGroupBox) and _text_contains_noise(ch.title()):
                ch.hide()
            elif isinstance(ch, QtWidgets.QLabel) and _text_contains_noise(ch.text()):
                ch.hide()
            elif isinstance(ch, QtWidgets.QFrame):
                ch.hide()
            else:
                ch.hide()
        except Exception:
            pass


def _ensure_field_width(widget: QtWidgets.QWidget, min_w: int = 120) -> None:
    try:
        widget.setMinimumWidth(min_w)
        widget.setMaximumWidth(16777215)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    except Exception:
        pass


def _fix_top_bar_position(self) -> None:
    """Keep the top tab bar expanding naturally inside the center layout."""
    bar = getattr(self.ui, "show_layout", None)
    if bar is None:
        return
    try:
        bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        bar.setMinimumHeight(58)
        bar.setMaximumHeight(72)
        bar.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass
    try:
        tabs = bar.findChild(QtWidgets.QTabWidget, "TopTabs")
        if tabs is not None:
            tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            tabs.updateGeometry()
    except Exception:
        pass


def _shift_plot_panel_right(self, dx: int = None) -> None:
    """Add a gutter between sidebar and plot area. Uses plot_panel geometry if available."""
    if dx is None:
        dx = PLOT_SHIFT_DX
    plot = getattr(self.ui, "plot_panel", None)
    if plot is None:
        return
    try:
        # If plot is layout-managed, moving may be ignored; we also add an internal left margin.
        plot.move(plot.x() + dx, plot.y())
        plot.resize(max(200, plot.width() - dx), plot.height())
    except Exception:
        pass
    try:
        lay = plot.layout()
        if lay is not None:
            m = lay.contentsMargins()
            lay.setContentsMargins(m.left() + dx, m.top(), m.right(), m.bottom())
    except Exception:
        pass
    try:
        plot.setFrameShape(QtWidgets.QFrame.NoFrame)
    except Exception:
        pass


def _force_combobox_popup(self) -> None:
    """Force QComboBox popup to use QListView with a safe palette so items are visible on all themes."""
    try:
        from PyQt5 import QtGui as _QtGui
    except Exception:
        _QtGui = None

    combos = self.findChildren(QtWidgets.QComboBox)
    for cb in combos:
        try:
            view = QtWidgets.QListView()
            view.setUniformItemSizes(True)
            view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            # Palette: ensure text is dark on white
            if _QtGui is not None:
                pal = view.palette()
                pal.setColor(_QtGui.QPalette.Base, _QtGui.QColor(255, 255, 255))
                pal.setColor(_QtGui.QPalette.Text, _QtGui.QColor(29, 29, 31))
                pal.setColor(_QtGui.QPalette.WindowText, _QtGui.QColor(29, 29, 31))
                pal.setColor(_QtGui.QPalette.Highlight, _QtGui.QColor(10, 132, 255))
                pal.setColor(_QtGui.QPalette.HighlightedText, _QtGui.QColor(255, 255, 255))
                view.setPalette(pal)

            view.setStyleSheet(
                "QListView{background:#FFFFFF;color:#1D1D1F;border:0px solid #E5E5EA;border-radius:10px;}"
                "QListView::item{color:#1D1D1F;padding:6px 10px;min-height:28px;}"
                "QListView::item:selected{background:#0A84FF;color:#FFFFFF;border-radius: 8px;}"
            )
            cb.setView(view)

            # Ensure the combobox itself uses a readable palette
            if _QtGui is not None:
                pal2 = cb.palette()
                pal2.setColor(_QtGui.QPalette.ButtonText, _QtGui.QColor(29, 29, 31))
                pal2.setColor(_QtGui.QPalette.Text, _QtGui.QColor(29, 29, 31))
                cb.setPalette(pal2)
        except Exception:
            pass


def _install_resize_hook(self) -> None:
    if getattr(self, "_safe_refactor_resize_hook_installed", False):
        return
    self._safe_refactor_resize_hook_installed = True

    old_resize = getattr(self, "resizeEvent", None)

    self._plot_resize_timer = QtCore.QTimer(self)
    self._plot_resize_timer.setSingleShot(True)
    self._plot_resize_timer.timeout.connect(self._refresh_current_plot_size)

    def _new_resize_event(evt):
        try:
            if callable(old_resize):
                old_resize(evt)
        finally:
            try:
                _fix_top_bar_position(self)
            except Exception:
                pass
            try:
                self._plot_resize_timer.start(80)
            except Exception:
                pass

    try:
        self.resizeEvent = _new_resize_event
    except Exception:
        pass


def _refactor_ui_layout(self) -> None:
    try:
        _set_label_style_clean(self)
        allow = set()
        for name in ("RBF_frame", "settings_layout", "run_layout", "Peak_analysis_frame", "export_frame"):
            gb = getattr(self.ui, name, None)
            if isinstance(gb, QtWidgets.QGroupBox):
                allow.add(gb)

        _hide_noise_global(self, allow_groupboxes=allow)
        _hide_top_bar_frames(self)

        _rebuild_sidebar_with_forms(self)

        _hide_noise_global(self, allow_groupboxes=allow)

        _replace_show_buttons_with_tabs(self)
        _fix_top_bar_position(self)
        _install_resize_hook(self)

        # Ensure ComboBox popups render text (fix blank dropdown list)
        _force_combobox_popup(self)

        _hide_noise_global(self, allow_groupboxes=allow)
    except Exception:
        pass
    # 在 _refactor_ui_layout(self) 的最后（比如 _fix_top_bar_position 之后）加：
    try:
        self.ui.plot_panel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ui.plot_panel.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass


def _rebuild_sidebar_with_forms(self) -> None:
    host = getattr(self.ui, "General_frame", None)
    if host is None:
        return

    _clear_layout_widget(host)

    scroll = QtWidgets.QScrollArea(host)
    scroll.setObjectName("SidebarScroll")
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    host_layout = QtWidgets.QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.setSpacing(0)
    host_layout.addWidget(scroll, 1)

    container = QtWidgets.QWidget()
    container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
    vbox = QtWidgets.QVBoxLayout(container)
    vbox.setContentsMargins(12, 12, 12, 12)
    vbox.setSpacing(12)

    groups = [
        getattr(self.ui, "settings_layout", None),
        getattr(self.ui, "RBF_frame", None),
        getattr(self.ui, "KK_frame", None),
        getattr(self.ui, "run_layout", None),
        getattr(self.ui, "Peak_analysis_frame", None),
        getattr(self.ui, "export_frame", None),
    ]
    groups = [g for g in groups if g is not None]

    for gb in groups:
        gb.setParent(container)
        gb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        vbox.addWidget(gb)
    vbox.addStretch(1)

    scroll.setWidget(container)

    host.setMinimumWidth(400)
    host.setMaximumWidth(600)
    host.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

    if hasattr(self.ui, "settings_layout"):
        _layout_settings_group(self)
    if hasattr(self.ui, "RBF_frame"):
        _layout_rbf_group(self)
    if hasattr(self.ui, "KK_frame"):
        _layout_kk_group(self)
    if hasattr(self.ui, "run_layout"):
        _layout_run_group(self)
    if hasattr(self.ui, "Peak_analysis_frame"):
        _layout_peak_group(self)
    if hasattr(self.ui, "export_frame"):
        _layout_export_group(self)

    class NoWheelFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Wheel:
                event.ignore()
                return True
            return super().eventFilter(obj, event)

    self._no_wheel_filter = NoWheelFilter(self)

    for cb in self.findChildren(QtWidgets.QComboBox):
        cb.installEventFilter(self._no_wheel_filter)


def _layout_settings_group(self) -> None:
    gb = self.ui.settings_layout
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)

    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    labels = [
        self.ui.import_label,
        self.ui.discre_label,
        self.ui.data_used_label,
        self.ui.induct_label,
        self.ui.der_label,
        self.ui.lambda_choice_label,
        self.ui.reg_param_label,
        self.ui.reg_param_label_2,
        self.ui.sample_no,
    ]

    for lab in labels:
        lab.setMinimumWidth(120)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        lab.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

    fields = [
        self.ui.discre_choice,
        self.ui.data_used_choice,
        self.ui.induct_choice,
        self.ui.der_choice,
        self.ui.lambda_choice,
        self.ui.reg_param_entry,
        self.ui.reg_param_entry_2,
        self.ui.sample_no_entry,
    ]
    for w in fields:
        w.setMinimumHeight(30)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.import_button.setFixedHeight(30)
    self.ui.import_button.setFixedWidth(120)
    self.ui.import_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    keep = set(labels)
    _hide_unmanaged_children(gb, keep)

    row = 0
    import_row = QtWidgets.QHBoxLayout()
    import_row.addStretch(1)
    import_row.addWidget(self.ui.import_button)
    grid.addWidget(self.ui.import_label, 0, 0)
    grid.addLayout(import_row, 0, 1)
    row += 1

    grid.addWidget(self.ui.discre_label, row, 0)
    grid.addWidget(self.ui.discre_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.data_used_label, row, 0)
    grid.addWidget(self.ui.data_used_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.induct_label, row, 0)
    grid.addWidget(self.ui.induct_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.der_label, row, 0)
    grid.addWidget(self.ui.der_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.lambda_choice_label, row, 0)
    grid.addWidget(self.ui.lambda_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.reg_param_label, row, 0)
    grid.addWidget(self.ui.reg_param_entry, row, 1)
    row += 1

    grid.addWidget(self.ui.reg_param_label_2, row, 0)
    grid.addWidget(self.ui.reg_param_entry_2, row, 1)
    row += 1

    grid.addWidget(self.ui.sample_no, row, 0)
    grid.addWidget(self.ui.sample_no_entry, row, 1)


def _layout_rbf_group(self) -> None:
    gb = self.ui.RBF_frame
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    for lab in [self.ui.shape_control_label, self.ui.FWHM_control_label]:
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    for w in [self.ui.shape_control_choice, self.ui.FWHM_entry]:
        w.setMinimumHeight(30)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    keep = {self.ui.shape_control_label, self.ui.FWHM_control_label}
    _hide_unmanaged_children(gb, keep)

    grid.addWidget(self.ui.shape_control_label, 0, 0)
    grid.addWidget(self.ui.shape_control_choice, 0, 1)
    grid.addWidget(self.ui.FWHM_control_label, 1, 0)
    grid.addWidget(self.ui.FWHM_entry, 1, 1)


def _layout_kk_group(self) -> None:
    gb = self.ui.KK_frame
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    for lab in [
        self.ui.cutoff_label,
        self.ui.max_elements_label,
        self.ui.fit_type_label,
        self.ui.analyze_kkr_label,
    ]:
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    for w in [self.ui.cutoff_entry, self.ui.max_elements_entry, self.ui.fit_type_choice]:
        w.setMinimumHeight(30)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.run_kkr_button.setFixedHeight(30)
    self.ui.run_kkr_button.setFixedWidth(110)
    self.ui.run_kkr_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    keep = {
        self.ui.cutoff_label,
        self.ui.max_elements_label,
        self.ui.fit_type_label,
        self.ui.analyze_kkr_label,
    }
    _hide_unmanaged_children(gb, keep)

    grid.addWidget(self.ui.cutoff_label, 0, 0)
    grid.addWidget(self.ui.cutoff_entry, 0, 1)

    grid.addWidget(self.ui.max_elements_label, 1, 0)
    grid.addWidget(self.ui.max_elements_entry, 1, 1)

    grid.addWidget(self.ui.fit_type_label, 2, 0)
    grid.addWidget(self.ui.fit_type_choice, 2, 1)

    grid.addWidget(self.ui.analyze_kkr_label, 3, 0)

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addStretch(1)
    btn_row.addWidget(self.ui.run_kkr_button)
    grid.addWidget(self.ui.analyze_kkr_label, 3, 0)
    grid.addLayout(btn_row, 3, 1)


def _layout_run_group(self) -> None:
    gb = self.ui.run_layout
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 0)

    rows = [
        (self.ui.simple_run_label, self.ui.simple_run_button),
        (self.ui.bayes_label, self.ui.bayesian_button),
        (self.ui.HT_label, self.ui.HT_button),
    ]
    keep = {r[0] for r in rows}
    _hide_unmanaged_children(gb, keep)

    for r, (lab, btn) in enumerate(rows):
        btn.setMinimumHeight(30)
        btn.setMinimumWidth(110)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        grid.addWidget(lab, r, 0)
        grid.addWidget(btn, r, 1, alignment=QtCore.Qt.AlignRight)

    # Fit buttons row (below Hilbert Transform). Keep alignment with Select buttons.
    if hasattr(self.ui, "fit_one_button") and hasattr(self.ui, "fit_all_button"):
        self.ui.fit_one_button.setMinimumHeight(30)
        self.ui.fit_one_button.setMinimumWidth(110)
        self.ui.fit_one_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.ui.fit_all_button.setMinimumHeight(30)
        self.ui.fit_all_button.setMinimumWidth(110)
        self.ui.fit_all_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        # row index = len(rows)
        grid.addWidget(self.ui.fit_one_button, len(rows), 0, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(self.ui.fit_all_button, len(rows), 1, alignment=QtCore.Qt.AlignRight)


def _layout_peak_group(self) -> None:
    gb = self.ui.Peak_analysis_frame
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)

    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    for lab in [self.ui.peak_method_label, self.ui.reg_param_2, self.ui.Peak_decon_run]:
        lab.setMinimumWidth(150)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        lab.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

    self.ui.peak_method_choice.setMinimumHeight(30)
    self.ui.peak_method_choice.setMinimumWidth(110)
    self.ui.peak_method_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.peak_num_entry.setMinimumHeight(30)
    self.ui.peak_num_entry.setMinimumWidth(110)
    self.ui.peak_num_entry.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.peak_decon_button.setFixedHeight(30)
    self.ui.peak_decon_button.setFixedWidth(110)
    self.ui.peak_decon_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    keep = {self.ui.peak_method_label, self.ui.reg_param_2, self.ui.Peak_decon_run}
    _hide_unmanaged_children(gb, keep)

    grid.addWidget(self.ui.peak_method_label, 0, 0)
    grid.addWidget(self.ui.peak_method_choice, 0, 1)

    grid.addWidget(self.ui.reg_param_2, 1, 0)
    grid.addWidget(self.ui.peak_num_entry, 1, 1)

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addStretch(1)
    btn_row.addWidget(self.ui.peak_decon_button)

    grid.addWidget(self.ui.Peak_decon_run, 2, 0)
    grid.addLayout(btn_row, 2, 1)


def _layout_export_group(self) -> None:
    gb = self.ui.export_frame
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(8)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 0)

    # 动态创建 Open / Save 两个项目按钮
    if not hasattr(self.ui, "open_project_label"):
        self.ui.open_project_label = QtWidgets.QLabel("Open Project", gb)
    if not hasattr(self.ui, "save_project_label"):
        self.ui.save_project_label = QtWidgets.QLabel("Save Project", gb)

    if not hasattr(self.ui, "open_project_button"):
        self.ui.open_project_button = QtWidgets.QPushButton("Open", gb)
    if not hasattr(self.ui, "save_project_button"):
        self.ui.save_project_button = QtWidgets.QPushButton("Save", gb)

    # 隐藏之前残留的 Project / Save As
    for name in [
        "project_title_label",
        "save_as_project_label",
        "save_as_project_button",
    ]:
        w = getattr(self.ui, name, None)
        if w is not None:
            try:
                w.hide()
            except Exception:
                pass

    # 只连接一次
    if not getattr(self, "_project_buttons_connected", False):
        self.ui.open_project_button.clicked.connect(self.open_project_callback)
        self.ui.save_project_button.clicked.connect(self.save_project_callback)
        self._project_buttons_connected = True

    rows = [
        (self.ui.export_DRT_label, self.ui.export_DRT_button),
        (self.ui.export_EIS_label, self.ui.export_EIS_button),
        (self.ui.export_fig_label, self.ui.export_fig_button),
        (self.ui.open_project_label, self.ui.open_project_button),
        (self.ui.save_project_label, self.ui.save_project_button),
    ]

    keep = {lab for lab, btn in rows}
    _hide_unmanaged_children(gb, keep)

    export_btn_qss = """
    QPushButton {
        background: #FFFFFF;
        border: 1px solid #D2D2D7;
        border-radius: 10px;
        padding: 0px 12px;
        min-height: 0px;
        max-height: 28px;
        height: 28px;
    }
    QPushButton:hover {
        background: #F2F2F7;
        border-color: #C7C7CC;
    }
    QPushButton:pressed {
        background: #EAEAEE;
        border-color: #BDBDC2;
    }
    """

    button_w = 110
    button_h = 28

    for row, (lab, btn) in enumerate(rows):
        lab.show()
        btn.show()

        lab.setStyleSheet("background: transparent;")
        lab.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lab.setFixedHeight(button_h)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        btn.setFixedSize(button_w, button_h)
        btn.setMinimumSize(button_w, button_h)
        btn.setMaximumSize(button_w, button_h)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        btn.setStyleSheet(export_btn_qss)

        grid.setRowMinimumHeight(row, button_h)
        grid.setRowStretch(row, 0)

        grid.addWidget(lab, row, 0)
        grid.addWidget(btn, row, 1, alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)


def _replace_show_buttons_with_tabs(self) -> None:
    bar = getattr(self.ui, "show_layout", None)
    if bar is None:
        return

    btn_map = [
        ("EIS Data", getattr(self.ui, "show_EIS", None)),
        ("K-K Residual", getattr(self.ui, "show_KK_res", None)),
        ("DRT Residual", getattr(self.ui, "show_re_res", None)),
        ("DRT", getattr(self.ui, "show_DRT", None)),
        ("DRT comparison", getattr(self.ui, "show_DRT_comp", None)),
        ("DRT Map", getattr(self.ui, "show_DRT_map", None)),
        ("Magnitude", getattr(self.ui, "show_mag", None)),
        ("Phase", getattr(self.ui, "show_phase", None)),
        ("Re Part", getattr(self.ui, "show_re", None)),
        ("Im Part", getattr(self.ui, "show_im", None)),
        ("EIS Score", getattr(self.ui, "show_score", None))
    ]

    hide_buttons = [
        getattr(self.ui, "show_EIS", None),
        getattr(self.ui, "show_KK_res", None),
        getattr(self.ui, "show_mag", None),
        getattr(self.ui, "show_phase", None),
        getattr(self.ui, "show_re", None),
        getattr(self.ui, "show_im", None),
        getattr(self.ui, "show_re_res", None),
        getattr(self.ui, "show_im_res", None),
        getattr(self.ui, "show_score", None),
        getattr(self.ui, "show_DRT", None),
        getattr(self.ui, "show_DRT_comp", None),
        getattr(self.ui, "show_DRT_map", None),
    ]

    for b in hide_buttons:
        if b is not None:
            b.hide()

    _clear_layout_widget(bar)

    lay = QtWidgets.QVBoxLayout(bar)
    lay.setContentsMargins(10, 4, 10, 0)
    lay.setSpacing(4)

    # ---------- top thin scrollbar ----------
    top_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal, bar)
    top_scrollbar.setObjectName("TopTabScrollBar")

    # Make the top scrollbar as thin as the sidebar scrollbar.
    top_scrollbar.setFixedHeight(2)

    top_scrollbar.setStyleSheet("""
        QScrollBar#TopTabScrollBar:horizontal {
            background: transparent;
            height: 2px;
            margin: 0px 6px 0px 6px;
            border: 0px;
        }

        QScrollBar#TopTabScrollBar::handle:horizontal {
            background: rgba(60, 60, 67, 0.35);
            border-radius: 3px;
            min-width: 60px;
        }

        QScrollBar#TopTabScrollBar::handle:horizontal:hover {
            background: rgba(60, 60, 67, 0.50);
        }

        QScrollBar#TopTabScrollBar::add-line:horizontal,
        QScrollBar#TopTabScrollBar::sub-line:horizontal {
            width: 0px;
            height: 0px;
            border: 0px;
            background: transparent;
        }

        QScrollBar#TopTabScrollBar::add-page:horizontal,
        QScrollBar#TopTabScrollBar::sub-page:horizontal {
            background: transparent;
            border: 0px;
        }
    """)

    # ---------- scroll area for tab buttons ----------
    scroll = QtWidgets.QScrollArea(bar)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    scroll.setFixedHeight(46)
    scroll.setStyleSheet("""
        QScrollArea {
            border: 0px;
            background: transparent;
        }
    """)

    tab_host = QtWidgets.QWidget(scroll)
    tab_host.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)

    tab_layout = QtWidgets.QHBoxLayout(tab_host)
    tab_layout.setContentsMargins(0, 0, 0, 0)
    tab_layout.setSpacing(8)

    self._top_tab_buttons = []

    def _make_tab_button(title, src_button):
        btn = QtWidgets.QPushButton(title, tab_host)
        btn.setCheckable(True)
        btn.setMinimumHeight(36)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        btn.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                border: 0px solid #D2D2D7;
                border-radius: 10px;
                padding: 6px 14px;
                color: #1D1D1F;
            }
            QPushButton:hover {
                background: #F2F2F7;
            }
            QPushButton:checked {
                background: #E9E9EE;
            }
        """)

        def _clicked():
            for b in getattr(self, "_top_tab_buttons", []):
                b.setChecked(False)
            btn.setChecked(True)
            if src_button is not None:
                src_button.click()

        btn.clicked.connect(_clicked)
        return btn

    for title, src_button in btn_map:
        btn = _make_tab_button(title, src_button)
        self._top_tab_buttons.append(btn)
        tab_layout.addWidget(btn)

    tab_layout.addStretch(1)
    scroll.setWidget(tab_host)

    # Put scrollbar ABOVE the button row.
    lay.addWidget(scroll)
    lay.addWidget(top_scrollbar)

    inner_bar = scroll.horizontalScrollBar()

    def _sync_scrollbar_range():
        try:
            top_scrollbar.blockSignals(True)
            top_scrollbar.setRange(inner_bar.minimum(), inner_bar.maximum())
            top_scrollbar.setPageStep(max(1, inner_bar.pageStep() // 10))
            top_scrollbar.setSingleStep(30)
            top_scrollbar.setValue(inner_bar.value())
            top_scrollbar.setVisible(inner_bar.maximum() > 0)
        finally:
            top_scrollbar.blockSignals(False)

    def _top_to_inner(value):
        inner_bar.setValue(value)

    def _inner_to_top(value):
        top_scrollbar.setValue(value)

    top_scrollbar.valueChanged.connect(_top_to_inner)
    inner_bar.valueChanged.connect(_inner_to_top)
    inner_bar.rangeChanged.connect(lambda *_: _sync_scrollbar_range())

    # ---------- wheel event: vertical wheel controls horizontal scrolling ----------
    # ---------- wheel event: mouse wheel switches the selected top tab ----------
    class TopTabWheelFilter(QtCore.QObject):
        def __init__(self, gui_window, scroll_area, target_scrollbar):
            super().__init__(gui_window)
            self._gui = gui_window
            self._scroll = scroll_area
            self._bar = target_scrollbar

        def _current_index(self):
            buttons = getattr(self._gui, "_top_tab_buttons", []) or []
            for i, btn in enumerate(buttons):
                try:
                    if btn.isChecked():
                        return i
                except Exception:
                    pass
            return 0

        def _activate_index(self, index):
            buttons = getattr(self._gui, "_top_tab_buttons", []) or []
            if not buttons:
                return

            index = max(0, min(int(index), len(buttons) - 1))
            btn = buttons[index]

            # Trigger the original button logic:
            # this checks the button and calls the hidden original show_xxx button.
            try:
                btn.click()
            except Exception:
                return

            # Make the selected tab visible inside the horizontal scroll area.
            try:
                x = btn.x()
                w = btn.width()
                view_w = self._scroll.viewport().width()
                left = self._bar.value()
                right = left + view_w

                if x < left:
                    self._bar.setValue(max(self._bar.minimum(), x - 8))
                elif x + w > right:
                    self._bar.setValue(min(self._bar.maximum(), x + w - view_w + 8))
            except Exception:
                pass

        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Wheel:
                try:
                    delta = event.angleDelta().y()
                    if delta == 0:
                        delta = event.angleDelta().x()

                    if delta == 0:
                        return False

                    cur = self._current_index()

                    # Wheel down -> next tab; wheel up -> previous tab.
                    if delta < 0:
                        self._activate_index(cur + 1)
                    else:
                        self._activate_index(cur - 1)

                    event.accept()
                    return True
                except Exception:
                    pass

            return super().eventFilter(obj, event)

    self._top_tab_wheel_filter = TopTabWheelFilter(self, scroll, inner_bar)

    try:
        scroll.viewport().installEventFilter(self._top_tab_wheel_filter)
        scroll.installEventFilter(self._top_tab_wheel_filter)
        tab_host.installEventFilter(self._top_tab_wheel_filter)
        top_scrollbar.installEventFilter(self._top_tab_wheel_filter)

        for btn in self._top_tab_buttons:
            btn.installEventFilter(self._top_tab_wheel_filter)
    except Exception:
        pass

    try:
        bar.setMinimumHeight(72)
        bar.setMaximumHeight(88)
        bar.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass

    # Make sure range is correct after Qt finishes layout.
    QtCore.QTimer.singleShot(0, _sync_scrollbar_range)
    QtCore.QTimer.singleShot(100, _sync_scrollbar_range)

    try:
        if self._top_tab_buttons:
            self._top_tab_buttons[0].setChecked(True)
        if btn_map[0][1] is not None:
            btn_map[0][1].click()
    except Exception:
        pass


try:
    GUI._refactor_ui_layout = _refactor_ui_layout
except Exception:
    pass


def launch_gui():
    # High DPI attributes must be set before QApplication is created.
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QtWidgets.QApplication([])
    try:
        apply_flat_theme(app)
    except Exception:
        pass

    MainWindow = GUI()
    # Layout refactor is cosmetic; keep logic intact.
    try:
        if hasattr(MainWindow, "_refactor_ui_layout"):
            MainWindow._refactor_ui_layout()
    except Exception:
        pass

    MainWindow.show()
    app.exec_()