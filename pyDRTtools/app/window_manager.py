# -*- coding: utf-8 -*-
"""Create and retain independent SuperDRTtools application windows."""

from PyQt5 import QtCore, QtWidgets

from .main_window import SuperDRTMainWindow
from ..ui.runtime_layout import refactor_ui_layout


_OPEN_WINDOWS = {}


def _forget_window(key):
    _OPEN_WINDOWS.pop(key, None)


def open_new_window(project_path=None):
    """Open a blank window, optionally loading a project into that window."""
    if QtWidgets.QApplication.instance() is None:
        raise RuntimeError('A QApplication must exist before opening a window.')

    window = SuperDRTMainWindow()
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    try:
        refactor_ui_layout(window)
    except Exception:
        pass

    key = id(window)
    _OPEN_WINDOWS[key] = window
    window.destroyed.connect(lambda _obj=None, window_key=key: _forget_window(window_key))
    window.show()

    if project_path:
        window.open_project_from_path(project_path, confirm_replacement=False)
    return window


def open_window_count():
    """Return the number of live managed windows (primarily for diagnostics)."""
    return len(_OPEN_WINDOWS)
