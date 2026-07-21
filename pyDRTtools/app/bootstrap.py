# -*- coding: utf-8 -*-
"""Qt application bootstrap isolated from window business logic."""

import multiprocessing as mp

from PyQt5 import QtCore, QtWidgets

from .window_manager import open_new_window
from ..ui.theme import apply_flat_theme


def launch_gui():
    mp.freeze_support()
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

    open_new_window()
    app.exec_()
