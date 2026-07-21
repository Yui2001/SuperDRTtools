# -*- coding: utf-8 -*-
"""Main-window composition root for SuperDRTtools."""

from PyQt5 import QtWidgets

from ..ui import layout
from ..controllers import (
    DRTComparisonControllerMixin,
    DRTExportControllerMixin,
    DRTMapControllerMixin,
    EISExportControllerMixin,
    FigureExportControllerMixin,
    FileListControllerMixin,
    FittingWorkflowMixin,
    ImportControllerMixin,
    KramersKronigControllerMixin,
    MaskControllerMixin,
    PlotControllerMixin,
    ProjectControllerMixin,
    PeakAnalysisControllerMixin,
    PeakExportControllerMixin,
)
from .window_setup import connect_window_signals, initialize_window_state


class SuperDRTMainWindow(
    ProjectControllerMixin,
    PeakAnalysisControllerMixin,
    ImportControllerMixin,
    MaskControllerMixin,
    FileListControllerMixin,
    KramersKronigControllerMixin,
    FittingWorkflowMixin,
    PlotControllerMixin,
    DRTComparisonControllerMixin,
    DRTMapControllerMixin,
    DRTExportControllerMixin,
    EISExportControllerMixin,
    PeakExportControllerMixin,
    FigureExportControllerMixin,
    QtWidgets.QMainWindow,
):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        initialize_window_state(self)
        connect_window_signals(self)
