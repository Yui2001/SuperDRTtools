# -*- coding: utf-8 -*-
"""Controller for the figure export feature."""

from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog


class FigureExportControllerMixin:
    """Own the figure export GUI workflow."""

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
