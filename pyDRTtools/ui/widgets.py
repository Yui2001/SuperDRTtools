# -*- coding: utf-8 -*-
"""Reusable file-list delegate and external file-drop event filter."""

import os

from PyQt5 import QtCore, QtGui, QtWidgets


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
    """Accept dropped EIS/project files without breaking list reordering."""

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
