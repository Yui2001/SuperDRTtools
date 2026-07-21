# -*- coding: utf-8 -*-
"""Controller for the importing feature."""

import copy
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from ..algorithms.runs import EIS_object
from ..ui.widgets import FileListDelegate


class ImportControllerMixin:
    """Own the importing GUI workflow."""

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
            self.file_meta[path] = {
                'fitted': False,
                'signature': None,
                'display_name': os.path.basename(path),
            }

            if hasattr(self.ui, "files_list"):
                item = QtWidgets.QListWidgetItem(self._display_name_for_key(path))
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
        """Import dropped EIS files, or replace the workspace with a project."""
        # keep local files; parser handles csv/txt/extensionless EIS text files.
        paths = [p for p in (paths or []) if p and os.path.isfile(p)]
        if not paths:
            return False

        project_paths = [p for p in paths if p.lower().endswith('.sdrtp')]
        if project_paths:
            if len(project_paths) > 1 or len(paths) > 1:
                QtWidgets.QMessageBox.information(
                    self,
                    'Open Project',
                    'A project replaces the current workspace. Only the first dropped project will be opened.',
                )
            return self.open_dropped_project(project_paths[0])

        newly_added = self._import_paths_into_store(paths)

        if not newly_added:
            return False

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
        return True

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Accept dropping EIS data or a project onto the main window."""
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
        """Append dropped EIS data or replace the workspace with a project."""
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
