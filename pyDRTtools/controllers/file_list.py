# -*- coding: utf-8 -*-
"""Controller for the file list feature."""

from PyQt5 import QtCore, QtWidgets


class FileListControllerMixin:
    """Own the file list GUI workflow."""

    def _set_current_file(self, key: str):
        if key not in self.data_store:
            return
        self.current_file_key = key
        self.data = self.data_store[key]
        self._eis_selected_raw_indices = []
        # keep current plot selection across switching
        self.plotting_callback(self.current_plot_option)
        # Showing/hiding the results card changes the sidebar geometry. Do it
        # after the list selection event and canvas replacement have completed.
        QtCore.QTimer.singleShot(0, self._refresh_peak_results_panel)

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
                QtCore.QTimer.singleShot(0, self._refresh_peak_results_panel)
            self._suppress_file_select = False
            return

        if getattr(self, 'current_plot_option', None) == 'DRT_comparison':
            self.current_file_key = key
            self.data = self.data_store.get(key)
            self._eis_selected_raw_indices = []
            self._select_drt_comparison_line_by_key(key)
            return

        if getattr(self, 'current_plot_option', None) == 'Peak_comparison':
            self.current_file_key = key
            self.data = self.data_store.get(key)
            self._eis_selected_raw_indices = []
            QtCore.QTimer.singleShot(0, self._refresh_peak_results_panel)
            return

        self._set_current_file(key)

    def _on_files_reordered(self, *args, **kwargs):
        self._mark_project_dirty()
        # Order affects DRT comparison colors AND DRT map row order.
        if getattr(self, 'current_plot_option', None) in ('DRT_comparison', 'DRT_map', 'Peak_comparison'):
            self.plotting_callback(self.current_plot_option)
        return

    def _show_files_context_menu(self, pos: QtCore.QPoint):
        if not hasattr(self.ui, "files_list"):
            return

        menu = QtWidgets.QMenu(self)
        item = self.ui.files_list.itemAt(pos)

        act_rename = QtWidgets.QAction("Rename", self)
        act_clear_current = QtWidgets.QAction("Clear", self)
        act_clear_all = QtWidgets.QAction("Clear all files", self)

        if item is None:
            act_rename.setEnabled(False)
            act_clear_current.setEnabled(False)

        menu.addAction(act_rename)
        menu.addAction(act_clear_current)
        menu.addSeparator()
        menu.addAction(act_clear_all)

        menu.setStyleSheet("""
            QMenu { background:#FFFFFF;color:#1D1D1F;border:0px solid #E5E5EA;border-radius:10px; }
            QMenu::item { color:#1D1D1F;padding:6px 10px;min-height:28px; }
        """)

        def _rename_current():
            if item is None:
                return
            key = item.data(QtCore.Qt.UserRole)
            if not key:
                return

            current_name = self._display_name_for_key(key)
            new_name, accepted = QtWidgets.QInputDialog.getText(
                self,
                "Rename file",
                "File name:",
                QtWidgets.QLineEdit.Normal,
                current_name,
            )
            if not accepted:
                return

            new_name = str(new_name).strip()
            if not new_name:
                QtWidgets.QMessageBox.warning(self, "Rename file", "The file name cannot be empty.")
                return

            # Rename only the displayed project name. Keep the original full-path
            # key unchanged so reloading, fitting and project save remain stable.
            meta = dict(getattr(self, 'file_meta', {}).get(key) or {})
            meta['display_name'] = new_name
            meta.setdefault('fitted', False)
            meta.setdefault('signature', None)
            self.file_meta[key] = meta
            item.setText(new_name)
            item.setToolTip(key)
            self._mark_project_dirty()

            if getattr(self, 'current_plot_option', None) in ('DRT_comparison', 'DRT_map', 'Peak_comparison'):
                try:
                    self.plotting_callback(self.current_plot_option)
                except Exception:
                    pass
            try:
                self.statusBar().showMessage(f'Renamed: {new_name}', 2000)
            except Exception:
                pass

        def _clear_current():
            if item is None:
                return
            key = item.data(QtCore.Qt.UserRole)
            self._remove_file_by_key(key)

        def _clear_all():
            self._remove_all_files()

        act_rename.triggered.connect(_rename_current)
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
        self._update_peak_comparison_availability()

        if self.current_file_key == key:
            self.current_file_key = None
            self.data = None

            if hasattr(self.ui, "files_list") and self.ui.files_list.count() > 0:
                self.ui.files_list.setCurrentRow(0)  # triggers selection callback
            else:
                self._refresh_peak_results_panel()
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
        self._refresh_peak_results_panel()
        self._update_peak_comparison_availability()
        self.plotting_callback(self.current_plot_option)  # clears
