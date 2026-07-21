# -*- coding: utf-8 -*-
"""Peak Analysis report export workflow."""

from __future__ import annotations

import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

from ..services.peak_export import write_peak_csv, write_peak_workbook


class PeakExportControllerMixin:
    """Export current peak results or a multi-sheet project workbook."""

    def export_peak_results(self):
        analyzed_keys = [
            key for key in self._get_file_keys_in_ui_order()
            if key in self.data_store
            and list(getattr(self.data_store[key], 'peak_results', []) or [])
        ]
        if not analyzed_keys:
            QtWidgets.QMessageBox.information(
                self, 'Export Peak', 'No Peak Analysis results are available.'
            )
            return

        current_key = getattr(self, 'current_file_key', None)
        current_available = current_key in analyzed_keys
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Export Peak Results')
        layout = QtWidgets.QVBoxLayout(dialog)
        selected_radio = QtWidgets.QRadioButton('Export selected file (CSV)', dialog)
        all_radio = QtWidgets.QRadioButton('Export all analyzed files (Excel workbook)', dialog)
        selected_radio.setEnabled(current_available)
        (selected_radio if current_available else all_radio).setChecked(True)
        layout.addWidget(selected_radio)
        layout.addWidget(all_radio)
        note = QtWidgets.QLabel(
            'All files are saved in one .xlsx workbook, with one worksheet per file.',
            dialog,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            if selected_radio.isChecked():
                self._export_selected_peak_csv(current_key)
            else:
                self._export_all_peak_workbook(analyzed_keys)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Export Peak Failed', str(exc))

    def _export_selected_peak_csv(self, key):
        display_name = self._display_name_for_key(key)
        default_name = f'{os.path.splitext(display_name)[0]}_peaks.csv'
        default_path = os.path.join(os.path.dirname(key), default_name)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Peak Results', default_path, 'CSV files (*.csv)'
        )
        if not path:
            return
        final_path = write_peak_csv(path, display_name, self.data_store[key])
        self.statusBar().showMessage(f'Peak results exported: {final_path}', 3000)

    def _export_all_peak_workbook(self, keys):
        default_path = 'Peak_results.xlsx'
        project_path = getattr(self, 'current_project_path', None)
        if project_path:
            base = os.path.splitext(os.path.basename(project_path))[0]
            default_path = os.path.join(os.path.dirname(project_path), f'{base}_peaks.xlsx')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export All Peak Results', default_path, 'Excel workbook (*.xlsx)'
        )
        if not path:
            return
        datasets = [
            (self._display_name_for_key(key), self.data_store[key])
            for key in keys
        ]
        final_path = write_peak_workbook(path, datasets)
        self.statusBar().showMessage(f'Peak workbook exported: {final_path}', 3000)
