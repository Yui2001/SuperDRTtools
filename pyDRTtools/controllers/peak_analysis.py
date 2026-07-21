# -*- coding: utf-8 -*-
"""Peak One/All execution, result presentation, and peak naming."""

from __future__ import annotations

import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..ui.canvas import Figure_Canvas


class PeakAnalysisControllerMixin:
    """Own asynchronous peak deconvolution and its user-facing results."""

    def _build_peak_params(self):
        return {
            'native_threads': 1,
            'rbf_type': str(self.ui.discre_choice.currentText()),
            'data_used': str(self.ui.data_used_choice.currentText()),
            'induct_used': int(self.ui.induct_choice.currentIndex()),
            'der_used': str(self.ui.der_choice.currentText()),
            'cv_type': str(self.ui.lambda_choice.currentText()),
            'reg_param': float(self.ui.reg_param_entry.text()),
            'shape_control': str(self.ui.shape_control_choice.currentText()),
            'coeff': float(self.ui.FWHM_entry.text()),
            'peak_method': str(self.ui.peak_method_choice.currentText()),
            'peak_count': str(self.ui.peak_count_choice.currentText()),
            'max_auto_peaks': 12,
        }

    @staticmethod
    def _peak_signature(params):
        return tuple((key, params[key]) for key in sorted(params) if key != 'native_threads')

    def peak_analysis_run_callback(self):
        """Backward-compatible alias for Peak One."""
        self.peak_selected_callback()

    def peak_selected_callback(self):
        key = getattr(self, 'current_file_key', None)
        if key is None or key not in self.data_store:
            QtWidgets.QMessageBox.information(self, 'Peak Analysis', 'Select an EIS file first.')
            return
        self._start_peak_tasks([key], skip_completed=False)

    def peak_all_callback(self):
        keys = self._get_file_keys_in_ui_order()
        if not keys:
            QtWidgets.QMessageBox.information(self, 'Peak Analysis', 'Import EIS data first.')
            return
        self._start_peak_tasks(keys, skip_completed=True)

    def _start_peak_tasks(self, keys, skip_completed):
        if getattr(self, '_peak_active', False):
            self.statusBar().showMessage('Peak Analysis is already running.', 2000)
            return
        if getattr(self, '_fit_all_active', False):
            QtWidgets.QMessageBox.information(
                self, 'Peak Analysis', 'Wait for Fit All to finish before running Peak Analysis.'
            )
            return

        try:
            params = self._build_peak_params()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Peak Analysis', str(exc))
            return
        signature = self._peak_signature(params)

        tasks = []
        skipped = 0
        for key in keys:
            if key not in self.data_store:
                continue
            meta = dict(self.file_meta.get(key) or {})
            if skip_completed and meta.get('peak_analyzed') and meta.get('peak_signature') == signature:
                skipped += 1
                continue
            tasks.append((key, self.data_store[key]))

        if not tasks:
            self.statusBar().showMessage(f'Peak Analysis complete: 0 analyzed, {skipped} skipped', 2500)
            return

        self._peak_active = True
        self._peak_total = len(tasks)
        self._peak_pending = len(tasks)
        self._peak_done = 0
        self._peak_errors = 0
        self._peak_skipped = skipped
        self._peak_process_note = ''
        self._set_peak_controls_enabled(False)

        try:
            info = self._peak_process_manager.start(tasks, params, signature)
            self._peak_worker_count = int(info.get('worker_count', 1) or 1)
            planned = int(info.get('initial_worker_count', self._peak_worker_count) or 1)
            if info.get('memory_limited_initial'):
                available = info.get('initial_available_commit_gib')
                suffix = f', {float(available):.1f} GiB commit available' if available is not None else ''
                self._peak_process_note = f'memory-safe start {planned}→{self._peak_worker_count}{suffix}'
            self._peak_poll_timer.start()
            self._update_peak_progress_status()
        except Exception as exc:
            self._peak_active = False
            self._peak_pending = 0
            self._set_peak_controls_enabled(True)
            self._shutdown_peak_process_pool(cancel=True, terminate=True)
            QtWidgets.QMessageBox.warning(self, 'Peak Analysis failed to start', str(exc))

    def _set_peak_controls_enabled(self, enabled):
        for name in (
            'peak_decon_button', 'peak_all_button', 'peak_method_choice', 'peak_count_choice',
            'fit_one_button', 'fit_all_button',
        ):
            widget = getattr(self.ui, name, None)
            if widget is not None:
                widget.setEnabled(bool(enabled))

    def _update_peak_progress_status(self):
        total = int(getattr(self, '_peak_total', 0) or 0)
        done = int(getattr(self, '_peak_done', 0) or 0)
        errors = int(getattr(self, '_peak_errors', 0) or 0)
        completed = min(total, done + errors)
        remaining = max(0, total - completed)
        percent = int(round(100.0 * completed / total)) if total else 100
        note = str(getattr(self, '_peak_process_note', '') or '')
        suffix = f' | {note}' if note else ''
        self.statusBar().showMessage(
            f'Peak Analysis: {completed}/{total} completed ({percent}%) | '
            f'{remaining} remaining | {int(getattr(self, "_peak_worker_count", 0) or 0)} processes | '
            f'{errors} errors | {int(getattr(self, "_peak_skipped", 0) or 0)} skipped{suffix}'
        )

    def _poll_peak_processes(self):
        if not getattr(self, '_peak_active', False):
            self._peak_poll_timer.stop()
            return
        try:
            batch = self._peak_process_manager.poll()
        except Exception as exc:
            self._finish_peak_batch_with_error(str(exc))
            return

        retry = batch.get('retry')
        scale = batch.get('scale')
        if retry:
            self._peak_worker_count = int(retry.get('to_workers', 1) or 1)
            self._peak_process_note = (
                f'resource fallback {retry.get("from_workers", "?")}→{retry.get("to_workers", "?")}'
            )
        elif scale:
            self._peak_worker_count = int(scale.get('to_workers', self._peak_worker_count) or 1)
            cpu = scale.get('cpu_percent')
            cpu_text = f', CPU {float(cpu):.0f}%' if cpu is not None else ''
            self._peak_process_note = (
                f'auto scale {scale.get("from_workers", "?")}→{scale.get("to_workers", "?")}{cpu_text}'
            )
        else:
            self._peak_worker_count = int(batch.get('worker_count', self._peak_worker_count) or 1)

        # Normalize replacement characters left by earlier Windows code-page
        # conversions in status-only text.
        self._peak_process_note = str(self._peak_process_note).replace(chr(0xfffd) * 2, '->')

        for key, entry, signature in batch.get('results', []):
            self._store_peak_result(key, entry, signature)
        for key, message in batch.get('errors', []):
            self._peak_errors += 1
            self._peak_pending = max(0, self._peak_pending - 1)
            self.statusBar().showMessage(f'Peak Analysis failed: {self._display_name_for_key(key)} — {message}')

        self._update_peak_progress_status()
        if batch.get('done') and self._peak_pending <= 0:
            self._finish_peak_batch()

    def _store_peak_result(self, key, entry, signature):
        self.data_store[key] = entry
        if key == self.current_file_key:
            self.data = entry
        meta = dict(self.file_meta.get(key) or {})
        meta.update({
            'fitted': True,
            'peak_analyzed': True,
            'peak_signature': signature,
        })
        meta.setdefault('display_name', os.path.basename(key))
        self.file_meta[key] = meta
        self._update_file_status(key, fitted=True)
        self._peak_done += 1
        self._peak_pending = max(0, self._peak_pending - 1)
        self._mark_project_dirty()
        if key == self.current_file_key:
            self._refresh_peak_results_panel()

    def _finish_peak_batch(self):
        self._peak_poll_timer.stop()
        self._peak_active = False
        self._set_peak_controls_enabled(True)
        self._shutdown_peak_process_pool(cancel=False, terminate=False)
        self._update_peak_comparison_availability()
        if self.current_file_key in self.data_store:
            self._refresh_peak_results_panel()
            self.plotting_callback('DRT_data')
        self.statusBar().showMessage(
            f'Peak Analysis complete: {self._peak_done}/{self._peak_total} analyzed, '
            f'{self._peak_errors} errors, {self._peak_skipped} skipped',
            4000,
        )

    def _finish_peak_batch_with_error(self, message):
        remaining = int(getattr(self, '_peak_pending', 0) or 0)
        self._peak_errors += remaining
        self._peak_pending = 0
        self._shutdown_peak_process_pool(cancel=True, terminate=True)
        self._peak_active = False
        self._set_peak_controls_enabled(True)
        QtWidgets.QMessageBox.warning(self, 'Peak Analysis failed', message)

    def _shutdown_peak_process_pool(self, cancel=False, terminate=False):
        try:
            self._peak_poll_timer.stop()
        except Exception:
            pass
        try:
            self._peak_process_manager.shutdown(cancel=cancel, terminate=terminate)
        except Exception:
            pass
        self._peak_worker_count = 0

    def _refresh_peak_results_panel(self):
        table = getattr(self.ui, 'peak_results_table', None)
        frame = getattr(self.ui, 'peak_results_frame', None)
        if table is None or frame is None:
            return
        results = list(getattr(self.data, 'peak_results', []) or []) if self.data is not None else []
        self._updating_peak_table = True
        table.blockSignals(True)
        try:
            table.setRowCount(len(results))
            for row, result in enumerate(results):
                values = (
                    str(result.get('name', f'Peak {row + 1}')),
                    f'{float(result.get("tau_s", np.nan)):.6g}',
                    f'{float(result.get("frequency_hz", np.nan)):.6g}',
                    f'{float(result.get("resistance_ohm", np.nan)):.6g}',
                    f'{float(result.get("fraction_percent", np.nan)):.3f}',
                    f'{float(result.get("fwhm_decades", np.nan)):.4g}',
                )
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if column == 0:
                        item.setData(QtCore.Qt.UserRole, values[0])
                    else:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                        item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                    table.setItem(row, column, item)
        finally:
            table.blockSignals(False)
            self._updating_peak_table = False
        # Results are presented as annotations beside the fitted peaks. Keep
        # this compatibility table populated for project/test compatibility,
        # but never occupy space in the settings sidebar.
        frame.hide()
        table.setToolTip('Double-click a peak name to rename it.')
        frame.updateGeometry()

    def _peak_name_scope(self, old_name, new_name):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle('Rename Peak')
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(f'Rename “{old_name}” to “{new_name}”?')
        box.setText(f'Rename "{old_name}" to "{new_name}"?')
        one = box.addButton('This Peak', QtWidgets.QMessageBox.AcceptRole)
        all_matching = box.addButton('All Matching Peaks', QtWidgets.QMessageBox.ActionRole)
        cancel = box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(one)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is one:
            return 'one'
        if clicked is all_matching:
            return 'all'
        if clicked is cancel:
            return 'cancel'
        return 'cancel'

    def _on_peak_result_item_changed(self, item):
        if self._updating_peak_table or item is None or item.column() != 0 or self.data is None:
            return
        results = list(getattr(self.data, 'peak_results', []) or [])
        row = item.row()
        if row < 0 or row >= len(results):
            return
        old_name = str(item.data(QtCore.Qt.UserRole) or results[row].get('name', '')).strip()
        new_name = str(item.text() or '').strip()
        if not new_name or new_name == old_name:
            table = self.ui.peak_results_table
            table.blockSignals(True)
            item.setText(old_name)
            table.blockSignals(False)
            return
        if any(index != row and str(result.get('name', '')) == new_name for index, result in enumerate(results)):
            QtWidgets.QMessageBox.warning(self, 'Rename Peak', 'Peak names must be unique within a file.')
            table = self.ui.peak_results_table
            table.blockSignals(True)
            item.setText(old_name)
            table.blockSignals(False)
            return

        scope = self._peak_name_scope(old_name, new_name)
        if scope == 'cancel':
            table = self.ui.peak_results_table
            table.blockSignals(True)
            item.setText(old_name)
            table.blockSignals(False)
            return
        if scope == 'all':
            for entry in self.data_store.values():
                for result in list(getattr(entry, 'peak_results', []) or []):
                    if str(result.get('name', '')) == old_name:
                        result['name'] = new_name
        else:
            results[row]['name'] = new_name
            self.data.peak_results = results

        self._mark_project_dirty()
        self._update_peak_comparison_availability()

        def _finish_item_rename():
            table = self.ui.peak_results_table
            table.blockSignals(True)
            try:
                item.setData(QtCore.Qt.UserRole, new_name)
            finally:
                table.blockSignals(False)
            self._rename_peak_in_current_plot(old_name, new_name)

        # Changing another item role while handling ``itemChanged`` recursively
        # re-enters Qt's model signal. Finish the bookkeeping after it returns.
        QtCore.QTimer.singleShot(0, _finish_item_rename)

    def _rename_peak_in_current_plot(self, old_name, new_name):
        """Update plot legends in place without replacing the live Qt canvas."""
        canvas = getattr(self, '_current_canvas', None)
        if canvas is None or self.current_plot_option not in ('DRT_data', 'Peak_comparison'):
            return
        changed = False
        for line in list(getattr(canvas.axes, 'lines', []) or []):
            if str(line.get_label()) == str(old_name):
                line.set_label(str(new_name))
                changed = True
        for annotation in list(getattr(canvas, '_peak_annotations', []) or []):
            text = str(annotation.get_text() or '')
            if text.split('\n', 1)[0] == str(old_name):
                remainder = text.split('\n', 1)
                annotation.set_text(
                    str(new_name) if len(remainder) == 1 else f'{new_name}\n{remainder[1]}'
                )
                changed = True
        if changed:
            canvas.axes.legend(frameon=False, fontsize=10, loc='best')
            canvas.draw_idle()

    def _on_peak_annotation_pick(self, event):
        """Rename a peak by double-clicking its annotation in the DRT plot."""
        mouse_event = getattr(event, 'mouseevent', None)
        if mouse_event is None or not bool(getattr(mouse_event, 'dblclick', False)):
            return
        artist = getattr(event, 'artist', None)
        gid = str(getattr(artist, 'get_gid', lambda: '')() or '')
        prefix = 'peak-result:'
        if not gid.startswith(prefix) or self.data is None:
            return
        try:
            row = int(gid[len(prefix):])
        except ValueError:
            return

        results = list(getattr(self.data, 'peak_results', []) or [])
        if row < 0 or row >= len(results):
            return
        old_name = str(results[row].get('name', f'Peak {row + 1}'))
        new_name, accepted = QtWidgets.QInputDialog.getText(
            self,
            'Rename Peak',
            'Peak name:',
            QtWidgets.QLineEdit.Normal,
            old_name,
        )
        if not accepted:
            return
        new_name = str(new_name or '').strip()
        if not new_name or new_name == old_name:
            return
        if any(
            index != row and str(result.get('name', '')) == new_name
            for index, result in enumerate(results)
        ):
            QtWidgets.QMessageBox.warning(
                self, 'Rename Peak', 'Peak names must be unique within a file.'
            )
            return

        scope = self._peak_name_scope(old_name, new_name)
        if scope == 'cancel':
            return
        if scope == 'all':
            for entry in self.data_store.values():
                for result in list(getattr(entry, 'peak_results', []) or []):
                    if str(result.get('name', '')) == old_name:
                        result['name'] = new_name
        else:
            results[row]['name'] = new_name
            self.data.peak_results = results

        self._mark_project_dirty()
        self._refresh_peak_results_panel()
        self._rename_peak_in_current_plot(old_name, new_name)

    def _peak_analyzed_keys(self):
        return [
            key for key in self._get_file_keys_in_ui_order()
            if key in self.data_store and list(getattr(self.data_store[key], 'peak_results', []) or [])
        ]

    def _update_peak_comparison_availability(self):
        available = len(self._peak_analyzed_keys()) >= 2
        button = getattr(self, '_top_tab_buttons_by_name', {}).get('Peak Comparison')
        if button is not None:
            button.setVisible(available)
        if not available and self.current_plot_option == 'Peak_comparison':
            self.plotting_callback('DRT_data')

    def _on_peak_comparison_button_press(self, event):
        """Open display settings from the Peak Comparison plot."""
        if getattr(event, 'button', None) == 3:
            self._show_peak_comparison_context_menu()

    def _show_peak_comparison_context_menu(self):
        menu = QtWidgets.QMenu(self)
        settings_action = menu.addAction('Setting')
        settings_action.triggered.connect(self._open_peak_comparison_settings)
        menu.exec_(QtGui.QCursor.pos())

    def _open_peak_comparison_settings(self):
        """Configure an optional manual y-axis range for Peak Comparison."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Peak Comparison settings')
        form = QtWidgets.QFormLayout(dialog)
        ymin_edit = QtWidgets.QLineEdit(dialog)
        ymax_edit = QtWidgets.QLineEdit(dialog)
        settings = getattr(self, 'peak_comp_settings', {}) or {}
        if settings.get('ymin') is not None:
            ymin_edit.setText(str(settings['ymin']))
        if settings.get('ymax') is not None:
            ymax_edit.setText(str(settings['ymax']))
        ymin_edit.setPlaceholderText('Auto')
        ymax_edit.setPlaceholderText('Auto')
        form.addRow('Y min', ymin_edit)
        form.addRow('Y max', ymax_edit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        form.addRow(buttons)

        def _apply():
            try:
                ymin = float(ymin_edit.text()) if ymin_edit.text().strip() else None
                ymax = float(ymax_edit.text()) if ymax_edit.text().strip() else None
                if ymin is not None and ymax is not None and ymin >= ymax:
                    raise ValueError('Y min must be smaller than Y max.')
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, 'Invalid settings', str(exc))
                return
            self.peak_comp_settings = {'ymin': ymin, 'ymax': ymax}
            self._mark_project_dirty()
            dialog.accept()

        buttons.accepted.connect(_apply)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.plotting_callback('Peak_comparison')

    def _on_peak_comparison_motion(self, event):
        """Show file, peak name, and resistance for the nearest plotted point."""
        canvas = getattr(self, '_peak_comp_canvas', None)
        if canvas is None or getattr(event, 'inaxes', None) is not canvas.axes:
            self._peak_comp_hover_last = None
            QtWidgets.QToolTip.hideText()
            return
        ex, ey = getattr(event, 'x', None), getattr(event, 'y', None)
        if ex is None or ey is None:
            return

        best = None
        for line in list(getattr(canvas, '_peak_comparison_lines', []) or []):
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)
            valid = np.isfinite(xdata) & np.isfinite(ydata)
            if not np.any(valid):
                continue
            valid_indices = np.flatnonzero(valid)
            points = canvas.axes.transData.transform(
                np.column_stack([xdata[valid], ydata[valid]])
            )
            distances = np.square(points[:, 0] - ex) + np.square(points[:, 1] - ey)
            local_index = int(np.argmin(distances))
            candidate = (float(distances[local_index]), line, int(valid_indices[local_index]))
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None or best[0] > 10.0 ** 2:
            if self._peak_comp_hover_last is not None:
                self._peak_comp_hover_last = None
                QtWidgets.QToolTip.hideText()
            return

        _, line, index = best
        peak_name = str(line.get_gid() or line.get_label())
        labels = list(getattr(canvas, '_peak_comparison_labels', []) or [])
        file_name = labels[index] if index < len(labels) else f'File {index + 1}'
        resistance = float(line.get_ydata()[index])
        tag = (peak_name, index)
        if self._peak_comp_hover_last == tag:
            return
        self._peak_comp_hover_last = tag
        QtWidgets.QToolTip.showText(
            QtGui.QCursor.pos(),
            f'File: {file_name}\nPeak: {peak_name}\nResistance: {resistance:.6g} ohm',
            self,
        )

    def _plot_peak_comparison(self):
        keys = self._peak_analyzed_keys()
        if len(keys) < 2:
            self._update_peak_comparison_availability()
            return

        names = []
        for key in keys:
            for result in self.data_store[key].peak_results:
                name = str(result.get('name', ''))
                if name and name not in names:
                    names.append(name)

        series = {}
        for name in names:
            values = []
            for key in keys:
                match = next(
                    (result for result in self.data_store[key].peak_results if result.get('name') == name),
                    None,
                )
                values.append(float(match['resistance_ohm']) if match is not None else np.nan)
            series[name] = values

        labels = [self._display_name_for_key(key) for key in keys]
        fig = Figure_Canvas()
        settings = getattr(self, 'peak_comp_settings', {}) or {}
        y_lim = (settings.get('ymin'), settings.get('ymax'))
        fig.Peak_comparison(labels, series, y_lim=y_lim)
        self._peak_comp_canvas = fig
        self._peak_comp_hover_last = None
        fig.mpl_connect('motion_notify_event', self._on_peak_comparison_motion)
        fig.mpl_connect('button_press_event', self._on_peak_comparison_button_press)
        self._show_canvas_in_plot_panel(fig)
        self.ui.plot_panel.show()
