# -*- coding: utf-8 -*-
"""Project-state capture, restore, save, open, and dirty-state workflow."""

import copy
import os
from types import SimpleNamespace

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from ..services.project_storage import load_project_state, save_project_state
from ..algorithms.runs import EIS_object
from ..services.eis_state import _ensure_mask_state
from ..ui.widgets import FileListDelegate


class ProjectControllerMixin:
    """Project behavior for the main window."""

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
            'res_H_re', 'res_H_im', 'out_scores', 'N_peaks', 'out_gamma_fit',
            'gamma_fit_tot', 'Gaussian', 'peak_components', 'peak_results',
            'peak_method', 'peak_count_mode', 'peak_analysis_complete',
            'peak_total_resistance_ohm', 'num_vectors', 'column_headings', 'df'
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
            'lambda_choice', 'shape_control_choice', 'fit_type_choice', 'peak_method_choice',
            'peak_count_choice'
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
            'peak_comp_settings': copy.deepcopy(self.peak_comp_settings),
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
            self.peak_comp_settings = copy.deepcopy(
                state.get('peak_comp_settings', self.peak_comp_settings)
                or self.peak_comp_settings
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
                    item = QtWidgets.QListWidgetItem(self._display_name_for_key(key))
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
        self._refresh_peak_results_panel()
        self._update_peak_comparison_availability()

    def _install_project_dirty_tracking(self):
        combo_names = (
            'discre_choice', 'data_used_choice', 'induct_choice', 'der_choice',
            'lambda_choice', 'shape_control_choice', 'fit_type_choice', 'peak_method_choice',
            'peak_count_choice'
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

    def _confirm_workspace_replacement(self):
        """Ask how to handle unsaved work before replacing the workspace."""
        if not self._project_dirty:
            return True

        reply = self._ask_unsaved_changes('opening another project')
        if reply == QtWidgets.QMessageBox.Cancel:
            return False
        if reply == QtWidgets.QMessageBox.Save:
            return bool(self.save_project_callback())
        return True

    def open_project_from_path(self, path, confirm_replacement=True):
        """Open a project selected by either the dialog or drag and drop."""
        if self._project_io_busy:
            return False

        path = os.path.abspath(str(path or ''))
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, 'Open Project Failed', 'Project file does not exist.')
            return False
        if not path.lower().endswith('.sdrtp'):
            QtWidgets.QMessageBox.warning(self, 'Open Project Failed', 'Unsupported project file type.')
            return False
        if confirm_replacement and not self._confirm_workspace_replacement():
            return False

        loaded = {}

        def _load():
            loaded['state'] = load_project_state(path)

        # Validate and deserialize before replacing anything in the current
        # workspace. A damaged project therefore leaves existing work intact.
        if not self._run_project_io(_load, 'Open Project Failed'):
            return False

        try:
            self._shutdown_fit_process_pool(cancel=True, terminate=True)
            self._shutdown_peak_process_pool(cancel=True, terminate=True)
            self._restore_project_state(loaded['state'])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Open Project Failed', str(exc))
            return False

        self.current_project_path = path
        self._set_project_clean()
        self.statusBar().showMessage(f'Project loaded: {path}', 2000)
        return True

    def _choose_dropped_project_target(self, path):
        """Ask whether a dropped project belongs in this or a new window."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle('Open Project')
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(f'Where would you like to open {os.path.basename(path)}?')
        box.setInformativeText(
            'Opening in this window replaces its current workspace. '
            'Opening in a new window keeps this window unchanged.'
        )
        new_button = box.addButton('New Window', QtWidgets.QMessageBox.AcceptRole)
        this_button = box.addButton('This Window', QtWidgets.QMessageBox.ActionRole)
        cancel_button = box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(this_button)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is new_button:
            return 'new'
        if clicked is this_button:
            return 'current'
        if clicked is cancel_button:
            return 'cancel'
        return 'cancel'

    def open_dropped_project(self, path):
        """Route a dropped project to a new or the current window."""
        target = self._choose_dropped_project_target(path)
        if target == 'cancel':
            return False
        if target == 'new':
            from ..app.window_manager import open_new_window

            open_new_window(project_path=path)
            return True
        return self.open_project_from_path(path, confirm_replacement=True)

    def open_new_window_callback(self):
        """Open an independent blank workspace from the Project card."""
        from ..app.window_manager import open_new_window

        open_new_window()
        return True

    def open_project_callback(self):
        """Choose a project file and delegate to the shared open workflow."""
        if self._project_io_busy:
            return False

        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', '', 'SuperDRTtools Project (*.sdrtp)'
        )
        if not path:
            return False
        return self.open_project_from_path(path)

    def closeEvent(self, event):
        if not self._project_dirty:
            self._shutdown_fit_process_pool(cancel=True, terminate=True)
            self._shutdown_peak_process_pool(cancel=True, terminate=True)
            self._release_plot_resources()
            event.accept()
            return

        reply = self._ask_unsaved_changes('closing SuperDRTtools')
        if reply == QtWidgets.QMessageBox.Cancel:
            event.ignore()
            return
        if reply == QtWidgets.QMessageBox.Discard:
            self._shutdown_fit_process_pool(cancel=True, terminate=True)
            self._shutdown_peak_process_pool(cancel=True, terminate=True)
            self._release_plot_resources()
            event.accept()
            return
        if self.save_project_callback():
            self._shutdown_fit_process_pool(cancel=True, terminate=True)
            self._shutdown_peak_process_pool(cancel=True, terminate=True)
            self._release_plot_resources()
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
