# -*- coding: utf-8 -*-
"""Controller for the fitting workflow feature."""

import os
from PyQt5 import QtCore, QtWidgets
from ..algorithms import basics
from ..infrastructure.parallel_fitting import run_fit_entry
from ..algorithms.runs import BHT_run, Bayesian_run, simple_run
from ..ui.widgets import FileListDelegate


class FittingWorkflowMixin:
    """Own the fitting workflow GUI workflow."""

    def _update_run_select_button_styles(self) -> None:
        """Blue background + white text for the selected run mode button."""
        selected_override = """
            QPushButton {
                background-color: #0A84FF;
                color: #FFFFFF;
            }
            QPushButton:hover { background-color: #0A84FF; }
            QPushButton:pressed { background-color: #0A84FF; }
        """

        # 先恢复默认样式
        self.ui.simple_run_button.setStyleSheet(self._run_btn_default_style.get('simple', ''))
        self.ui.bayesian_button.setStyleSheet(self._run_btn_default_style.get('bayesian', ''))
        self.ui.HT_button.setStyleSheet(self._run_btn_default_style.get('BHT', ''))

        height = 30
        # ✅ 恢复后立刻锁高度（避免恢复默认也触发变高）
        self.ui.simple_run_button.setFixedHeight(height)
        self.ui.bayesian_button.setFixedHeight(height)
        self.ui.HT_button.setFixedHeight(height)

        # 再给选中的叠加颜色（在默认基础上叠加，不覆盖其它）
        if self.selected_run_mode == 'simple':
            base = self._run_btn_default_style.get('simple', '')
            self.ui.simple_run_button.setStyleSheet(base + "\n" + selected_override)
            self.ui.simple_run_button.setFixedHeight(height)

        elif self.selected_run_mode == 'bayesian':
            base = self._run_btn_default_style.get('bayesian', '')
            self.ui.bayesian_button.setStyleSheet(base + "\n" + selected_override)
            self.ui.bayesian_button.setFixedHeight(height)

        elif self.selected_run_mode == 'BHT':
            base = self._run_btn_default_style.get('BHT', '')
            self.ui.HT_button.setStyleSheet(base + "\n" + selected_override)
            self.ui.HT_button.setFixedHeight(height)

    def _select_run_mode(self, mode: str):
        """Select which run mode will be used by Fit One / Fit All."""
        self.selected_run_mode = mode

        # Update button texts (keep size/alignment unchanged)
        try:
            self.ui.simple_run_button.setText('Selected' if mode == 'simple' else 'Select')
            self.ui.bayesian_button.setText('Selected' if mode == 'bayesian' else 'Select')
            self.ui.HT_button.setText('Selected' if mode == 'BHT' else 'Select')
        except Exception:
            pass

        self._update_run_select_button_styles()

    def _confirm_raw_fit_if_needed(self, keys, mode: str) -> bool:
        """Allow fitting without showing a K-K validation warning dialog.

        DRT fitting uses the current raw/masked experimental EIS data, not the
        K-K fitted curve, so K-K validation is treated as an optional quality
        check rather than a mandatory gate for fitting.
        """
        return True

    def _current_fit_signature(self, mode: str):
        """Build a signature of the current UI options + selected mode to support skip logic."""
        induct_used = int(self.ui.induct_choice.currentIndex())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())

        if mode == 'simple' or mode == 'bayesian':
            rbf_type = str(self.ui.discre_choice.currentText())
            data_used = str(self.ui.data_used_choice.currentText())
            der_used = str(self.ui.der_choice.currentText())
            cv_type = str(self.ui.lambda_choice.currentText())
            reg_param = float(self.ui.reg_param_entry.text())
            sig = ('mode', mode,
                   'rbf', rbf_type,
                   'data_used', data_used,
                   'induct', induct_used,
                   'der', der_used,
                   'cv', cv_type,
                   'reg', reg_param,
                   'shape', shape_control,
                   'coeff', coeff)
            if mode == 'bayesian':
                sample_number = int(self.ui.sample_no_entry.text())
                sig = sig + ('samples', sample_number)
            return sig

        if mode == 'BHT':
            rbf_type = str(self.ui.discre_choice.currentText())
            der_used = str(self.ui.der_choice.currentText())
            return ('mode', mode,
                    'rbf', rbf_type,
                    'induct', induct_used,
                    'der', der_used,
                    'shape', shape_control,
                    'coeff', coeff)

        return ('mode', mode)

    def _update_file_status(self, key: str, fitted: bool):
        """Update the right-side status label in the Files list."""
        if not hasattr(self.ui, 'files_list') or not key:
            return
        status_text = 'Fitted' if fitted else 'Unfitted'
        for i in range(self.ui.files_list.count()):
            it = self.ui.files_list.item(i)
            if it and it.data(QtCore.Qt.UserRole) == key:
                it.setData(FileListDelegate.STATUS_ROLE, status_text)
                # repaint without disturbing selection
                self.ui.files_list.viewport().update()
                break

    def _fit_key(self, key: str, mode: str, signature=None, update_ui: bool = False):
        """Run the selected mode on one file and store the result."""
        if key not in self.data_store:
            return

        signature = signature if signature is not None else self._current_fit_signature(mode)
        params = self._build_fit_params(mode)
        entry = run_fit_entry(self.data_store[key], mode, params)

        if update_ui and mode == 'simple':
            try:
                self.ui.reg_param_entry_2.setText(str(entry.lambda_value))
            except Exception:
                pass

        self.data_store[key] = entry
        if key == getattr(self, 'current_file_key', None):
            self.data = entry

        if key in getattr(self, 'file_meta', {}):
            meta = dict(self.file_meta.get(key) or {})
            meta.update({
                'fitted': True,
                'signature': signature,
                'peak_analyzed': False,
                'peak_signature': None,
            })
            meta.setdefault('display_name', os.path.basename(key))
            self.file_meta[key] = meta
        self._update_file_status(key, fitted=True)
        self._mark_project_dirty()
        self._refresh_peak_results_panel()
        self._update_peak_comparison_availability()

    def fit_selected_callback(self):
        """Fit only the current file and keep progress visible in the status bar."""
        if getattr(self, '_peak_active', False):
            self.statusBar().showMessage('Wait for Peak Analysis to finish.', 2000)
            return
        if getattr(self, 'current_file_key', None) is None or self.data is None:
            return

        mode = getattr(self, 'selected_run_mode', 'simple')
        if not self._confirm_raw_fit_if_needed([self.current_file_key], mode):
            return

        key = self.current_file_key
        display_name = self._display_name_for_key(key)
        try:
            self.statusBar().showMessage(f'Fitting: 0/1 completed (0%) | {display_name}')
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        try:
            self._fit_key(key, mode, update_ui=True)
            self.plotting_callback(
                self.current_plot_option if getattr(self, 'current_plot_option', None) in ('DRT_comparison',
                                                                                           'DRT_map') else 'DRT_data')
        except Exception as e:
            try:
                self.statusBar().showMessage(f'Fitting failed: {display_name}', 5000)
                QtWidgets.QMessageBox.warning(self, 'Fit failed', str(e))
            except Exception:
                pass
            return

        try:
            self.statusBar().showMessage(f'Fitting complete: 1/1 (100%) | {display_name}', 4000)
        except Exception:
            pass

    def fit_all_callback(self):
        """Fit all files in the list in parallel; skip those already fitted with the same signature."""
        if getattr(self, '_peak_active', False):
            self.statusBar().showMessage('Wait for Peak Analysis to finish.', 2000)
            return
        if not getattr(self, 'data_store', None):
            return

        # prevent overlapping Fit All runs
        if getattr(self, '_fit_all_active', False):
            try:
                self.statusBar().showMessage('Fit all is running...', 1500)
            except Exception:
                pass
            return

        mode = getattr(self, 'selected_run_mode', 'simple')

        # snapshot signature + parameters once (UI must only be read in main thread)
        try:
            signature = self._current_fit_signature(mode)
            params = self._build_fit_params(mode)
        except Exception as e:
            try:
                QtWidgets.QMessageBox.warning(self, "Fit parameters error", str(e))
            except Exception:
                pass
            return

        # preserve current view
        self._fit_all_restore = {
            'current_key': getattr(self, 'current_file_key', None),
            'current_plot': getattr(self, 'current_plot_option', 'EIS_data'),
        }

        # iterate in UI order if possible
        keys = []
        if hasattr(self.ui, 'files_list'):
            for i in range(self.ui.files_list.count()):
                it = self.ui.files_list.item(i)
                if it:
                    k = it.data(QtCore.Qt.UserRole)
                    if k:
                        keys.append(k)
        if not keys:
            keys = list(self.data_store.keys())

        n_skip = 0
        to_fit = []
        for key in keys:
            meta = getattr(self, 'file_meta', {}).get(key)
            if meta and meta.get('fitted') and meta.get('signature') == signature:
                n_skip += 1
                continue
            if key in self.data_store:
                to_fit.append(key)

        if not to_fit:
            try:
                self.statusBar().showMessage(f'Fit all done: 0 fitted, {n_skip} skipped', 2000)
            except Exception:
                pass
            return

        if not self._confirm_raw_fit_if_needed(to_fit, mode):
            try:
                self.statusBar().showMessage('Fit all cancelled.', 1500)
            except Exception:
                pass
            return

        self._fit_all_active = True
        self._fit_all_total = len(to_fit)
        self._fit_all_pending = len(to_fit)
        self._fit_all_done = 0
        self._fit_all_skipped = n_skip
        self._fit_all_errors = 0

        try:
            self.statusBar().showMessage(
                f'Preparing Fit All: 0/{self._fit_all_total} tasks submitted'
            )
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        # The standalone backend starts as many independent fits as current
        # CPU and commit-memory headroom safely allow, then scales gradually.
        params['native_threads'] = 1
        tasks = [(key, self.data_store[key]) for key in to_fit]

        try:
            info = self._fit_process_manager.start(tasks, mode, params, signature)
            self._fit_worker_count = int(info.get('worker_count', 1) or 1)
            self._fit_initial_worker_count = int(info.get('initial_worker_count', self._fit_worker_count) or 1)
            self._fit_max_worker_count = int(info.get('maximum_worker_count', self._fit_worker_count) or self._fit_worker_count)
            if info.get('memory_limited_initial'):
                available = info.get('initial_available_commit_gib')
                available_text = (
                    f', {float(available):.1f} GiB commit available'
                    if available is not None else ''
                )
                self._fit_process_note = (
                    f'memory-safe start {self._fit_initial_worker_count}→'
                    f'{self._fit_worker_count}{available_text}'
                )
            else:
                self._fit_process_note = ''
            self.statusBar().showMessage(
                f'Preparing Fit All: {self._fit_all_total}/{self._fit_all_total} tasks queued | '
                f'{self._fit_worker_count} active / {self._fit_max_worker_count} max processes '
                f'(physical {info.get("physical_cores", "?")}, logical {info.get("logical_cores", "?")})'
            )
            self._fit_poll_timer.start()
        except Exception as e:
            self._fit_all_active = False
            self._fit_all_pending = 0
            self._shutdown_fit_process_pool(cancel=True, terminate=True)
            error_text = str(e)
            try:
                QtWidgets.QMessageBox.warning(self, 'Fit All failed to start', error_text)
            except Exception:
                pass
            return

        self._update_fit_all_progress_status()

    def _update_fit_all_progress_status(self):
        """Keep Fit All progress visible until all process tasks finish."""
        total = int(getattr(self, '_fit_all_total', 0) or 0)
        done = int(getattr(self, '_fit_all_done', 0) or 0)
        errors = int(getattr(self, '_fit_all_errors', 0) or 0)
        completed = min(total, done + errors)
        remaining = max(0, total - completed)
        percent = int(round(100.0 * completed / total)) if total else 100
        processes = int(getattr(self, '_fit_worker_count', 0) or 0)
        skipped = int(getattr(self, '_fit_all_skipped', 0) or 0)
        note = str(getattr(self, '_fit_process_note', '') or '')
        suffix = f' | {note}' if note else ''
        try:
            self.statusBar().showMessage(
                f'Fitting: {completed}/{total} completed ({percent}%) | '
                f'{remaining} remaining | {processes} processes | '
                f'{errors} errors | {skipped} skipped{suffix}'
            )
        except Exception:
            pass

    def _build_fit_params(self, mode: str) -> dict:
        """Read all fitting parameters from UI once; safe to pass into worker processes."""
        params = {}
        # For parallel Fit All, limit each task to 1 native BLAS/OMP thread so
        # multiple fits can run concurrently without oversubscribing the CPU.
        params['native_threads'] = 1
        params['rbf_type'] = str(self.ui.discre_choice.currentText())
        params['induct_used'] = int(self.ui.induct_choice.currentIndex())
        params['der_used'] = str(self.ui.der_choice.currentText())
        params['shape_control'] = str(self.ui.shape_control_choice.currentText())
        params['coeff'] = float(self.ui.FWHM_entry.text())

        if mode in ('simple', 'bayesian'):
            params['data_used'] = str(self.ui.data_used_choice.currentText())
            params['cv_type'] = str(self.ui.lambda_choice.currentText())
            params['reg_param'] = float(self.ui.reg_param_entry.text())

        if mode == 'bayesian':
            params['sample_number'] = int(float(self.ui.sample_no_entry.text()))

        return params

    def _poll_fit_processes(self):
        """Collect process results and handle automatic page-file fallback."""
        if not getattr(self, '_fit_all_active', False):
            self._fit_poll_timer.stop()
            return

        try:
            batch = self._fit_process_manager.poll()
        except Exception as exc:
            # A manager-level error is terminal; account for every unfinished task.
            unfinished = max(0, int(getattr(self, '_fit_all_pending', 0) or 0))
            for index in range(unfinished):
                self._on_fit_worker_error(f'Fit task {index + 1}', str(exc))
            return

        retry = batch.get('retry')
        scale = batch.get('scale')
        if retry:
            self._fit_worker_count = int(retry.get('to_workers', 1) or 1)
            self._fit_max_worker_count = self._fit_worker_count
            self._fit_process_note = (
                f'virtual memory fallback {retry.get("from_workers", "?")}→'
                f'{retry.get("to_workers", "?")}'
            )
        elif scale:
            self._fit_worker_count = int(scale.get('to_workers', self._fit_worker_count) or self._fit_worker_count)
            cpu_value = scale.get('cpu_percent')
            cpu_text = f' | CPU {float(cpu_value):.0f}%' if cpu_value is not None else ''
            self._fit_process_note = (
                f'auto scale {scale.get("from_workers", "?")}→'
                f'{scale.get("to_workers", "?")}{cpu_text}'
            )
        else:
            self._fit_worker_count = int(batch.get('worker_count', self._fit_worker_count) or self._fit_worker_count)

        for key, fitted_entry, signature in batch.get('results', []):
            self._on_fit_worker_finished(key, fitted_entry, signature)

        for key, message in batch.get('errors', []):
            self._on_fit_worker_error(key, message)

        if retry or scale:
            self._update_fit_all_progress_status()

        if batch.get('done') and getattr(self, '_fit_all_pending', 0) <= 0:
            self._fit_poll_timer.stop()

    def _shutdown_fit_process_pool(self, cancel: bool = False, terminate: bool = False):
        """Stop the standalone Fit All backend."""
        try:
            self._fit_poll_timer.stop()
        except Exception:
            pass
        try:
            self._fit_process_manager.shutdown(cancel=cancel, terminate=terminate)
        except Exception:
            pass
        self._fit_worker_count = 0

    @QtCore.pyqtSlot(str, object, object)
    def _on_fit_worker_finished(self, key: str, fitted_entry, signature):
        """Main-thread slot: store results + update status."""
        try:
            self.data_store[key] = fitted_entry
            if key == getattr(self, 'current_file_key', None):
                self.data = fitted_entry

            if key in getattr(self, 'file_meta', {}):
                meta = dict(self.file_meta.get(key) or {})
                meta.update({
                    'fitted': True,
                    'signature': signature,
                    'peak_analyzed': False,
                    'peak_signature': None,
                })
                meta.setdefault('display_name', os.path.basename(key))
                self.file_meta[key] = meta
            self._update_file_status(key, fitted=True)
            self._mark_project_dirty()
            self._update_peak_comparison_availability()
            if key == self.current_file_key:
                self._refresh_peak_results_panel()
        except Exception:
            pass

        try:
            self._fit_all_done += 1
            self._fit_all_pending -= 1
        except Exception:
            return

        self._update_fit_all_progress_status()
        if self._fit_all_pending <= 0:
            self._finish_fit_all_parallel()

    @QtCore.pyqtSlot(str, str)
    def _on_fit_worker_error(self, key: str, msg: str):
        try:
            self._fit_all_errors += 1
            self._fit_all_pending -= 1
        except Exception:
            return

        try:
            if key in getattr(self, 'file_meta', {}):
                meta = dict(self.file_meta.get(key) or {})
                meta.update({'fitted': False, 'signature': None})
                meta.setdefault('display_name', os.path.basename(key))
                self.file_meta[key] = meta
            self._update_file_status(key, fitted=False)
        except Exception:
            pass

        self._update_fit_all_progress_status()
        if self._fit_all_pending <= 0:
            self._finish_fit_all_parallel()

    def _finish_fit_all_parallel(self):
        """Finalize Fit All after all process workers complete."""
        self._fit_all_active = False
        self._shutdown_fit_process_pool(cancel=False)
        self._fit_process_note = ''

        try:
            current_key = self._fit_all_restore.get('current_key')
            current_plot = self._fit_all_restore.get('current_plot', 'EIS_data')
        except Exception:
            current_key = getattr(self, 'current_file_key', None)
            current_plot = getattr(self, 'current_plot_option', 'EIS_data')

        if current_key in getattr(self, 'data_store', {}):
            self.current_file_key = current_key
            self.data = self.data_store[current_key]

        self.current_plot_option = current_plot
        try:
            self.plotting_callback(current_plot)
        except Exception:
            pass

        try:
            self.statusBar().showMessage(
                f'Fit all done: {self._fit_all_done}/{self._fit_all_total} fitted, '
                f'{self._fit_all_skipped} skipped, {self._fit_all_errors} errors',
                5000
            )
        except Exception:
            pass

    def simple_run_callback(self):  # callback for simple ridge regularization

        if self.data is None:
            return

        # we first read the parameters chosen by the users
        rbf_type = str(self.ui.discre_choice.currentText())
        data_used = str(self.ui.data_used_choice.currentText())
        induct_used = int(self.ui.induct_choice.currentIndex())
        der_used = str(self.ui.der_choice.currentText())
        cv_type = str(self.ui.lambda_choice.currentText())
        reg_param = float(self.ui.reg_param_entry.text())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())

        # we perform the computation
        self.data = simple_run(self.data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                               der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control,
                               coeff=coeff)

        entry = self.data
        # optimally select the regularization level
        if cv_type == 'custom':
            entry.lambda_value = reg_param
        else:
            entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                       data_used, induct_used, -3, cv_type)

        # Update the QLineEdit with the computed regularization parameter
        self.ui.reg_param_entry_2.setText(str(entry.lambda_value))

        self._mark_project_dirty()
        self.plotting_callback('DRT_data')

    def bayesian_run_callback(self):  # callback for Bayesian regularization

        if self.data is None:
            return

        # we first read the parameters chosen by the users
        rbf_type = str(self.ui.discre_choice.currentText())
        data_used = str(self.ui.data_used_choice.currentText())
        induct_used = int(self.ui.induct_choice.currentIndex())
        der_used = str(self.ui.der_choice.currentText())
        cv_type = str(self.ui.lambda_choice.currentText())
        reg_param = float(self.ui.reg_param_entry.text())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())
        sample_number = int(self.ui.sample_no_entry.text())

        # we perform the computation
        self.data = Bayesian_run(self.data, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                                 der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control,
                                 coeff=coeff, NMC_sample=sample_number)
        self._mark_project_dirty()
        self.plotting_callback('DRT_data')

    def BHT_run_callback(self):  # callback for Hilbert transform run

        if self.data is None:
            return

        # we first read the parameters chosen by the users
        rbf_type = str(self.ui.discre_choice.currentText())
        der_used = str(self.ui.der_choice.currentText())
        shape_control = str(self.ui.shape_control_choice.currentText())
        coeff = float(self.ui.FWHM_entry.text())

        # we perform the computation
        self.data = BHT_run(self.data, rbf_type, der_used, shape_control, coeff)
        ##
        self._mark_project_dirty()
        self.plotting_callback('DRT_data')

