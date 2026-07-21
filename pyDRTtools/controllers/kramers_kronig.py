# -*- coding: utf-8 -*-
"""Controller for the kramers kronig feature."""

from PyQt5 import QtWidgets
from ..services.eis_state import (
    _build_kk_signature, _clear_kk_results, _refresh_kk_current_arrays,
    _run_lin_kk,
)


class KramersKronigControllerMixin:
    """Own the kramers kronig GUI workflow."""

    def kk_run_callback(self):
        """Run K-K validation only for files whose data/settings changed."""
        keys = self._get_file_keys_in_ui_order()
        if not keys:
            if self.data is None:
                return
            keys = [getattr(self, 'current_file_key', None)] if getattr(self, 'current_file_key', None) else []
            if not keys:
                return

        try:
            c = float(self.ui.cutoff_entry.text())
            max_m = int(float(self.ui.max_elements_entry.text()))
            fit_type = str(self.ui.fit_type_choice.currentText()).strip().lower()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'K-K parameters error', f'Invalid K-K parameters: {e}')
            return

        self.kk_settings = {'c': c, 'max_m': max_m, 'fit_type': fit_type}

        done = 0
        skipped = 0
        errors = 0
        last_error = None
        for key in keys:
            if key is None:
                continue
            entry = self.data_store.get(key, self.data if key == getattr(self, 'current_file_key', None) else None)
            if entry is None:
                continue

            try:
                current_signature = _build_kk_signature(entry, c=c, max_m=max_m, fit_type=fit_type)
                if bool(getattr(entry, 'kk_valid', False)) and getattr(entry, 'kk_signature', None) == current_signature:
                    _refresh_kk_current_arrays(entry)
                    if key in self.data_store:
                        self.data_store[key] = entry
                    if key == getattr(self, 'current_file_key', None):
                        self.data = entry
                    skipped += 1
                    continue

                _clear_kk_results(entry)
                entry = _run_lin_kk(entry, c=c, max_m=max_m, fit_type=fit_type)
                if key in self.data_store:
                    self.data_store[key] = entry
                if key == getattr(self, 'current_file_key', None):
                    self.data = entry

                # K-K validation does not change the DRT input data. Therefore,
                # it must not reset the existing Fit status/signature.
                done += 1
            except Exception as e:
                errors += 1
                last_error = str(e)
                try:
                    _clear_kk_results(entry)
                except Exception:
                    pass

        if getattr(self, 'current_file_key', None) in self.data_store:
            self.data = self.data_store[self.current_file_key]

        if getattr(self, 'current_plot_option', None) in ('EIS_data', 'KK_residual'):
            self.plotting_callback(self.current_plot_option)

        if done > 0:
            self._mark_project_dirty()

        msg = f'K-K analysis done: {done} processed, {skipped} skipped'
        if errors:
            msg += f', {errors} error(s)'
        try:
            self.statusBar().showMessage(msg, 3000)
        except Exception:
            pass

        if errors and last_error:
            try:
                QtWidgets.QMessageBox.warning(self, 'K-K analysis warning', last_error)
            except Exception:
                pass
