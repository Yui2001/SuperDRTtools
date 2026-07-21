# -*- coding: utf-8 -*-
"""Controller for the masking feature."""

import os
import numpy as np
from PyQt5 import QtGui, QtWidgets
from ..services.eis_state import (
    _clear_kk_results, _ensure_mask_state, _get_entry_raw_arrays,
    _rebuild_visible_data_from_raw, _refresh_kk_current_arrays,
    _safe_kk_residual_max,
)


class MaskControllerMixin:
    """Own the masking GUI workflow."""

    def _apply_inductance_to_key(self, key: str, reset_method: bool = True, clear_kk: bool = True):
        """Rebuild one file's working data after inductance/mask changes."""
        if key not in self.data_store:
            return

        obj = self.data_store[key]
        _ensure_mask_state(obj)
        obj = _rebuild_visible_data_from_raw(obj, int(self.ui.induct_choice.currentIndex()))

        if reset_method:
            obj.method = 'none'
        if clear_kk:
            _clear_kk_results(obj)
        else:
            _refresh_kk_current_arrays(obj)

        if reset_method and key in getattr(self, 'file_meta', {}):
            meta = dict(self.file_meta.get(key) or {})
            meta.update({'fitted': False, 'signature': None})
            meta.setdefault('display_name', os.path.basename(key))
            self.file_meta[key] = meta
            self._update_file_status(key, fitted=False)

        self.data_store[key] = obj

    def _rebuild_after_mask_change(self, key: str, refresh_plot: bool = True):
        """Rebuild one dataset after mask changes and optionally refresh the EIS plot."""
        if key not in getattr(self, 'data_store', {}):
            return
        self._apply_inductance_to_key(key, reset_method=True, clear_kk=False)
        if key == getattr(self, 'current_file_key', None):
            self.data = self.data_store[key]
        if refresh_plot:
            self.plotting_callback('EIS_data')
        self._mark_project_dirty()

    def _toggle_mask_point(self, key: str, raw_index: int, masked: bool):
        entry = self.data_store.get(key)
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        if 0 <= int(raw_index) < entry.mask_manual_raw.size:
            entry.mask_manual_raw[int(raw_index)] = bool(masked)
        self.data_store[key] = entry
        self._eis_selected_raw_indices = []
        self._rebuild_after_mask_change(key)

    def _set_mask_for_raw_indices(self, key: str, raw_indices, masked: bool):
        entry = self.data_store.get(key)
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        idx = np.asarray(list(raw_indices or []), dtype=int).reshape(-1)
        if idx.size:
            idx = idx[(idx >= 0) & (idx < entry.mask_manual_raw.size)]
            if idx.size:
                entry.mask_manual_raw[idx] = bool(masked)
        self.data_store[key] = entry
        self._eis_selected_raw_indices = []
        self._rebuild_after_mask_change(key)

    def _clear_masks_for_key(self, key: str, clear_manual: bool = True, clear_auto: bool = True):
        entry = self.data_store.get(key)
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        if clear_manual:
            entry.mask_manual_raw[:] = False
        if clear_auto:
            entry.mask_auto_raw[:] = False
        self.data_store[key] = entry
        self._eis_selected_raw_indices = []
        self._rebuild_after_mask_change(key)

    @staticmethod
    def _build_auto_mask_from_settings(entry, freq_min=None, freq_max=None, kk_threshold=None):
        """Build an automatic mask from frequency and/or K-K residual criteria.

        Frequency min/max define the frequency range to KEEP. Points outside that
        range are masked. The previous implementation masked points inside the
        selected range, which could remove the entire spectrum and make the Qt
        callback terminate with an uncaught exception.
        """
        entry = _ensure_mask_state(entry)
        freq0, _, _, _ = _get_entry_raw_arrays(entry)
        auto_mask = np.zeros(freq0.size, dtype=bool)

        if freq_min is not None or freq_max is not None:
            f_lo = float(freq_min) if freq_min is not None else None
            f_hi = float(freq_max) if freq_max is not None else None

            if f_lo is not None and (not np.isfinite(f_lo) or f_lo <= 0):
                raise ValueError('Frequency min must be a positive finite number.')
            if f_hi is not None and (not np.isfinite(f_hi) or f_hi <= 0):
                raise ValueError('Frequency max must be a positive finite number.')
            if f_lo is not None and f_hi is not None and f_lo > f_hi:
                raise ValueError('Frequency min cannot be greater than frequency max.')

            # Mask points OUTSIDE the retained frequency range.
            if f_lo is not None:
                auto_mask |= freq0 < f_lo
            if f_hi is not None:
                auto_mask |= freq0 > f_hi

        if kk_threshold is not None:
            threshold = float(kk_threshold)
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError('K-K residual threshold must be a non-negative finite number.')

            kk_re = np.asarray(
                getattr(entry, 'kk_res_re_raw_pct', np.full(freq0.size, np.nan)),
                dtype=float,
            ).reshape(-1)
            kk_im = np.asarray(
                getattr(entry, 'kk_res_im_raw_pct', np.full(freq0.size, np.nan)),
                dtype=float,
            ).reshape(-1)
            if kk_re.size != freq0.size or kk_im.size != freq0.size:
                raise ValueError('Current K-K residuals are unavailable for threshold masking.')
            kk_max = _safe_kk_residual_max(kk_re, kk_im, freq0.size)
            auto_mask |= np.isfinite(kk_max) & (kk_max > threshold)

        return auto_mask

    def _apply_mask_settings_to_keys(self, keys, freq_min=None, freq_max=None, kk_threshold=None):
        """Safely apply automatic mask settings to one or more files.

        Every target file is validated before any file is changed. This prevents
        Apply to All Files from leaving a partially modified project when one
        spectrum would have too few remaining points.
        """
        valid_keys = [k for k in (keys or []) if k in self.data_store]
        if not valid_keys:
            raise ValueError('No valid files were selected for masking.')

        induct_index = int(self.ui.induct_choice.currentIndex())
        proposed = []

        # Validation pass: do not mutate any entry yet.
        for target_key in valid_keys:
            entry = self.data_store.get(target_key)
            if entry is None:
                continue

            entry = _ensure_mask_state(entry)
            auto_mask = self._build_auto_mask_from_settings(
                entry,
                freq_min=freq_min,
                freq_max=freq_max,
                kk_threshold=kk_threshold,
            )

            freq0, _, zim0, _ = _get_entry_raw_arrays(entry)
            manual_mask = np.asarray(
                getattr(entry, 'mask_manual_raw', np.zeros(freq0.size, dtype=bool)),
                dtype=bool,
            ).reshape(-1)

            visible_keep = np.ones(freq0.size, dtype=bool)
            if induct_index == 2:  # discard inductive data
                visible_keep = (-zim0) > 0

            active_keep = visible_keep & (~(manual_mask | auto_mask))
            remaining = int(np.count_nonzero(active_keep))
            if remaining < 3:
                raise ValueError(
                    f'{os.path.basename(target_key)} would have only {remaining} valid point(s) '
                    'after masking. Please relax the frequency range, K-K threshold, '
                    'manual mask, or inductance filtering.'
                )

            proposed.append((target_key, entry, auto_mask))

        if not proposed:
            raise ValueError('No valid files were available for masking.')

        # Commit pass. Validation above guarantees that the normal rebuild cannot
        # fail because all points were removed.
        changed_keys = []
        try:
            for target_key, entry, auto_mask in proposed:
                entry.mask_auto_raw = np.asarray(auto_mask, dtype=bool).copy()
                entry.mask_settings = {
                    'freq_min': freq_min,
                    'freq_max': freq_max,
                    'kk_threshold': kk_threshold,
                }
                self.data_store[target_key] = entry
                self._rebuild_after_mask_change(target_key, refresh_plot=False)
                changed_keys.append(target_key)
        except Exception as exc:
            raise RuntimeError(f'Failed while rebuilding masked data: {exc}') from exc

        current_key = getattr(self, 'current_file_key', None)
        if current_key in changed_keys:
            self.data = self.data_store.get(current_key)
            self.plotting_callback('EIS_data')

    def _clear_auto_mask_for_keys(self, keys):
        """Clear automatic masks for one or more files."""
        valid_keys = [k for k in (keys or []) if k in self.data_store]
        for key in valid_keys:
            entry = self.data_store.get(key)
            if entry is None:
                continue
            entry = _ensure_mask_state(entry)
            entry.mask_auto_raw[:] = False
            entry.mask_settings = {'freq_min': None, 'freq_max': None, 'kk_threshold': None}
            self.data_store[key] = entry
            self._rebuild_after_mask_change(key, refresh_plot=False)

        current_key = getattr(self, 'current_file_key', None)
        if current_key in valid_keys:
            self.data = self.data_store.get(current_key)
            self.plotting_callback('EIS_data')

    def _apply_mask_settings_to_current(self, freq_min=None, freq_max=None, kk_threshold=None):
        """Backward-compatible wrapper: apply mask settings only to the current file."""
        key = getattr(self, 'current_file_key', None)
        if key is None:
            return
        self._apply_mask_settings_to_keys([key], freq_min=freq_min, freq_max=freq_max, kk_threshold=kk_threshold)

    def _open_mask_setting_dialog(self):
        """Open a clearer mask-settings dialog for the current file or all files."""
        key = getattr(self, 'current_file_key', None)
        entry = self.data_store.get(key) if key else None
        if entry is None:
            return
        entry = _ensure_mask_state(entry)
        settings = getattr(entry, 'mask_settings', {}) or {}

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Mask Settings')
        dlg.setModal(True)
        dlg.resize(560, 260)

        root = QtWidgets.QVBoxLayout(dlg)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QtWidgets.QLabel('Keep points inside the frequency range and mask points exceeding the K-K residual threshold.', dlg)
        title.setWordWrap(True)
        title.setStyleSheet('font-weight: 600;')
        root.addWidget(title)

        subtitle = QtWidgets.QLabel(
            'Apply affects the current file only. Use "Apply to All Files" to apply the same criteria to every imported file.',
            dlg,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet('color: #6E6E73;')
        root.addWidget(subtitle)

        form_widget = QtWidgets.QWidget(dlg)
        form = QtWidgets.QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        fmin_edit = QtWidgets.QLineEdit(dlg)
        fmax_edit = QtWidgets.QLineEdit(dlg)
        thr_edit = QtWidgets.QLineEdit(dlg)
        for edit in (fmin_edit, fmax_edit, thr_edit):
            edit.setPlaceholderText('Optional')
            edit.setMinimumHeight(30)

        if settings.get('freq_min') is not None:
            fmin_edit.setText(str(settings.get('freq_min')))
        if settings.get('freq_max') is not None:
            fmax_edit.setText(str(settings.get('freq_max')))
        if settings.get('kk_threshold') is not None:
            thr_edit.setText(str(settings.get('kk_threshold')))

        form.addRow('Frequency min (Hz)', fmin_edit)
        form.addRow('Frequency max (Hz)', fmax_edit)
        form.addRow('Lin-KK residual > (%)', thr_edit)
        root.addWidget(form_widget)

        button_grid = QtWidgets.QGridLayout()
        button_grid.setHorizontalSpacing(8)
        button_grid.setVerticalSpacing(8)

        clear_auto_btn = QtWidgets.QPushButton('Clear Current File Auto Mask', dlg)
        clear_auto_all_btn = QtWidgets.QPushButton('Clear All Files Auto Masks', dlg)
        apply_btn = QtWidgets.QPushButton('Apply to This File', dlg)
        apply_all_btn = QtWidgets.QPushButton('Apply to All Files', dlg)

        apply_btn.setDefault(True)
        apply_btn.setAutoDefault(True)

        # 让四个按钮宽度一致，上下更整齐
        for btn in (clear_auto_btn, clear_auto_all_btn, apply_btn, apply_all_btn):
            btn.setMinimumWidth(170)
            btn.setMinimumHeight(32)

        # 两排两列，上下对齐
        button_grid.addWidget(clear_auto_btn, 0, 0)
        button_grid.addWidget(clear_auto_all_btn, 0, 1)
        button_grid.addWidget(apply_btn, 1, 0)
        button_grid.addWidget(apply_all_btn, 1, 1)

        root.addLayout(button_grid)

        def _parse_optional(edit):
            s = (edit.text() or '').strip()
            return None if s == '' else float(s)

        def _read_settings(target_keys):
            try:
                fmin = _parse_optional(fmin_edit)
                fmax = _parse_optional(fmax_edit)
                thr = _parse_optional(thr_edit)
                if thr is not None:
                    missing = []
                    for target_key in target_keys:
                        target_entry = self.data_store.get(target_key)
                        if target_entry is None or not getattr(target_entry, 'kk_valid', False):
                            missing.append(os.path.basename(target_key))
                    if missing:
                        raise ValueError(
                            'Please run K-K validation first for: ' + ', '.join(missing[:5]) + (' ...' if len(missing) > 5 else '')
                        )
                return fmin, fmax, thr
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Invalid mask setting', str(e))
                return None

        def _apply_current():
            values = _read_settings([key])
            if values is None:
                return
            try:
                fmin, fmax, thr = values
                self._apply_mask_settings_to_current(
                    freq_min=fmin,
                    freq_max=fmax,
                    kk_threshold=thr,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, 'Mask could not be applied', str(exc))
                return
            dlg.accept()

        def _apply_all():
            target_keys = self._get_file_keys_in_ui_order()
            values = _read_settings(target_keys)
            if values is None:
                return
            try:
                fmin, fmax, thr = values
                self._apply_mask_settings_to_keys(
                    target_keys,
                    freq_min=fmin,
                    freq_max=fmax,
                    kk_threshold=thr,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, 'Masks could not be applied', str(exc))
                return
            dlg.accept()

        def _clear_current():
            try:
                self._clear_auto_mask_for_keys([key])
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, 'Auto mask could not be cleared', str(exc))
                return
            dlg.accept()

        def _clear_all():
            try:
                self._clear_auto_mask_for_keys(self._get_file_keys_in_ui_order())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, 'Auto masks could not be cleared', str(exc))
                return
            dlg.accept()

        apply_btn.clicked.connect(_apply_current)
        apply_all_btn.clicked.connect(_apply_all)
        clear_auto_btn.clicked.connect(_clear_current)
        clear_auto_all_btn.clicked.connect(_clear_all)
        dlg.exec_()

    def _find_eis_raw_index_from_event(self, fig, event, threshold_px: float = 10.0):
        payload = getattr(fig, '_eis_plot_payload', None)
        if not payload or getattr(event, 'inaxes', None) is not fig.axes:
            return None
        xs = np.asarray(payload.get('x', []), dtype=float)
        ys = np.asarray(payload.get('y', []), dtype=float)
        raw_indices = np.asarray(payload.get('raw_indices', []), dtype=int)
        if xs.size == 0 or ys.size == 0 or raw_indices.size != xs.size:
            return None
        try:
            pts = fig.axes.transData.transform(np.column_stack([xs, ys]))
        except Exception:
            return None
        dist2 = (pts[:, 0] - event.x) ** 2 + (pts[:, 1] - event.y) ** 2
        j = int(np.argmin(dist2))
        if float(np.sqrt(dist2[j])) > float(threshold_px):
            return None
        return int(raw_indices[j])

    def _apply_eis_selection_highlight(self, fig):
        if fig is None:
            return

        ax = fig.axes
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
        old_autoscale = ax.get_autoscale_on()

        for artist in getattr(fig, '_eis_selection_artists', []):
            try:
                artist.remove()
            except Exception:
                pass
        fig._eis_selection_artists = []

        payload = getattr(fig, '_eis_plot_payload', None)
        if payload:
            selected = np.asarray(getattr(self, '_eis_selected_raw_indices', []), dtype=int).reshape(-1)
            raw_indices = np.asarray(payload.get('raw_indices', []), dtype=int).reshape(-1)
            xs = np.asarray(payload.get('x', []), dtype=float).reshape(-1)
            ys = np.asarray(payload.get('y', []), dtype=float).reshape(-1)

            if selected.size > 0 and raw_indices.size > 0 and xs.size == raw_indices.size and ys.size == raw_indices.size:
                keep = np.isin(raw_indices, selected)
                if np.any(keep):
                    artist = ax.scatter(
                        xs[keep], ys[keep],
                        s=95, facecolors='none',
                        edgecolors='#00B5FF',
                        linewidths=1.5, zorder=30
                    )
                    fig._eis_selection_artists = [artist]

        ax.set_autoscale_on(False)
        ax.set_xlim(old_xlim)
        ax.set_ylim(old_ylim)

        try:
            fig.draw_idle()
        except Exception:
            fig.draw()

        ax.set_autoscale_on(old_autoscale)

    def _on_eis_box_select(self, eclick, erelease):
        fig = getattr(self, '_eis_canvas', None)
        if fig is None or getattr(fig, '_eis_plot_payload', None) is None:
            return
        if eclick is None or erelease is None:
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return

        payload = fig._eis_plot_payload
        raw_indices = np.asarray(payload.get('raw_indices', []), dtype=int).reshape(-1)
        xs = np.asarray(payload.get('x', []), dtype=float).reshape(-1)
        ys = np.asarray(payload.get('y', []), dtype=float).reshape(-1)
        if raw_indices.size == 0 or xs.size != raw_indices.size or ys.size != raw_indices.size:
            return

        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        self._eis_selected_raw_indices = raw_indices[keep].astype(int).tolist()
        self._apply_eis_selection_highlight(fig)

    def _clear_eis_selection(self):
        self._eis_selected_raw_indices = []
        fig = getattr(self, '_eis_canvas', None)
        if fig is not None:
            self._apply_eis_selection_highlight(fig)

    def _show_eis_context_menu(self, fig, event):
        key = getattr(self, 'current_file_key', None)
        entry = self.data_store.get(key) if key else None
        if entry is None:
            return
        entry = _ensure_mask_state(entry)

        raw_index = self._find_eis_raw_index_from_event(fig, event)
        menu = QtWidgets.QMenu(self)

        selected = np.asarray(getattr(self, '_eis_selected_raw_indices', []), dtype=int).reshape(-1)
        if selected.size:
            selected = selected[(selected >= 0) & (selected < entry.mask_total_raw.size)]
            if selected.size:
                any_masked = bool(np.any(entry.mask_total_raw[selected]))
                any_unmasked = bool(np.any(~entry.mask_total_raw[selected]))

                if any_unmasked:
                    act_mask_sel = QtWidgets.QAction(f'Mask selected points ({selected.size})', self)
                    act_mask_sel.triggered.connect(lambda: self._set_mask_for_raw_indices(key, selected.tolist(), True))
                    menu.addAction(act_mask_sel)
                if any_masked:
                    act_unmask_sel = QtWidgets.QAction(f'Unmask selected points ({selected.size})', self)
                    act_unmask_sel.triggered.connect(lambda: self._set_mask_for_raw_indices(key, selected.tolist(), False))
                    menu.addAction(act_unmask_sel)

                act_clear_sel = QtWidgets.QAction('Clear selection', self)
                act_clear_sel.triggered.connect(self._clear_eis_selection)
                menu.addAction(act_clear_sel)
                menu.addSeparator()

        if raw_index is not None and selected.size == 0:
            is_masked = bool(entry.mask_total_raw[int(raw_index)])
            act_toggle = QtWidgets.QAction('Unmask point' if is_masked else 'Mask point', self)
            act_toggle.triggered.connect(lambda: self._toggle_mask_point(key, raw_index, not is_masked))
            menu.addAction(act_toggle)
            menu.addSeparator()

        act_setting = QtWidgets.QAction('Mask setting', self)
        act_setting.triggered.connect(self._open_mask_setting_dialog)
        menu.addAction(act_setting)

        act_clear = QtWidgets.QAction('Clear all masks', self)
        act_clear.triggered.connect(lambda: self._clear_masks_for_key(key, clear_manual=True, clear_auto=True))
        menu.addAction(act_clear)

        try:
            menu.exec_(QtGui.QCursor.pos())
        except Exception:
            menu.exec_()

    def _on_eis_button_press(self, event):
        fig = getattr(self, '_eis_canvas', None)
        if fig is None:
            return
        if getattr(event, 'button', None) == 3:
            if getattr(event, 'inaxes', None) is fig.axes:
                self._show_eis_context_menu(fig, event)
            return
        if getattr(event, 'button', None) == 1 and getattr(event, 'inaxes', None) is fig.axes:
            raw_index = self._find_eis_raw_index_from_event(fig, event)
            if raw_index is None and len(getattr(self, '_eis_selected_raw_indices', [])) > 0:
                self._clear_eis_selection()

    def inductance_callback(self):

        keys = self._get_file_keys_in_ui_order()
        if not keys:
            if self.data is None:
                return
            keys = [getattr(self, 'current_file_key', None)] if getattr(self, 'current_file_key', None) else []
            if not keys:
                return

        errors = []
        for key in keys:
            try:
                self._apply_inductance_to_key(key, reset_method=True)
            except Exception as e:
                errors.append((key, str(e)))

        current_key = getattr(self, 'current_file_key', None)
        if current_key in getattr(self, 'data_store', {}):
            self.data = self.data_store[current_key]
        elif keys:
            first_ok = next((k for k in keys if k in getattr(self, 'data_store', {})), None)
            if first_ok is not None:
                self._set_current_file(first_ok)

        # keep current plot selection (do not force EIS)
        self.plotting_callback(self.current_plot_option)

        if errors:
            try:
                msg = '; '.join(f"{os.path.basename(k) if k else 'Unknown'}: {err}" for k, err in errors[:3])
                if len(errors) > 3:
                    msg += f"; +{len(errors) - 3} more"
                QtWidgets.QMessageBox.warning(self, 'Inductance option warning', msg)
            except Exception:
                pass
