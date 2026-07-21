# -*- coding: utf-8 -*-
"""Controller for the eis export feature."""

import csv
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from ..services.eis_state import _get_entry_raw_arrays


class EISExportControllerMixin:
    """Own the eis export GUI workflow."""

    @staticmethod
    def _eis_export_safe_name(name):
        """Return a file-system-safe CSV base name without changing the UI name."""
        name = str(name or '').strip()
        if not name:
            name = 'EIS'
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, '_')
        name = name.rstrip(' .')
        return name or 'EIS'

    @staticmethod
    def _eis_export_array(values, size, dtype=float):
        """Return a fixed-length array padded with NaN values."""
        out = np.full(int(size), np.nan, dtype=dtype)
        try:
            arr = np.asarray(values, dtype=dtype).reshape(-1)
        except Exception:
            return out
        count = min(out.size, arr.size)
        if count:
            out[:count] = arr[:count]
        return out

    def _eis_export_payload(self, key):
        """Build raw-grid-aligned EIS data for one imported file.

        Raw frequency is used as the shared row grid. Masked, Lin-KK and DRT
        values are placed at their corresponding raw indices; unavailable or
        masked positions are left blank in the CSV. This preserves point-wise
        correspondence while allowing raw and masked datasets to coexist in
        one table.
        """
        entry = self.data_store.get(key)
        if entry is None:
            raise ValueError('The file is no longer available.')

        freq0, zre0, zim0, _ = _get_entry_raw_arrays(entry)
        n_raw = int(freq0.size)
        if n_raw == 0:
            raise ValueError('The file contains no EIS points.')

        active_idx = np.asarray(
            getattr(entry, 'active_raw_indices', np.arange(n_raw)),
            dtype=int,
        ).reshape(-1)
        active_idx = active_idx[(active_idx >= 0) & (active_idx < n_raw)]

        def map_active(values, dtype=float):
            result = np.full(n_raw, np.nan, dtype=dtype)
            try:
                arr = np.asarray(values, dtype=dtype).reshape(-1)
            except Exception:
                return result
            count = min(active_idx.size, arr.size)
            if count:
                result[active_idx[:count]] = arr[:count]
            return result

        masked_re = map_active(getattr(entry, 'Z_prime', []))
        masked_im = map_active(getattr(entry, 'Z_double_prime', []))

        kk_re = np.full(n_raw, np.nan, dtype=float)
        kk_im = np.full(n_raw, np.nan, dtype=float)
        kk_fit_raw = getattr(entry, 'kk_Z_fit_raw', None)
        if kk_fit_raw is not None:
            try:
                kk_complex = np.asarray(kk_fit_raw, dtype=complex).reshape(-1)
                count = min(n_raw, kk_complex.size)
                if count:
                    kk_re[:count] = kk_complex.real[:count]
                    kk_im[:count] = kk_complex.imag[:count]
            except Exception:
                pass
        elif getattr(entry, 'kk_Z_fit', None) is not None:
            try:
                kk_complex = np.asarray(entry.kk_Z_fit, dtype=complex).reshape(-1)
                count = min(active_idx.size, kk_complex.size)
                if count:
                    kk_re[active_idx[:count]] = kk_complex.real[:count]
                    kk_im[active_idx[:count]] = kk_complex.imag[:count]
            except Exception:
                pass

        drt_re = map_active(getattr(entry, 'mu_Z_re', []))
        drt_im = map_active(getattr(entry, 'mu_Z_im', []))

        drt_res_re_values = getattr(entry, 'res_re', None)
        drt_res_im_values = getattr(entry, 'res_im', None)
        if drt_res_re_values is None or drt_res_im_values is None:
            # BHT compatibility: use the fitted residual arrays available for
            # that mode instead of exporting shifted/misaligned columns.
            drt_res_re_values = getattr(entry, 'res_H_re', [])
            drt_res_im_values = getattr(entry, 'res_H_im', [])
        drt_res_re = map_active(drt_res_re_values)
        drt_res_im = map_active(drt_res_im_values)

        kk_res_re = self._eis_export_array(
            getattr(entry, 'kk_res_re_raw_pct', []), n_raw
        )
        kk_res_im = self._eis_export_array(
            getattr(entry, 'kk_res_im_raw_pct', []), n_raw
        )
        if not np.any(np.isfinite(kk_res_re)) and getattr(entry, 'kk_res_re_pct', None) is not None:
            kk_res_re = map_active(getattr(entry, 'kk_res_re_pct', []))
        if not np.any(np.isfinite(kk_res_im)) and getattr(entry, 'kk_res_im_pct', None) is not None:
            kk_res_im = map_active(getattr(entry, 'kk_res_im_pct', []))

        initial_im = np.asarray(zim0, dtype=float)
        return {
            'key': key,
            'name': self._display_name_for_key(key),
            'freq': np.asarray(freq0, dtype=float),
            'initial_re': np.asarray(zre0, dtype=float),
            'initial_neg_im': -initial_im,
            'initial_im': initial_im,
            'masked_re': masked_re,
            'masked_neg_im': -masked_im,
            'masked_im': masked_im,
            'kk_re': kk_re,
            'kk_neg_im': -kk_im,
            'kk_im': kk_im,
            'drt_re': drt_re,
            'drt_neg_im': -drt_im,
            'drt_im': drt_im,
            'kk_res_re': kk_res_re,
            'kk_res_neg_im': -kk_res_im,
            'kk_res_im': kk_res_im,
            'drt_res_re': drt_res_re,
            'drt_res_neg_im': -drt_res_im,
            'drt_res_im': drt_res_im,
        }

    @staticmethod
    def _eis_export_columns(options):
        """Return grouped CSV column definitions in the requested order."""
        groups = []
        if options['freq']:
            groups.append(('Frequency', [('Freq/Hz', 'freq')]))

        # All EIS columns are optional. Keep the display/export order:
        # Freq, Z', -Z'', Z''.
        value_components = []
        if options['z_re']:
            value_components.append(("Z'/ohm", 're'))
        if options['z_neg_im']:
            value_components.append(("-Z''/ohm", 'neg_im'))
        if options['z_im']:
            value_components.append(("Z''/ohm", 'im'))

        residual_components_ohm = []
        residual_components_pct = []
        if options['z_re']:
            residual_components_ohm.append(("Z' residual/ohm", 're'))
            residual_components_pct.append(("Z' residual/%", 're'))
        if options['z_neg_im']:
            residual_components_ohm.append(("-Z'' residual/ohm", 'neg_im'))
            residual_components_pct.append(("-Z'' residual/%", 'neg_im'))
        if options['z_im']:
            residual_components_ohm.append(("Z'' residual/ohm", 'im'))
            residual_components_pct.append(("Z'' residual/%", 'im'))

        type_map = [
            ('initial', 'Initial EIS data', 'initial'),
            ('masked', 'Masked EIS data', 'masked'),
            ('lin_kk', 'Lin-KK EIS data', 'kk'),
            ('drt', 'DRT EIS data', 'drt'),
        ]
        for option_key, title, prefix in type_map:
            if options[option_key] and value_components:
                groups.append((title, [
                    (header, f'{prefix}_{component}')
                    for header, component in value_components
                ]))

        if options['lin_kk_residual'] and residual_components_pct:
            groups.append(('Lin-KK Residual', [
                (header, f'kk_res_{component}')
                for header, component in residual_components_pct
            ]))
        if options['drt_residual'] and residual_components_ohm:
            groups.append(('DRT Residual', [
                (header, f'drt_res_{component}')
                for header, component in residual_components_ohm
            ]))
        return groups

    @staticmethod
    def _eis_export_headers(groups, include_name=False):
        first = ['File Name'] if include_name else []
        second = [''] if include_name else []
        for title, columns in groups:
            first.append(title)
            first.extend([''] * (len(columns) - 1))
            second.extend(header for header, _ in columns)
        return first, second

    @staticmethod
    def _eis_export_value(value):
        try:
            value = float(value)
        except Exception:
            return ''
        return value if np.isfinite(value) else ''

    def _write_eis_separate_csv(self, path, payload, groups):
        header_1, header_2 = self._eis_export_headers(groups, include_name=False)
        with open(path, 'w', newline='', encoding='utf-8-sig') as save_file:
            writer = csv.writer(save_file)
            writer.writerow(header_1)
            writer.writerow(header_2)
            row_count = int(payload['freq'].size)
            for row_index in range(row_count):
                row = []
                for _, columns in groups:
                    for _, field in columns:
                        values = payload.get(field, [])
                        value = values[row_index] if row_index < len(values) else np.nan
                        row.append(self._eis_export_value(value))
                writer.writerow(row)

    def _write_eis_merged_csv(self, path, payloads, groups):
        block_header_1, block_header_2 = self._eis_export_headers(groups, include_name=True)
        header_1 = []
        header_2 = []
        for payload in payloads:
            header_1.extend(block_header_1)
            second = list(block_header_2)
            second[0] = payload['name']
            header_2.extend(second)

        block_width = len(block_header_1)
        max_rows = max(int(payload['freq'].size) for payload in payloads)
        with open(path, 'w', newline='', encoding='utf-8-sig') as save_file:
            writer = csv.writer(save_file)
            writer.writerow(header_1)
            writer.writerow(header_2)
            for row_index in range(max_rows):
                merged_row = []
                for payload in payloads:
                    if row_index >= int(payload['freq'].size):
                        merged_row.extend([''] * block_width)
                        continue
                    merged_row.append(payload['name'])
                    for _, columns in groups:
                        for _, field in columns:
                            values = payload.get(field, [])
                            value = values[row_index] if row_index < len(values) else np.nan
                            merged_row.append(self._eis_export_value(value))
                writer.writerow(merged_row)

    def _show_eis_export_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Export EIS')
        dialog.setModal(True)
        dialog.setMinimumWidth(620)

        root = QtWidgets.QVBoxLayout(dialog)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(12)

        info = QtWidgets.QLabel(
            'Export files in the same order as the Files panel. '
            'Unavailable fitted values are exported as blank cells.'
        )
        info.setWordWrap(True)
        root.addWidget(info)

        def add_option_row(title, widgets):
            box = QtWidgets.QGroupBox(title, dialog)
            row = QtWidgets.QHBoxLayout(box)
            row.setContentsMargins(14, 12, 14, 12)
            row.setSpacing(18)
            for widget in widgets:
                row.addWidget(widget)
            row.addStretch(1)
            root.addWidget(box)
            return box

        cb_freq = QtWidgets.QCheckBox('Freq', dialog)
        cb_z_re = QtWidgets.QCheckBox("Z'", dialog)
        cb_z_neg_im = QtWidgets.QCheckBox("-Z''", dialog)
        cb_z_im = QtWidgets.QCheckBox("Z''", dialog)

        # All four columns can be selected independently.
        cb_freq.setChecked(True)
        cb_z_re.setChecked(True)
        cb_z_neg_im.setChecked(True)
        cb_z_im.setChecked(False)
        add_option_row(
            'EIS data columns',
            [cb_freq, cb_z_re, cb_z_neg_im, cb_z_im],
        )

        cb_initial = QtWidgets.QCheckBox('Initial data', dialog)
        cb_masked = QtWidgets.QCheckBox('Masked data', dialog)
        cb_kk = QtWidgets.QCheckBox('Lin-KK fitted data', dialog)
        cb_drt = QtWidgets.QCheckBox('DRT inversion data', dialog)
        for checkbox in (cb_initial, cb_masked, cb_kk, cb_drt):
            checkbox.setChecked(True)
        add_option_row('EIS data types', [cb_initial, cb_masked, cb_kk, cb_drt])

        cb_drt_res = QtWidgets.QCheckBox('DRT residual', dialog)
        cb_kk_res = QtWidgets.QCheckBox('Lin-KK residual', dialog)
        cb_drt_res.setChecked(True)
        cb_kk_res.setChecked(True)
        add_option_row('Residuals', [cb_drt_res, cb_kk_res])

        rb_separate = QtWidgets.QRadioButton('Separate', dialog)
        rb_merged = QtWidgets.QRadioButton('Merged', dialog)
        rb_separate.setChecked(True)
        add_option_row('Data format', [rb_separate, rb_merged])

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        export_button = buttons.addButton('Export', QtWidgets.QDialogButtonBox.AcceptRole)
        export_button.setDefault(True)
        buttons.rejected.connect(dialog.reject)
        export_button.clicked.connect(dialog.accept)
        root.addWidget(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None

        return {
            'freq': cb_freq.isChecked(),
            'z_re': cb_z_re.isChecked(),
            'z_neg_im': cb_z_neg_im.isChecked(),
            'z_im': cb_z_im.isChecked(),
            'initial': cb_initial.isChecked(),
            'masked': cb_masked.isChecked(),
            'lin_kk': cb_kk.isChecked(),
            'drt': cb_drt.isChecked(),
            'drt_residual': cb_drt_res.isChecked(),
            'lin_kk_residual': cb_kk_res.isChecked(),
            'format': 'merged' if rb_merged.isChecked() else 'separate',
        }

    def export_EIS(self):
        """Export configurable EIS datasets for every file in the Files panel."""
        keys = [key for key in self._get_file_keys_in_ui_order() if key in self.data_store]
        if not keys:
            QtWidgets.QMessageBox.information(self, 'Export EIS', 'No EIS files are available.')
            return

        options = self._show_eis_export_dialog()
        if options is None:
            return

        groups = self._eis_export_columns(options)
        if not groups:
            QtWidgets.QMessageBox.warning(
                self,
                'Export EIS',
                'Please select at least one EIS data type or residual type to export.',
            )
            return

        payloads = []
        skipped = []
        for key in keys:
            try:
                payloads.append(self._eis_export_payload(key))
            except Exception as exc:
                skipped.append(f'{self._display_name_for_key(key)}: {exc}')

        if not payloads:
            QtWidgets.QMessageBox.warning(
                self,
                'Export EIS',
                'None of the files could be exported.' + ('\n\n' + '\n'.join(skipped) if skipped else ''),
            )
            return

        try:
            if options['format'] == 'separate':
                directory = QFileDialog.getExistingDirectory(
                    self,
                    'Choose directory for separate EIS CSV files',
                    '',
                )
                if not directory:
                    return

                used_paths = set()
                for payload in payloads:
                    base_name = self._eis_export_safe_name(payload['name'])
                    candidate = os.path.join(directory, base_name + '.csv')
                    suffix = 2
                    normalized = os.path.normcase(os.path.abspath(candidate))
                    while normalized in used_paths or os.path.exists(candidate):
                        candidate = os.path.join(directory, f'{base_name}_{suffix}.csv')
                        normalized = os.path.normcase(os.path.abspath(candidate))
                        suffix += 1
                    self._write_eis_separate_csv(candidate, payload, groups)
                    used_paths.add(normalized)

                message = f'Exported {len(payloads)} EIS CSV file(s) to {directory}'
            else:
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    'Save merged EIS data',
                    'EIS_merged.csv',
                    'CSV files (*.csv)',
                )
                if not path:
                    return
                if not path.lower().endswith('.csv'):
                    path += '.csv'
                self._write_eis_merged_csv(path, payloads, groups)
                message = f'Exported merged EIS data to {path}'

            if skipped:
                message += f' | {len(skipped)} file(s) skipped'
            self.statusBar().showMessage(message, 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Export EIS failed', str(exc))
