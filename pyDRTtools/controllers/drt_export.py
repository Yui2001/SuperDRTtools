# -*- coding: utf-8 -*-
"""Controller for the drt export feature."""

import csv
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog


class DRTExportControllerMixin:
    """Own the drt export GUI workflow."""

    def export_DRT(self):  # callback for exporting the DRT results
        """Export DRT with options: selected, all separate, merged (matrix / 3-cols)."""

        # --- helpers ---
        def _is_fitted_entry(entry) -> bool:
            try:
                return entry is not None and getattr(entry, 'method', 'none') != 'none' and hasattr(entry,
                                                                                                    'out_tau_vec')
            except Exception:
                return False

        def _ui_order_keys():
            return self._get_file_keys_in_ui_order()

        def _get_gamma_for_export(entry):
            """Representative gamma curve consistent with DRT plot/comparison."""
            if entry is None:
                return None
            m = getattr(entry, 'method', 'none')
            if m in ('simple', 'peak'):
                return getattr(entry, 'gamma', None)
            if m == 'credit':
                return getattr(entry, 'gamma', None)  # MAP
            if m == 'BHT':
                return getattr(entry, 'mu_gamma_fine_re', None)
            return getattr(entry, 'gamma', None)

        def _write_single_drt(entry, path: str):
            """Write one file's DRT result (same format as legacy export_DRT)."""
            if not _is_fitted_entry(entry):
                return

            # ensure extension exists if user typed a name without one
            root, ext0 = os.path.splitext(path)
            if ext0 == '':
                path = root + '.csv'

            if entry.method == 'simple':
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.L])
                    writer.writerow(['R', entry.R])
                    writer.writerow(['tau', 'gamma'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n], entry.gamma[n]])

            elif entry.method == 'credit':
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.L])
                    writer.writerow(['R', entry.R])
                    writer.writerow(['tau', 'MAP', 'Mean', 'Upperbound', 'Lowerbound'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n], entry.gamma[n],
                                         entry.mean[n], entry.upper_bound[n], entry.lower_bound[n]])

            elif entry.method == 'BHT':
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.mu_L_0])
                    writer.writerow(['R', entry.mu_R_inf])
                    writer.writerow(['tau', 'gamma_Re', 'gamma_Im'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n],
                                         entry.mu_gamma_fine_re[n],
                                         entry.mu_gamma_fine_im[n]])

            elif entry.method == 'peak':
                # same as simple's main gamma export
                with open(path, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(['L', entry.L])
                    writer.writerow(['R', entry.R])
                    writer.writerow(['tau', 'gamma'])
                    for n in range(entry.out_tau_vec.shape[0]):
                        writer.writerow([entry.out_tau_vec[n], entry.gamma[n]])

            # peaks dataframe if present
            if hasattr(entry, 'df'):
                try:
                    df_path = os.path.splitext(path)[0] + '_peaks.csv'
                    entry.df.to_csv(df_path, index=False)
                except Exception:
                    pass

        # --- gather fitted entries ---
        keys_all = _ui_order_keys()
        fitted_keys = []
        fitted_entries = []
        for k in keys_all:
            e = self.data_store.get(k) if hasattr(self, 'data_store') else None
            if _is_fitted_entry(e):
                fitted_keys.append(k)
                fitted_entries.append(e)

        if not fitted_entries:
            return

        current_key = getattr(self, 'current_file_key', None)
        current_entry = self.data_store.get(current_key) if current_key else None
        has_selected = _is_fitted_entry(current_entry)

        # --- dialog UI ---
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export DRT")
        dlg.setModal(True)
        vbox = QtWidgets.QVBoxLayout(dlg)

        rb_selected = QtWidgets.QRadioButton("Export selected file", dlg)
        rb_all = QtWidgets.QRadioButton("Export all fitted files (separate)", dlg)
        rb_merged = QtWidgets.QRadioButton("Export merged (tau & gamma)", dlg)

        merged_type = QtWidgets.QComboBox(dlg)
        merged_type.addItems([
            "Matrix: tau, y1, y2, ...",
            "3 columns: name, tau, gamma"
        ])
        merged_row = QtWidgets.QHBoxLayout()
        merged_row.addWidget(QtWidgets.QLabel("Merged format", dlg))
        merged_row.addWidget(merged_type)

        if has_selected:
            rb_selected.setChecked(True)
        else:
            rb_all.setChecked(True)
            rb_selected.setEnabled(False)

        vbox.addWidget(rb_selected)
        vbox.addWidget(rb_all)
        vbox.addWidget(rb_merged)
        vbox.addLayout(merged_row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg
        )
        vbox.addWidget(btns)

        def _update_enabled():
            merged_type.setEnabled(rb_merged.isChecked())

        rb_selected.toggled.connect(_update_enabled)
        rb_all.toggled.connect(_update_enabled)
        rb_merged.toggled.connect(_update_enabled)
        _update_enabled()

        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # --- do export based on choice ---
        if rb_selected.isChecked():
            if not has_selected:
                return
            base = os.path.splitext(os.path.basename(current_key))[0] + "_fitted"
            ext = os.path.splitext(current_key)[1].lower()
            if ext not in ('.csv', '.txt'):
                ext = '.csv'
            default_path = os.path.join(os.path.dirname(current_key), base + ext)
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save DRT (selected file)",
                default_path,
                "CSV files (*.csv);; TXT files (*.txt)"
            )
            if not path:
                return
            _write_single_drt(current_entry, path)
            return

        if rb_all.isChecked():
            folder = QFileDialog.getExistingDirectory(self, "Select folder to export all fitted DRT files")
            if not folder:
                return
            for k, e in zip(fitted_keys, fitted_entries):
                base = os.path.splitext(os.path.basename(k))[0] + "_fitted"
                ext = os.path.splitext(k)[1].lower()
                if ext not in ('.csv', '.txt'):
                    ext = '.csv'
                out_path = os.path.join(folder, base + ext)
                try:
                    _write_single_drt(e, out_path)
                except Exception:
                    continue
            return

        if rb_merged.isChecked():
            # prepare common tau grid (use first fitted file, interpolate others if needed)
            tau0 = np.asarray(fitted_entries[0].out_tau_vec, dtype=float)
            if tau0.ndim != 1 or tau0.size < 2:
                return

            # X grid in log space for interpolation when needed
            lx0 = np.log10(tau0)

            names = [os.path.splitext(os.path.basename(k))[0] for k in fitted_keys]

            if merged_type.currentIndex() == 0:
                # Matrix: header + tau + each gamma column
                default_name = "DRT_merged_matrix.csv"
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save merged DRT (matrix)",
                    default_name,
                    "CSV files (*.csv)"
                )
                if not path:
                    return
                if not path.lower().endswith('.csv'):
                    path = path + '.csv'

                cols = []
                for e in fitted_entries:
                    g = _get_gamma_for_export(e)
                    if g is None:
                        cols.append(np.full_like(tau0, np.nan, dtype=float))
                        continue
                    g = np.asarray(g, dtype=float)
                    tau = np.asarray(e.out_tau_vec, dtype=float)
                    if tau.shape == tau0.shape and np.allclose(tau, tau0, rtol=0, atol=0):
                        cols.append(g)
                    else:
                        # interpolate onto tau0 in log-x
                        try:
                            lx = np.log10(tau)
                            order = np.argsort(lx)
                            lx = lx[order]
                            g2 = g[order]
                            cols.append(np.interp(lx0, lx, g2))
                        except Exception:
                            cols.append(np.full_like(tau0, np.nan, dtype=float))

                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['tau'] + names)
                    for i in range(tau0.size):
                        row = [tau0[i]] + [float(c[i]) for c in cols]
                        w.writerow(row)
                return

            else:
                # 3 columns: name, tau, gamma (NO log10 on tau, unlike merge_DRT_三列.py conversion)
                default_name = "DRT_merged_3cols.csv"
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save merged DRT (3 columns)",
                    default_name,
                    "CSV files (*.csv)"
                )
                if not path:
                    return
                if not path.lower().endswith('.csv'):
                    path = path + '.csv'

                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    # mimic merge_DRT_三列.py style: no header row
                    for name, e in zip(names, fitted_entries):
                        g = _get_gamma_for_export(e)
                        if g is None:
                            continue
                        tau = np.asarray(e.out_tau_vec, dtype=float)
                        g = np.asarray(g, dtype=float)
                        n = min(tau.size, g.size)
                        for i in range(n):
                            w.writerow([name, tau[i], g[i]])
                return
