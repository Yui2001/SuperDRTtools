# -*- coding: utf-8 -*-
"""Controller for the drt map feature."""

import matplotlib as mpl
import numpy as np
from PyQt5 import QtGui, QtWidgets
from ..ui.canvas import Figure_Canvas


class DRTMapControllerMixin:
    """Own the drt map GUI workflow."""

    def _show_drt_map_context_menu(self):
        menu = QtWidgets.QMenu(self)
        act = QtWidgets.QAction("Setting", self)
        menu.addAction(act)
        act.triggered.connect(self._open_drt_map_settings)
        try:
            menu.exec_(QtGui.QCursor.pos())
        except Exception:
            menu.exec_()

    def _open_drt_map_settings(self):
        """Configure DRT Map color scaling and Origin-style level settings."""
        # Determine current auto min/max if needed (based on current Z matrix)
        auto_min = None
        auto_max = None
        try:
            z = getattr(self, '_drt_map_Z', None)
            if z is not None and hasattr(z, 'size') and z.size:
                auto_min = float(np.nanmin(z))
                auto_max = float(np.nanmax(z))
        except Exception:
            pass

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DRT Map settings")
        dlg.setModal(True)
        form = QtWidgets.QFormLayout(dlg)

        from_edit = QtWidgets.QLineEdit(dlg)
        to_edit = QtWidgets.QLineEdit(dlg)
        major_edit = QtWidgets.QLineEdit(dlg)
        minor_edit = QtWidgets.QLineEdit(dlg)

        # placeholders / defaults
        vmin = self.drt_map_settings.get('vmin')
        vmax = self.drt_map_settings.get('vmax')
        if vmin is None and auto_min is not None:
            vmin = auto_min
        if vmax is None and auto_max is not None:
            vmax = auto_max

        if vmin is not None:
            from_edit.setText(str(vmin))
        if vmax is not None:
            to_edit.setText(str(vmax))

        major_edit.setText(str(self.drt_map_settings.get('major_levels', 10)))
        minor_edit.setText(str(self.drt_map_settings.get('minor_levels', 10)))

        fill_choice = QtWidgets.QComboBox(dlg)
        fill_choice.addItems(["Fill to Contour Lines", "Fill to Grid Lines"])
        fill_mode = (self.drt_map_settings.get('fill_mode') or 'contour').lower()
        fill_choice.setCurrentIndex(0 if fill_mode == 'contour' else 1)

        form.addRow("From", from_edit)
        form.addRow("To", to_edit)
        form.addRow("Major levels", major_edit)
        form.addRow("Minor levels", minor_edit)
        form.addRow("Fill mode", fill_choice)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dlg
        )
        form.addRow(btns)

        def _apply():
            try:
                vmin2 = float(from_edit.text().strip())
                vmax2 = float(to_edit.text().strip())
                if vmin2 >= vmax2:
                    raise ValueError("From must be smaller than To.")

                maj = int(float(major_edit.text().strip()))
                minu = int(float(minor_edit.text().strip()))
                if maj < 2:
                    raise ValueError("Major levels must be >= 2.")
                if minu < 2:
                    raise ValueError("Minor levels must be >= 2.")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Invalid settings", str(e))
                return

            self.drt_map_settings['vmin'] = vmin2
            self.drt_map_settings['vmax'] = vmax2
            self.drt_map_settings['major_levels'] = maj
            self.drt_map_settings['minor_levels'] = minu
            self.drt_map_settings['fill_mode'] = 'contour' if fill_choice.currentIndex() == 0 else 'grid'
            self._mark_project_dirty()
            dlg.accept()

        btns.accepted.connect(_apply)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            if getattr(self, 'current_plot_option', None) == 'DRT_map':
                self.plotting_callback('DRT_map')

    @staticmethod
    def _logspace_edges(x):
        """Compute pcolormesh-friendly bin edges for a strictly positive x grid."""
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            return np.array([x[0] * 0.9, x[0] * 1.1], dtype=float)
        edges = np.empty(x.size + 1, dtype=float)
        # internal edges at geometric mean
        edges[1:-1] = np.sqrt(x[:-1] * x[1:])
        # extrapolate first/last
        edges[0] = x[0] * (x[0] / edges[1])
        edges[-1] = x[-1] * (x[-1] / edges[-2])
        return edges

    def _on_drt_map_button_press(self, event):
        # right-click -> settings menu
        if getattr(event, 'button', None) == 3:
            self._show_drt_map_context_menu()
            return

    def _on_drt_map_motion(self, event):
        # show tooltip for hovered point (only on the heatmap axes, not on the colorbar)
        ax = getattr(event, 'inaxes', None)
        main_ax = getattr(self, '_drt_map_ax', None)
        if ax is None or main_ax is None or ax is not main_ax:
            self._drt_map_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        x = getattr(event, 'xdata', None)
        y = getattr(event, 'ydata', None)
        if x is None or y is None:
            self._drt_map_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        xgrid = getattr(self, '_drt_map_x', None)
        zmat = getattr(self, '_drt_map_Z', None)
        keys = getattr(self, '_drt_map_keys', None)
        if xgrid is None or zmat is None or keys is None:
            return

        n_files = int(zmat.shape[0])
        # y is 0..n_files (we invert y-axis), so row is int(y) clipped
        row = int(np.clip(int(y), 0, n_files - 1))

        # nearest x index
        try:
            xi = int(np.searchsorted(xgrid, x))
            if xi <= 0:
                col = 0
            elif xi >= len(xgrid):
                col = len(xgrid) - 1
            else:
                col = xi - 1 if abs(xgrid[xi - 1] - x) <= abs(xgrid[xi] - x) else xi
        except Exception:
            col = 0

        last = getattr(self, '_drt_map_hover_last', None)
        if last == (row, col):
            return
        self._drt_map_hover_last = (row, col)

        try:
            val = float(zmat[row, col])
        except Exception:
            return

        key = keys[row] if row < len(keys) else None
        fname = self._display_name_for_key(key) if key else f"Row {row + 1}"

        try:
            tau_val = float(xgrid[col])
        except Exception:
            tau_val = float(x)

        text = f"File: {fname}\nτ: {tau_val:.3e} s\nγ: {val:.4g} Ω"
        try:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
        except Exception:
            pass

    def _plot_drt_map(self):
        # Build 2D heatmap from all fitted entries, ordered by Files list (top->bottom).
        keys = self._get_file_keys_in_ui_order()
        if not keys or not getattr(self, 'data_store', None):
            self._clear_plot_panel()
            return

        curves = []
        entry_keys = []

        for k in keys:
            entry = self.data_store.get(k)
            if entry is None:
                continue
            if getattr(entry, 'method', 'none') == 'none':
                continue
            if not hasattr(entry, 'out_tau_vec'):
                continue

            x = np.asarray(entry.out_tau_vec, dtype=float)
            y = None
            if entry.method in ('simple', 'peak'):
                y = getattr(entry, 'gamma', None)
            elif entry.method == 'credit':
                y = getattr(entry, 'gamma', None)  # MAP
            elif entry.method == 'BHT':
                y = getattr(entry, 'mu_gamma_fine_re', None)
            else:
                y = getattr(entry, 'gamma', None)

            if y is None:
                continue
            y = np.asarray(y, dtype=float)
            if x.size != y.size:
                continue

            curves.append((x, y))
            entry_keys.append(k)

        if not curves:
            self._clear_plot_panel()
            return

        # ---- Common X grid (log-x) ----
        xs = [c[0] for c in curves]
        same_grid = True
        for a in xs[1:]:
            if a.shape != xs[0].shape or not np.allclose(a, xs[0], rtol=0, atol=0):
                same_grid = False
                break

        if same_grid:
            xgrid = xs[0]
        else:
            xmin = min(float(np.min(x)) for x in xs)
            xmax = max(float(np.max(x)) for x in xs)
            npts = max(len(x) for x in xs)
            xgrid = np.logspace(np.log10(xmin), np.log10(xmax), npts)

        # ---- Interpolate all curves onto xgrid (log-x) ----
        zmat = np.zeros((len(curves), len(xgrid)), dtype=float)
        lxg = np.log10(xgrid)
        for i, (x, y) in enumerate(curves):
            lx = np.log10(x)
            if np.any(np.diff(lx) <= 0):
                order = np.argsort(lx)
                lx = lx[order]
                y2 = y[order]
            else:
                y2 = y
            zmat[i, :] = np.interp(lxg, lx, y2)

        # ---- Settings: From/To and Origin-style levels ----
        vmin = self.drt_map_settings.get('vmin')
        vmax = self.drt_map_settings.get('vmax')
        if vmin is None:
            vmin = float(np.nanmin(zmat))
        if vmax is None:
            vmax = float(np.nanmax(zmat))
        if vmin == vmax:
            vmax = vmin + 1e-12

        # Clamp both sides for plotting: values below vmin use the exact
        # color assigned to vmin, while values above vmax use the exact
        # color assigned to vmax. This prevents contourf from leaving
        # out-of-range regions white.
        zmat_for_plot = np.clip(zmat, vmin, vmax)

        major_levels = int(self.drt_map_settings.get('major_levels', 10) or 10)
        minor_levels = int(self.drt_map_settings.get('minor_levels', 10) or 10)
        total_levels = max(2, major_levels * minor_levels)

        fill_mode = (self.drt_map_settings.get('fill_mode') or 'contour').lower()
        # Origin-like warming palette (blue -> white -> red)
        cmap = mpl.cm.get_cmap('RdBu_r', total_levels)

        fig = Figure_Canvas()
        ax = fig.axes

        # ---- Layout: heatmap big; vertical short colorbar on the left ----
        heat_top = 0.90
        heat_bottom = 0.12

        cbar_left = -0.005
        cbar_w = 0.028

        gap_x = 0.030
        heat_left = 0.05 + cbar_w + gap_x
        heat_right = 0.92

        ax.set_position([heat_left, heat_bottom, heat_right - heat_left, heat_top - heat_bottom])

        # short vertical colorbar: about 1/4 of heatmap height
        heat_h = (heat_top - heat_bottom)
        cbar_h = heat_h * 0.25
        cbar_bottom = heat_bottom
        cax = fig.figure.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])

        n_files = zmat.shape[0]

        # ---- Plot heatmap ----
        if fill_mode == 'grid':
            x_edges = self._logspace_edges(xgrid)
            y_edges = np.linspace(0, n_files, n_files + 1)
            m = ax.pcolormesh(
                x_edges, y_edges, zmat_for_plot,
                cmap=cmap,
                shading='auto',
                vmin=vmin,
                vmax=vmax
            )
            ax.invert_yaxis()  # first file at top
        else:
            # Fill to Contour Lines (smooth, no blank band):
            # use y-edges (0..n_files) and a z-matrix with (n_files+1) rows
            y_edges = np.arange(n_files + 1, dtype=float)

            if n_files == 1:
                z_edges = np.vstack([zmat_for_plot[0], zmat_for_plot[0]])
            else:
                z_edges = np.empty((n_files + 1, zmat_for_plot.shape[1]), dtype=float)
                z_edges[0, :] = zmat_for_plot[0, :]
                z_edges[-1, :] = zmat_for_plot[-1, :]
                z_edges[1:-1, :] = 0.5 * (zmat_for_plot[:-1, :] + zmat_for_plot[1:, :])

            levels = np.linspace(vmin, vmax, total_levels)
            m = ax.contourf(
                xgrid, y_edges, z_edges,
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            ax.set_ylim(n_files, 0)  # first file at top

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau/s$')
        ax.set_ylabel('Files')
        ax.set_xlim([float(np.min(xgrid)), float(np.max(xgrid))])
        # 每个文件之间一个刻度（边界线）
        ax.set_yticks(range(0, n_files + 1))
        # 隐藏 y 轴数字，但保留刻度线
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='major', left=True, labelleft=False)

        # ---- Colorbar: only 3 ticks (min/mid/max), no extra configuration ----
        cbar = fig.figure.colorbar(m, cax=cax, orientation='vertical')
        try:
            ticks = [vmin, (vmin + vmax) / 2.0, vmax]
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=8)
            cbar.outline.set_linewidth(0.6)
            cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        except Exception:
            pass

        try:
            ticks = [vmin, (vmin + vmax) / 2.0, vmax]
            cbar.set_ticks(ticks)
            cbar.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        except Exception:
            pass

        # store for hover + axis filtering
        self._drt_map_canvas = fig
        self._drt_map_ax = ax
        self._drt_map_cbar_ax = cbar.ax
        self._drt_map_x = xgrid
        self._drt_map_Z = zmat
        self._drt_map_keys = entry_keys
        self._drt_map_hover_last = None

        try:
            fig.mpl_connect('button_press_event', self._on_drt_map_button_press)
        except Exception:
            pass
        try:
            fig.mpl_connect('motion_notify_event', self._on_drt_map_motion)
        except Exception:
            pass

        self._show_canvas_in_plot_panel(fig)
        self._drt_map_scene = self._current_scene
        self._drt_map_proxy = self._current_proxy
        self.ui.plot_panel.show()
