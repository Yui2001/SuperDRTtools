# -*- coding: utf-8 -*-
"""Controller for the drt comparison feature."""

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..ui.canvas import Figure_Canvas


class DRTComparisonControllerMixin:
    """Own the drt comparison GUI workflow."""

    def _on_drt_comp_pick(self, event):
        # When user clicks a line, select the corresponding file in the Files list.
        # Only react to left-click picks. (On some backends/Qt embeddings, non-left
        # interactions can still generate pick events.)
        try:
            me = getattr(event, 'mouseevent', None)
            if me is not None and getattr(me, 'button', None) != 1:
                return
        except Exception:
            pass

        artist = getattr(event, 'artist', None)
        if artist is None or not hasattr(self.ui, 'files_list'):
            return
        key = None
        try:
            key = artist.get_gid()
        except Exception:
            key = None
        if not key:
            return

        for i in range(self.ui.files_list.count()):
            it = self.ui.files_list.item(i)
            if it and it.data(QtCore.Qt.UserRole) == key:
                self._suppress_file_select = True
                self.ui.files_list.setCurrentRow(i)
                break

        # highlight: selected line stays opaque; others become 80% transparent
        try:
            self._apply_drt_comp_selection(artist)
        except Exception:
            pass

    def _drt_comp_force_redraw(self):
        """Force an immediate repaint of the DRT comparison plot.

        Because the Matplotlib canvas is embedded inside a QGraphicsView via
        QGraphicsProxyWidget, canvas.draw()/draw_idle() alone may not flush to
        the screen immediately. Wheel/resize events often trigger a repaint,
        which is why alpha changes can appear only after scrolling.

        This method forces BOTH the Matplotlib canvas redraw AND the Qt
        QGraphicsView/scene repaint.
        """
        canvas = getattr(self, '_drt_comp_canvas', None)
        if canvas is None:
            return

        view = getattr(self.ui, 'plot_panel', None)
        scene = getattr(self, '_drt_comp_scene', None)
        proxy = getattr(self, '_drt_comp_proxy', None)

        def _do():
            # 1) redraw matplotlib
            try:
                canvas.draw()
            except Exception:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass

            # 2) force Qt repaint of the embedded widget + scene + viewport
            try:
                if proxy is not None:
                    proxy.update()
            except Exception:
                pass
            try:
                if scene is not None:
                    scene.invalidate(scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                    scene.update()
            except Exception:
                pass
            try:
                if view is not None:
                    view.viewport().update()
                    view.viewport().repaint()
                    view.update()
                    view.repaint()
            except Exception:
                pass

        # Queue repaint to the next Qt event loop tick (important for mpl callbacks)
        try:
            QtCore.QTimer.singleShot(0, _do)
        except Exception:
            _do()

    def _on_drt_comp_button_press(self, event):
        """DRT comparison interactions (robust in QGraphicsView embedding).

        - Right click in axes -> context menu with "Setting".
        - Left click on a curve -> select that curve, dim others (alpha=0.2), bring selected curve to top.
        - Left click on blank/outside axes -> clear selection (all opaque).
        """
        ax = getattr(event, 'inaxes', None)

        # click outside axes
        if ax is None:
            self._clear_drt_comp_selection()
            return

        btn = getattr(event, 'button', None)

        # right click -> menu
        if btn == 3:
            self._show_drt_comp_context_menu()
            return

        # only handle left click for selection
        if btn != 1:
            return

        # manual hit-test for lines (pick_event is unreliable under QGraphicsProxyWidget)
        clicked_line = None
        try:
            for ln in (self._drt_comp_lines or []):
                ok, _ = ln.contains(event)
                if ok:
                    clicked_line = ln
                    break
        except Exception:
            clicked_line = None

        if clicked_line is None:
            # blank click -> clear selection
            self._clear_drt_comp_selection()
            return

        # select corresponding file in Files list
        key = None
        try:
            key = clicked_line.get_gid()
        except Exception:
            key = None

        if key and hasattr(self.ui, 'files_list'):
            for i in range(self.ui.files_list.count()):
                it = self.ui.files_list.item(i)
                if it and it.data(QtCore.Qt.UserRole) == key:
                    self._suppress_file_select = True
                    self.ui.files_list.setCurrentRow(i)
                    break

        self._apply_drt_comp_selection(clicked_line)

    def _on_drt_comp_motion(self, event):
        """Hover tooltip on DRT comparison curves (file, tau, gamma)."""
        ax = getattr(event, 'inaxes', None)
        canvas = getattr(self, '_drt_comp_canvas', None)
        if ax is None or canvas is None:
            self._drt_comp_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        if getattr(event, 'xdata', None) is None or getattr(event, 'ydata', None) is None:
            self._drt_comp_hover_last = None
            try:
                QtWidgets.QToolTip.hideText()
            except Exception:
                pass
            return

        ex = getattr(event, 'x', None)
        ey = getattr(event, 'y', None)
        if ex is None or ey is None:
            return

        best = None  # (dist2, line, idx)

        lines_to_check = (self._drt_comp_lines or [])
        # If a curve is selected, only show hover info for the selected curve
        if getattr(self, '_drt_comp_selected', None) is not None:
            lines_to_check = [self._drt_comp_selected]

        try:
            for ln in lines_to_check:
                xdata = np.asarray(ln.get_xdata(), dtype=float)
                ydata = np.asarray(ln.get_ydata(), dtype=float)
                if xdata.size == 0:
                    continue
                pts = ax.transData.transform(np.column_stack([xdata, ydata]))
                dx = pts[:, 0] - ex
                dy = pts[:, 1] - ey
                dist2 = dx * dx + dy * dy
                j = int(np.argmin(dist2))
                d2 = float(dist2[j])
                if best is None or d2 < best[0]:
                    best = (d2, ln, j)
        except Exception:
            best = None

        if best is None:
            return

        d2, ln, idx = best
        # threshold ~ 8px
        if d2 > (8.0 * 8.0):
            if self._drt_comp_hover_last is not None:
                self._drt_comp_hover_last = None
                try:
                    QtWidgets.QToolTip.hideText()
                except Exception:
                    pass
            return

        key = None
        try:
            key = ln.get_gid()
        except Exception:
            key = None
        fname = self._display_name_for_key(key) if key else "Curve"

        try:
            tau_val = float(ln.get_xdata()[idx])
            gam_val = float(ln.get_ydata()[idx])
        except Exception:
            return

        tag = (key, idx)
        if self._drt_comp_hover_last == tag:
            return
        self._drt_comp_hover_last = tag

        text = f"File: {fname}\nτ: {tau_val:.3e} s\nγ: {gam_val:.4g} Ω"
        try:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
        except Exception:
            pass

    def _show_drt_comp_context_menu(self):
        """Show right-click context menu on DRT comparison plot."""
        menu = QtWidgets.QMenu(self)
        act = QtWidgets.QAction("Setting", self)
        menu.addAction(act)
        act.triggered.connect(self._open_drt_comparison_settings)
        try:
            menu.exec_(QtGui.QCursor.pos())
        except Exception:
            menu.exec_()

    def _apply_drt_comp_selection(self, selected_line):
        """Highlight selected DRT curve and dim all other curves."""

        if not self._drt_comp_lines:
            return

        highlight_color = "#08306B"

        for line in self._drt_comp_lines:
            try:
                if not hasattr(line, "_drt_original_color"):
                    line._drt_original_color = line.get_color()

                if not hasattr(line, "_drt_original_linewidth"):
                    line._drt_original_linewidth = line.get_linewidth()

                if line is selected_line:
                    line.set_color(highlight_color)
                    line.set_alpha(1.0)
                    line.set_linewidth(line._drt_original_linewidth)
                    line.set_zorder(10)

                else:
                    line.set_color(line._drt_original_color)
                    line.set_alpha(0.2)
                    line.set_linewidth(line._drt_original_linewidth)
                    line.set_zorder(1)

            except Exception:
                pass

        self._drt_comp_selected = selected_line
        self._drt_comp_force_redraw()

    def _clear_drt_comp_selection(self):
        """Clear DRT curve selection and restore every curve's original style."""

        if not self._drt_comp_lines:
            self._drt_comp_selected = None
            return

        for line in self._drt_comp_lines:
            try:
                original_color = getattr(
                    line,
                    "_drt_original_color",
                    line.get_color()
                )

                original_linewidth = getattr(
                    line,
                    "_drt_original_linewidth",
                    2.0
                )

                line.set_color(original_color)
                line.set_alpha(1.0)
                line.set_linewidth(original_linewidth)
                line.set_zorder(1)

            except Exception:
                pass

        self._drt_comp_selected = None
        self._drt_comp_force_redraw()

    def _open_drt_comparison_settings(self):
        """Dialog: set y-limits and gradient colors for DRT comparison."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DRT comparison settings")
        dlg.setModal(True)

        form = QtWidgets.QFormLayout(dlg)

        ymin_edit = QtWidgets.QLineEdit(dlg)
        ymax_edit = QtWidgets.QLineEdit(dlg)
        ymin_edit.setPlaceholderText("auto")
        ymax_edit.setPlaceholderText("auto")

        if self.drt_comp_settings.get('ymin') is not None:
            ymin_edit.setText(str(self.drt_comp_settings.get('ymin')))
        if self.drt_comp_settings.get('ymax') is not None:
            ymax_edit.setText(str(self.drt_comp_settings.get('ymax')))

        start_edit = QtWidgets.QLineEdit(dlg)
        end_edit = QtWidgets.QLineEdit(dlg)
        start_edit.setText(self.drt_comp_settings.get('start_color', '#08306B'))
        end_edit.setText(self.drt_comp_settings.get('end_color', '#DEEBF7'))

        def _pick_color(target_edit: QtWidgets.QLineEdit):
            try:
                current = QtGui.QColor(target_edit.text().strip())
            except Exception:
                current = QtGui.QColor("#08306B")
            col = QtWidgets.QColorDialog.getColor(current, dlg)
            if col.isValid():
                target_edit.setText(col.name().upper())

        pick_start = QtWidgets.QPushButton("Pick", dlg)
        pick_end = QtWidgets.QPushButton("Pick", dlg)
        pick_start.clicked.connect(lambda: _pick_color(start_edit))
        pick_end.clicked.connect(lambda: _pick_color(end_edit))

        start_row = QtWidgets.QHBoxLayout()
        start_row.addWidget(start_edit)
        start_row.addWidget(pick_start)

        end_row = QtWidgets.QHBoxLayout()
        end_row.addWidget(end_edit)
        end_row.addWidget(pick_end)

        form.addRow("Y min", ymin_edit)
        form.addRow("Y max", ymax_edit)
        form.addRow("Gradient start", start_row)
        form.addRow("Gradient end", end_row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dlg
        )
        form.addRow(btns)

        def _parse_float_or_none(s):
            s = (s or '').strip()
            if s == '':
                return None
            return float(s)

        def _apply():
            try:
                ymin = _parse_float_or_none(ymin_edit.text())
                ymax = _parse_float_or_none(ymax_edit.text())
                if ymin is not None and ymax is not None and ymin >= ymax:
                    raise ValueError("Y min must be smaller than Y max.")
                start_hex = start_edit.text().strip()
                end_hex = end_edit.text().strip()
                self._hex_to_rgb(start_hex)
                self._hex_to_rgb(end_hex)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Invalid settings", str(e))
                return

            self.drt_comp_settings['ymin'] = ymin
            self.drt_comp_settings['ymax'] = ymax
            self.drt_comp_settings['start_color'] = start_hex.upper()
            self.drt_comp_settings['end_color'] = end_hex.upper()
            self._mark_project_dirty()
            dlg.accept()

        btns.accepted.connect(_apply)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            if getattr(self, 'current_plot_option', None) == 'DRT_comparison':
                self.plotting_callback('DRT_comparison')

    def _plot_drt_comparison(self):
        # Build comparison plot from all fitted entries, using UI order and gradient colors.
        keys = self._get_file_keys_in_ui_order()
        if not keys or not getattr(self, 'data_store', None):
            self._clear_plot_panel()
            return

        # collect entries that have DRT data
        entries = []
        entry_keys = []
        for k in keys:
            entry = self.data_store.get(k)
            if entry is None:
                continue
            if getattr(entry, 'method', 'none') == 'none':
                continue
            if not hasattr(entry, 'out_tau_vec'):
                continue
            entries.append(entry)
            entry_keys.append(k)

        if not entries:
            self._clear_plot_panel()
            return

        cols = self._compute_gradient_colors(
            len(entries),
            self.drt_comp_settings.get('start_color', '#08306B'),
            self.drt_comp_settings.get('end_color', '#DEEBF7'),
        )

        y_lim = None
        if self.drt_comp_settings.get('ymin') is not None or self.drt_comp_settings.get('ymax') is not None:
            y_lim = (self.drt_comp_settings.get('ymin'), self.drt_comp_settings.get('ymax'))

        fig = Figure_Canvas()
        fig.DRT_comparison(entries, cols, y_lim=y_lim, keys=entry_keys)

        # store lines for highlighting/dimming
        self._drt_comp_canvas = fig
        self._drt_comp_lines = list(getattr(fig, '_drt_comp_lines', []) or [])
        self._drt_comp_selected = None

        # interactivity
        try:
            fig.mpl_connect('button_press_event', self._on_drt_comp_button_press)
            fig.mpl_connect('motion_notify_event', self._on_drt_comp_motion)
        except Exception:
            pass

        self._show_canvas_in_plot_panel(fig)
        self._drt_comp_scene = self._current_scene
        self._drt_comp_proxy = self._current_proxy
        self.ui.plot_panel.show()
        self._select_drt_comparison_line_by_key(self.current_file_key)
