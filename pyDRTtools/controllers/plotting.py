# -*- coding: utf-8 -*-
"""Controller for the plotting feature."""

import os
import re
from matplotlib.widgets import RectangleSelector
from PyQt5 import QtCore, QtWidgets
from ..ui.canvas import Figure_Canvas


class PlotControllerMixin:
    """Own the plotting GUI workflow."""

    def plotting_callback(self, plot_to_show):

        if plot_to_show in ('Re_residual', 'Im_residual'):
            plot_to_show = 'DRT_residual'

        # remember current plot selection so file switching keeps the same view
        self.current_plot_option = plot_to_show

        if plot_to_show == 'DRT_comparison':
            self._plot_drt_comparison()
            return

        if plot_to_show == 'DRT_map':
            self._plot_drt_map()
            return

        if plot_to_show == 'Peak_comparison':
            self._plot_peak_comparison()
            return

        # not plotting if no data has been imported
        if self.data is None:
            self._clear_plot_panel()
            return

        # initalize the figure object
        fig = Figure_Canvas()

        # rendering the figure object on to the gui
        fig_to_plot = getattr(fig, plot_to_show)
        fig_to_plot(self.data)
        self._show_canvas_in_plot_panel(fig)
        if plot_to_show == 'EIS_data':
            self._eis_canvas = fig
            try:
                fig.mpl_connect('button_press_event', self._on_eis_button_press)
            except Exception:
                pass
            try:
                self._eis_rect_selector = RectangleSelector(
                    fig.axes,
                    self._on_eis_box_select,
                    useblit=False,
                    button=[1],
                    interactive=False,
                    spancoords='data',
                    minspanx=0.0,
                    minspany=0.0,
                )
                fig._eis_rect_selector = self._eis_rect_selector
            except Exception:
                self._eis_rect_selector = None
            try:
                self._apply_eis_selection_highlight(fig)
            except Exception:
                pass
        elif plot_to_show == 'DRT_data' and list(getattr(self.data, 'peak_results', []) or []):
            try:
                fig.mpl_connect('pick_event', self._on_peak_annotation_pick)
            except Exception:
                pass
        self.ui.plot_panel.show()

    def _clear_plot_panel(self):
        """Clear the shared plot scene without leaving native Qt objects behind."""
        try:
            scene = self._plot_scene()
            self._dispose_current_canvas()
            self._current_scene = scene
            self._current_canvas = None
            self._current_proxy = None
        except Exception:
            pass

    def _plot_scene(self):
        """Return the single scene owned by the plot view for its whole lifetime."""
        view = self.ui.plot_panel
        scene = getattr(self, '_current_scene', None)
        if scene is None:
            scene = QtWidgets.QGraphicsScene(view)
            view.setScene(scene)
            self._current_scene = scene
        return scene

    def _dispose_current_canvas(self):
        """Safely detach an obsolete Qt-embedded Matplotlib canvas.

        Immediate ``QGraphicsScene.clear()`` destruction can race with queued
        paint/resize events on Windows. Detach the proxy first and defer native
        object deletion until Qt returns to its event loop.
        """
        old_canvas = getattr(self, '_current_canvas', None)
        old_proxy = getattr(self, '_current_proxy', None)
        scene = getattr(self, '_current_scene', None)
        selector = getattr(self, '_eis_rect_selector', None)
        if selector is not None:
            try:
                selector.set_active(False)
                selector.disconnect_events()
            except Exception:
                pass
        self._eis_rect_selector = None
        for name in ('_eis_canvas', '_drt_comp_canvas', '_drt_map_canvas', '_peak_comp_canvas'):
            if getattr(self, name, None) is old_canvas:
                setattr(self, name, None)
        if old_proxy is not None:
            try:
                if scene is not None:
                    scene.removeItem(old_proxy)
            except Exception:
                pass
            try:
                old_proxy.setWidget(None)
            except Exception:
                pass
            try:
                old_proxy.deleteLater()
            except Exception:
                pass
        if old_canvas is not None:
            try:
                old_canvas.setParent(None)
                old_canvas.deleteLater()
            except Exception:
                pass
        self._current_canvas = None
        self._current_proxy = None

    def _release_plot_resources(self):
        """Release Matplotlib widgets before Qt destroys the main window."""
        self._clear_plot_panel()
        self._drt_comp_lines = []
        self._drt_comp_selected = None

    def _show_canvas_in_plot_panel(self, fig):
        view = self.ui.plot_panel
        scene = self._plot_scene()
        self._dispose_current_canvas()
        proxy = scene.addWidget(fig)

        view.setAlignment(QtCore.Qt.AlignCenter)

        margin = 8
        vw = max(200, view.viewport().width() - 2 * margin)
        vh = max(200, view.viewport().height() - 2 * margin)

        dpi = fig.figure.get_dpi()
        fig.figure.set_size_inches(vw / dpi, vh / dpi, forward=True)
        fig.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        fig.resize(vw, vh)

        proxy.setPos(margin, margin)
        scene.setSceneRect(0, 0, vw + 2 * margin, vh + 2 * margin)

        self._current_canvas = fig
        self._current_scene = scene
        self._current_proxy = proxy

        try:
            fig.draw_idle()
        except Exception:
            fig.draw()

    def _refresh_current_plot_size(self):
        """Resize the live canvas without rebuilding the plot during a Qt resize event."""
        canvas = getattr(self, '_current_canvas', None)
        view = getattr(self.ui, 'plot_panel', None)
        proxy = getattr(self, '_current_proxy', None)
        if canvas is None or view is None or proxy is None:
            return
        try:
            margin = 8
            width = max(200, view.viewport().width() - 2 * margin)
            height = max(200, view.viewport().height() - 2 * margin)
            dpi = canvas.figure.get_dpi()
            canvas.figure.set_size_inches(width / dpi, height / dpi, forward=True)
            canvas.resize(width, height)
            proxy.setPos(margin, margin)
            self._plot_scene().setSceneRect(0, 0, width + 2 * margin, height + 2 * margin)
            canvas.draw_idle()
        except Exception:
            pass

    def _get_file_keys_in_ui_order(self):
        keys = []
        if hasattr(self.ui, 'files_list'):
            for i in range(self.ui.files_list.count()):
                it = self.ui.files_list.item(i)
                if it is None:
                    continue
                k = it.data(QtCore.Qt.UserRole)
                if k:
                    keys.append(k)
        if not keys:
            keys = list(getattr(self, 'data_store', {}).keys())
        return keys

    def _display_name_for_key(self, key):
        """Return the user-facing name while retaining the original source path."""
        if not key:
            return ""
        try:
            meta = (getattr(self, 'file_meta', {}).get(key, {}) or {})
            name = meta.get('display_name')
            if name is not None and str(name).strip():
                return str(name).strip()
        except Exception:
            pass
        return os.path.basename(key)

    @staticmethod
    def _hex_to_rgb(hex_color: str):
        h = (hex_color or '').strip()
        if not h.startswith('#'):
            h = '#' + h
        if not re.fullmatch(r"#([0-9a-fA-F]{6})", h):
            raise ValueError(f"Invalid hex color: {hex_color}")
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    @staticmethod
    def _rgb_to_hex(rgb):
        return '#%02X%02X%02X' % tuple(int(max(0, min(255, c))) for c in rgb)

    def _compute_gradient_colors(self, n: int, start_hex: str, end_hex: str):
        if n <= 0:
            return []
        if n == 1:
            return [start_hex]
        s = self._hex_to_rgb(start_hex)
        e = self._hex_to_rgb(end_hex)
        cols = []
        for i in range(n):
            t = i / (n - 1)
            rgb = (
                int(round(s[0] + (e[0] - s[0]) * t)),
                int(round(s[1] + (e[1] - s[1]) * t)),
                int(round(s[2] + (e[2] - s[2]) * t)),
            )
            cols.append(self._rgb_to_hex(rgb))
        return cols
