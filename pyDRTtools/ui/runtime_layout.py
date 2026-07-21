# -*- coding: utf-8 -*-
"""Responsive sidebar, plot-panel, and top-navigation layout helpers."""

from PyQt5 import QtCore, QtWidgets


_NOISE_TEXTS = ("Options for RBF", "Options for kBL", "Options for KBL")

# ---- Tuning knobs (you can change these numbers manually) ----
TOPBAR_GUTTER = 0  # kept for compatibility; top bar now uses layouts
PLOT_SHIFT_DX = 0  # kept for compatibility; plot area now uses layouts


def _set_label_style_clean(widget: QtWidgets.QWidget) -> None:
    for lab in widget.findChildren(QtWidgets.QLabel):
        try:
            lab.setStyleSheet("background: transparent;")
        except Exception:
            pass


def _text_contains_noise(t: str) -> bool:
    if not t:
        return False
    ts = t.strip()
    for n in _NOISE_TEXTS:
        if n in ts:
            return True
    return False


def _hide_noise_global(root: QtWidgets.QWidget, allow_groupboxes=None) -> None:
    """Hide ghost labels AND ghost groupboxes (titles) from old absolute layout."""
    allow_groupboxes = allow_groupboxes or set()

    for lab in root.findChildren(QtWidgets.QLabel):
        try:
            if _text_contains_noise(lab.text()):
                lab.hide()
        except Exception:
            pass

    for gb in root.findChildren(QtWidgets.QGroupBox):
        try:
            if gb in allow_groupboxes:
                continue
            if _text_contains_noise(gb.title()):
                gb.hide()
        except Exception:
            pass


def _hide_top_bar_frames(root: QtWidgets.QWidget) -> None:
    try:
        for fr in root.findChildren(QtWidgets.QFrame):
            if fr.frameShape() in (QtWidgets.QFrame.HLine, QtWidgets.QFrame.VLine, QtWidgets.QFrame.Box,
                                   QtWidgets.QFrame.Panel):
                fr.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass


def _clear_layout_widget(parent: QtWidgets.QWidget) -> None:
    lay = parent.layout()
    if lay is None:
        return
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(parent)
    QtWidgets.QWidget().setLayout(lay)


def _hide_unmanaged_children(container: QtWidgets.QWidget, keep: set) -> None:
    for ch in container.findChildren((QtWidgets.QLabel, QtWidgets.QFrame, QtWidgets.QGroupBox)):
        if ch in keep:
            continue
        try:
            if isinstance(ch, QtWidgets.QGroupBox) and _text_contains_noise(ch.title()):
                ch.hide()
            elif isinstance(ch, QtWidgets.QLabel) and _text_contains_noise(ch.text()):
                ch.hide()
            elif isinstance(ch, QtWidgets.QFrame):
                ch.hide()
            else:
                ch.hide()
        except Exception:
            pass


def _ensure_field_width(widget: QtWidgets.QWidget, min_w: int = 120) -> None:
    try:
        widget.setMinimumWidth(min_w)
        widget.setMaximumWidth(16777215)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    except Exception:
        pass


def _fix_top_bar_position(self) -> None:
    """Keep the top tab bar expanding naturally inside the center layout."""
    bar = getattr(self.ui, "show_layout", None)
    if bar is None:
        return
    try:
        bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        bar.setMinimumHeight(58)
        bar.setMaximumHeight(72)
        bar.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass
    try:
        tabs = bar.findChild(QtWidgets.QTabWidget, "TopTabs")
        if tabs is not None:
            tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            tabs.updateGeometry()
    except Exception:
        pass


def _shift_plot_panel_right(self, dx: int = None) -> None:
    """Add a gutter between sidebar and plot area. Uses plot_panel geometry if available."""
    if dx is None:
        dx = PLOT_SHIFT_DX
    plot = getattr(self.ui, "plot_panel", None)
    if plot is None:
        return
    try:
        # If plot is layout-managed, moving may be ignored; we also add an internal left margin.
        plot.move(plot.x() + dx, plot.y())
        plot.resize(max(200, plot.width() - dx), plot.height())
    except Exception:
        pass
    try:
        lay = plot.layout()
        if lay is not None:
            m = lay.contentsMargins()
            lay.setContentsMargins(m.left() + dx, m.top(), m.right(), m.bottom())
    except Exception:
        pass
    try:
        plot.setFrameShape(QtWidgets.QFrame.NoFrame)
    except Exception:
        pass


def _force_combobox_popup(self) -> None:
    """Force QComboBox popup to use QListView with a safe palette so items are visible on all themes."""
    try:
        from PyQt5 import QtGui as _QtGui
    except Exception:
        _QtGui = None

    combos = self.findChildren(QtWidgets.QComboBox)
    for cb in combos:
        try:
            view = QtWidgets.QListView()
            view.setUniformItemSizes(True)
            view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            # Palette: ensure text is dark on white
            if _QtGui is not None:
                pal = view.palette()
                pal.setColor(_QtGui.QPalette.Base, _QtGui.QColor(255, 255, 255))
                pal.setColor(_QtGui.QPalette.Text, _QtGui.QColor(29, 29, 31))
                pal.setColor(_QtGui.QPalette.WindowText, _QtGui.QColor(29, 29, 31))
                pal.setColor(_QtGui.QPalette.Highlight, _QtGui.QColor(10, 132, 255))
                pal.setColor(_QtGui.QPalette.HighlightedText, _QtGui.QColor(255, 255, 255))
                view.setPalette(pal)

            view.setStyleSheet(
                "QListView{background:#FFFFFF;color:#1D1D1F;border:0px solid #E5E5EA;border-radius:10px;}"
                "QListView::item{color:#1D1D1F;padding:6px 10px;min-height:28px;}"
                "QListView::item:selected{background:#0A84FF;color:#FFFFFF;border-radius: 8px;}"
            )
            cb.setView(view)

            # Ensure the combobox itself uses a readable palette
            if _QtGui is not None:
                pal2 = cb.palette()
                pal2.setColor(_QtGui.QPalette.ButtonText, _QtGui.QColor(29, 29, 31))
                pal2.setColor(_QtGui.QPalette.Text, _QtGui.QColor(29, 29, 31))
                cb.setPalette(pal2)
        except Exception:
            pass


def _install_resize_hook(self) -> None:
    if getattr(self, "_safe_refactor_resize_hook_installed", False):
        return
    self._safe_refactor_resize_hook_installed = True

    old_resize = getattr(self, "resizeEvent", None)

    self._plot_resize_timer = QtCore.QTimer(self)
    self._plot_resize_timer.setSingleShot(True)
    self._plot_resize_timer.timeout.connect(self._refresh_current_plot_size)

    def _new_resize_event(evt):
        try:
            if callable(old_resize):
                old_resize(evt)
        finally:
            try:
                _fix_top_bar_position(self)
            except Exception:
                pass
            try:
                self._plot_resize_timer.start(80)
            except Exception:
                pass

    try:
        self.resizeEvent = _new_resize_event
    except Exception:
        pass


def _refactor_ui_layout(self) -> None:
    try:
        _set_label_style_clean(self)
        allow = set()
        for name in ("project_frame", "RBF_frame", "settings_layout", "run_layout", "Peak_analysis_frame", "export_frame"):
            gb = getattr(self.ui, name, None)
            if isinstance(gb, QtWidgets.QGroupBox):
                allow.add(gb)

        _hide_noise_global(self, allow_groupboxes=allow)
        _hide_top_bar_frames(self)

        _rebuild_sidebar_with_forms(self)

        _hide_noise_global(self, allow_groupboxes=allow)

        _replace_show_buttons_with_tabs(self)
        _fix_top_bar_position(self)
        _install_resize_hook(self)

        # Ensure ComboBox popups render text (fix blank dropdown list)
        _force_combobox_popup(self)

        _hide_noise_global(self, allow_groupboxes=allow)
    except Exception:
        pass
    # 在 _refactor_ui_layout(self) 的最后（比如 _fix_top_bar_position 之后）加：
    try:
        self.ui.plot_panel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ui.plot_panel.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass


def _rebuild_sidebar_with_forms(self) -> None:
    host = getattr(self.ui, "General_frame", None)
    if host is None:
        return

    _clear_layout_widget(host)

    scroll = QtWidgets.QScrollArea(host)
    scroll.setObjectName("SidebarScroll")
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    host_layout = QtWidgets.QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.setSpacing(0)
    host_layout.addWidget(scroll, 1)

    container = QtWidgets.QWidget()
    container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
    vbox = QtWidgets.QVBoxLayout(container)
    vbox.setContentsMargins(12, 12, 12, 12)
    vbox.setSpacing(12)

    groups = [
        getattr(self.ui, "project_frame", None),
        getattr(self.ui, "settings_layout", None),
        getattr(self.ui, "RBF_frame", None),
        getattr(self.ui, "run_layout", None),
        getattr(self.ui, "KK_frame", None),
        getattr(self.ui, "Peak_analysis_frame", None),
        getattr(self.ui, "export_frame", None),
    ]
    groups = [g for g in groups if g is not None]

    for gb in groups:
        gb.setParent(container)
        gb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        vbox.addWidget(gb)
    vbox.addStretch(1)

    scroll.setWidget(container)

    host.setMinimumWidth(400)
    host.setMaximumWidth(600)
    host.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

    if hasattr(self.ui, "project_frame"):
        _layout_project_group(self)
    if hasattr(self.ui, "settings_layout"):
        _layout_settings_group(self)
    if hasattr(self.ui, "RBF_frame"):
        _layout_rbf_group(self)
    if hasattr(self.ui, "run_layout"):
        _layout_run_group(self)
    if hasattr(self.ui, "KK_frame"):
        _layout_kk_group(self)
    if hasattr(self.ui, "Peak_analysis_frame"):
        _layout_peak_group(self)
    if hasattr(self.ui, "export_frame"):
        _layout_export_group(self)

    class NoWheelFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Wheel:
                event.ignore()
                return True
            return super().eventFilter(obj, event)

    self._no_wheel_filter = NoWheelFilter(self)

    for cb in self.findChildren(QtWidgets.QComboBox):
        cb.installEventFilter(self._no_wheel_filter)


def _layout_settings_group(self) -> None:
    gb = self.ui.settings_layout
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)

    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    labels = [
        self.ui.import_label,
        self.ui.discre_label,
        self.ui.data_used_label,
        self.ui.induct_label,
        self.ui.der_label,
        self.ui.lambda_choice_label,
        self.ui.reg_param_label,
        self.ui.reg_param_label_2,
        self.ui.sample_no,
    ]

    for lab in labels:
        lab.setMinimumWidth(120)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        lab.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

    fields = [
        self.ui.discre_choice,
        self.ui.data_used_choice,
        self.ui.induct_choice,
        self.ui.der_choice,
        self.ui.lambda_choice,
        self.ui.reg_param_entry,
        self.ui.reg_param_entry_2,
        self.ui.sample_no_entry,
    ]
    for w in fields:
        w.setMinimumHeight(30)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.import_button.setFixedHeight(30)
    self.ui.import_button.setFixedWidth(120)
    self.ui.import_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    keep = set(labels)
    _hide_unmanaged_children(gb, keep)

    row = 0
    import_row = QtWidgets.QHBoxLayout()
    import_row.addStretch(1)
    import_row.addWidget(self.ui.import_button)
    grid.addWidget(self.ui.import_label, 0, 0)
    grid.addLayout(import_row, 0, 1)
    row += 1

    grid.addWidget(self.ui.discre_label, row, 0)
    grid.addWidget(self.ui.discre_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.data_used_label, row, 0)
    grid.addWidget(self.ui.data_used_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.induct_label, row, 0)
    grid.addWidget(self.ui.induct_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.der_label, row, 0)
    grid.addWidget(self.ui.der_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.lambda_choice_label, row, 0)
    grid.addWidget(self.ui.lambda_choice, row, 1)
    row += 1

    grid.addWidget(self.ui.reg_param_label, row, 0)
    grid.addWidget(self.ui.reg_param_entry, row, 1)
    row += 1

    grid.addWidget(self.ui.reg_param_label_2, row, 0)
    grid.addWidget(self.ui.reg_param_entry_2, row, 1)
    row += 1

    grid.addWidget(self.ui.sample_no, row, 0)
    grid.addWidget(self.ui.sample_no_entry, row, 1)


def _layout_rbf_group(self) -> None:
    gb = self.ui.RBF_frame
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    for lab in [self.ui.shape_control_label, self.ui.FWHM_control_label]:
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    for w in [self.ui.shape_control_choice, self.ui.FWHM_entry]:
        w.setMinimumHeight(30)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    keep = {self.ui.shape_control_label, self.ui.FWHM_control_label}
    _hide_unmanaged_children(gb, keep)

    grid.addWidget(self.ui.shape_control_label, 0, 0)
    grid.addWidget(self.ui.shape_control_choice, 0, 1)
    grid.addWidget(self.ui.FWHM_control_label, 1, 0)
    grid.addWidget(self.ui.FWHM_entry, 1, 1)


def _layout_kk_group(self) -> None:
    gb = self.ui.KK_frame
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    for lab in [
        self.ui.cutoff_label,
        self.ui.max_elements_label,
        self.ui.fit_type_label,
        self.ui.analyze_kkr_label,
    ]:
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    for w in [self.ui.cutoff_entry, self.ui.max_elements_entry, self.ui.fit_type_choice]:
        w.setMinimumHeight(30)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.run_kkr_button.setFixedHeight(30)
    self.ui.run_kkr_button.setFixedWidth(110)
    self.ui.run_kkr_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    keep = {
        self.ui.cutoff_label,
        self.ui.max_elements_label,
        self.ui.fit_type_label,
        self.ui.analyze_kkr_label,
    }
    _hide_unmanaged_children(gb, keep)

    grid.addWidget(self.ui.cutoff_label, 0, 0)
    grid.addWidget(self.ui.cutoff_entry, 0, 1)

    grid.addWidget(self.ui.max_elements_label, 1, 0)
    grid.addWidget(self.ui.max_elements_entry, 1, 1)

    grid.addWidget(self.ui.fit_type_label, 2, 0)
    grid.addWidget(self.ui.fit_type_choice, 2, 1)

    grid.addWidget(self.ui.analyze_kkr_label, 3, 0)

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addStretch(1)
    btn_row.addWidget(self.ui.run_kkr_button)
    grid.addWidget(self.ui.analyze_kkr_label, 3, 0)
    grid.addLayout(btn_row, 3, 1)


def _layout_run_group(self) -> None:
    gb = self.ui.run_layout
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 0)

    rows = [
        (self.ui.simple_run_label, self.ui.simple_run_button),
        (self.ui.bayes_label, self.ui.bayesian_button),
        (self.ui.HT_label, self.ui.HT_button),
    ]
    keep = {r[0] for r in rows}
    _hide_unmanaged_children(gb, keep)

    for r, (lab, btn) in enumerate(rows):
        btn.setMinimumHeight(30)
        btn.setMinimumWidth(110)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        grid.addWidget(lab, r, 0)
        grid.addWidget(btn, r, 1, alignment=QtCore.Qt.AlignRight)

    # Fit buttons row (below Hilbert Transform). Keep alignment with Select buttons.
    if hasattr(self.ui, "fit_one_button") and hasattr(self.ui, "fit_all_button"):
        self.ui.fit_one_button.setMinimumHeight(30)
        self.ui.fit_one_button.setMinimumWidth(110)
        self.ui.fit_one_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.ui.fit_all_button.setMinimumHeight(30)
        self.ui.fit_all_button.setMinimumWidth(110)
        self.ui.fit_all_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        # row index = len(rows)
        grid.addWidget(self.ui.fit_one_button, len(rows), 0, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(self.ui.fit_all_button, len(rows), 1, alignment=QtCore.Qt.AlignRight)


def _layout_peak_group(self) -> None:
    gb = self.ui.Peak_analysis_frame
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(10)

    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    for lab in [self.ui.peak_method_label, self.ui.reg_param_2]:
        lab.setMinimumWidth(150)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        lab.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

    self.ui.peak_method_choice.setMinimumHeight(30)
    self.ui.peak_method_choice.setMinimumWidth(110)
    self.ui.peak_method_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.peak_num_entry.hide()
    self.ui.Peak_decon_run.hide()
    self.ui.peak_count_choice.setMinimumHeight(30)
    self.ui.peak_count_choice.setMinimumWidth(110)
    self.ui.peak_count_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    self.ui.peak_decon_button.setFixedHeight(30)
    self.ui.peak_decon_button.setFixedWidth(110)
    self.ui.peak_decon_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    self.ui.peak_all_button.setFixedHeight(30)
    self.ui.peak_all_button.setFixedWidth(110)
    self.ui.peak_all_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    keep = {self.ui.peak_method_label, self.ui.reg_param_2}
    _hide_unmanaged_children(gb, keep)

    grid.addWidget(self.ui.peak_method_label, 0, 0)
    grid.addWidget(self.ui.peak_method_choice, 0, 1)

    grid.addWidget(self.ui.reg_param_2, 1, 0)
    grid.addWidget(self.ui.peak_count_choice, 1, 1)

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addWidget(self.ui.peak_decon_button)
    btn_row.addStretch(1)
    btn_row.addWidget(self.ui.peak_all_button)

    grid.addLayout(btn_row, 2, 0, 1, 2)


def _card_button_style():
    return """
    QPushButton {
        background: #FFFFFF;
        border: 1px solid #D2D2D7;
        border-radius: 10px;
        padding: 0px 12px;
        min-height: 0px;
        max-height: 28px;
        height: 28px;
    }
    QPushButton:hover {
        background: #F2F2F7;
        border-color: #C7C7CC;
    }
    QPushButton:pressed {
        background: #EAEAEE;
        border-color: #BDBDC2;
    }
    """


def _layout_two_column_card(gb, rows):
    _set_label_style_clean(gb)
    _clear_layout_widget(gb)

    grid = QtWidgets.QGridLayout(gb)
    grid.setContentsMargins(16, 10, 16, 14)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(8)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 0)

    keep = {lab for lab, _ in rows}
    _hide_unmanaged_children(gb, keep)

    button_w = 110
    button_h = 28
    button_style = _card_button_style()
    for row, (label, button) in enumerate(rows):
        label.show()
        button.show()
        label.setStyleSheet('background: transparent;')
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        label.setFixedHeight(button_h)
        label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        button.setFixedSize(button_w, button_h)
        button.setMinimumSize(button_w, button_h)
        button.setMaximumSize(button_w, button_h)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        button.setStyleSheet(button_style)

        grid.setRowMinimumHeight(row, button_h)
        grid.addWidget(label, row, 0)
        grid.addWidget(button, row, 1, alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)


def _layout_project_group(self) -> None:
    gb = self.ui.project_frame
    rows = [
        (self.ui.open_project_label, self.ui.open_project_button),
        (self.ui.save_project_label, self.ui.save_project_button),
        (self.ui.new_window_label, self.ui.new_window_button),
    ]
    for name in ('save_project_as_label', 'save_project_as_button'):
        widget = getattr(self.ui, name, None)
        if widget is not None:
            widget.hide()
    self.ui.save_project_button.setToolTip('Right-click to choose Save Project As...')
    _layout_two_column_card(gb, rows)


def _layout_export_group(self) -> None:
    gb = self.ui.export_frame
    self.ui.export_EIS_label.setText('EIS')
    rows = [
        (self.ui.export_DRT_label, self.ui.export_DRT_button),
        (self.ui.export_EIS_label, self.ui.export_EIS_button),
        (self.ui.export_peak_label, self.ui.export_peak_button),
        (self.ui.export_fig_label, self.ui.export_fig_button),
    ]
    _layout_two_column_card(gb, rows)

def _replace_show_buttons_with_tabs(self) -> None:
    bar = getattr(self.ui, "show_layout", None)
    if bar is None:
        return

    btn_map = [
        ("EIS Data", getattr(self.ui, "show_EIS", None)),
        ("Lin-KK Residual", getattr(self.ui, "show_KK_res", None)),
        ("DRT Residual", getattr(self.ui, "show_re_res", None)),
        ("DRT", getattr(self.ui, "show_DRT", None)),
        ("DRT comparison", getattr(self.ui, "show_DRT_comp", None)),
        ("DRT Map", getattr(self.ui, "show_DRT_map", None)),
        ("Peak Comparison", getattr(self.ui, "show_peak_comp", None)),
        ("Magnitude", getattr(self.ui, "show_mag", None)),
        ("Phase", getattr(self.ui, "show_phase", None)),
        ("Re Part", getattr(self.ui, "show_re", None)),
        ("Im Part", getattr(self.ui, "show_im", None)),
        ("EIS Score", getattr(self.ui, "show_score", None))
    ]

    hide_buttons = [
        getattr(self.ui, "show_EIS", None),
        getattr(self.ui, "show_KK_res", None),
        getattr(self.ui, "show_mag", None),
        getattr(self.ui, "show_phase", None),
        getattr(self.ui, "show_re", None),
        getattr(self.ui, "show_im", None),
        getattr(self.ui, "show_re_res", None),
        getattr(self.ui, "show_im_res", None),
        getattr(self.ui, "show_score", None),
        getattr(self.ui, "show_DRT", None),
        getattr(self.ui, "show_DRT_comp", None),
        getattr(self.ui, "show_DRT_map", None),
        getattr(self.ui, "show_peak_comp", None),
    ]

    for b in hide_buttons:
        if b is not None:
            b.hide()

    _clear_layout_widget(bar)

    lay = QtWidgets.QVBoxLayout(bar)
    lay.setContentsMargins(10, 4, 10, 0)
    lay.setSpacing(4)

    # ---------- top thin scrollbar ----------
    top_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal, bar)
    top_scrollbar.setObjectName("TopTabScrollBar")

    # Make the top scrollbar as thin as the sidebar scrollbar.
    top_scrollbar.setFixedHeight(2)

    top_scrollbar.setStyleSheet("""
        QScrollBar#TopTabScrollBar:horizontal {
            background: transparent;
            height: 2px;
            margin: 0px 6px 0px 6px;
            border: 0px;
        }

        QScrollBar#TopTabScrollBar::handle:horizontal {
            background: rgba(60, 60, 67, 0.35);
            border-radius: 3px;
            min-width: 60px;
        }

        QScrollBar#TopTabScrollBar::handle:horizontal:hover {
            background: rgba(60, 60, 67, 0.50);
        }

        QScrollBar#TopTabScrollBar::add-line:horizontal,
        QScrollBar#TopTabScrollBar::sub-line:horizontal {
            width: 0px;
            height: 0px;
            border: 0px;
            background: transparent;
        }

        QScrollBar#TopTabScrollBar::add-page:horizontal,
        QScrollBar#TopTabScrollBar::sub-page:horizontal {
            background: transparent;
            border: 0px;
        }
    """)

    # ---------- scroll area for tab buttons ----------
    scroll = QtWidgets.QScrollArea(bar)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    scroll.setFixedHeight(46)
    scroll.setStyleSheet("""
        QScrollArea {
            border: 0px;
            background: transparent;
        }
    """)

    tab_host = QtWidgets.QWidget(scroll)
    tab_host.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)

    tab_layout = QtWidgets.QHBoxLayout(tab_host)
    tab_layout.setContentsMargins(0, 0, 0, 0)
    tab_layout.setSpacing(8)

    self._top_tab_buttons = []
    self._top_tab_buttons_by_name = {}

    def _make_tab_button(title, src_button):
        btn = QtWidgets.QPushButton(title, tab_host)
        btn.setCheckable(True)
        btn.setMinimumHeight(36)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        btn.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                border: 0px solid #D2D2D7;
                border-radius: 10px;
                padding: 6px 14px;
                color: #1D1D1F;
            }
            QPushButton:hover {
                background: #F2F2F7;
            }
            QPushButton:checked {
                background: #E9E9EE;
            }
        """)

        def _clicked():
            for b in getattr(self, "_top_tab_buttons", []):
                b.setChecked(False)
            btn.setChecked(True)
            if src_button is not None:
                src_button.click()

        btn.clicked.connect(_clicked)
        return btn

    for title, src_button in btn_map:
        btn = _make_tab_button(title, src_button)
        self._top_tab_buttons.append(btn)
        self._top_tab_buttons_by_name[title] = btn
        if title == 'Peak Comparison':
            btn.hide()
        tab_layout.addWidget(btn)

    tab_layout.addStretch(1)
    scroll.setWidget(tab_host)

    # Put scrollbar ABOVE the button row.
    lay.addWidget(scroll)
    lay.addWidget(top_scrollbar)

    inner_bar = scroll.horizontalScrollBar()

    def _sync_scrollbar_range():
        try:
            top_scrollbar.blockSignals(True)
            top_scrollbar.setRange(inner_bar.minimum(), inner_bar.maximum())
            top_scrollbar.setPageStep(max(1, inner_bar.pageStep() // 10))
            top_scrollbar.setSingleStep(30)
            top_scrollbar.setValue(inner_bar.value())
            top_scrollbar.setVisible(inner_bar.maximum() > 0)
        finally:
            top_scrollbar.blockSignals(False)

    def _top_to_inner(value):
        inner_bar.setValue(value)

    def _inner_to_top(value):
        top_scrollbar.setValue(value)

    top_scrollbar.valueChanged.connect(_top_to_inner)
    inner_bar.valueChanged.connect(_inner_to_top)
    inner_bar.rangeChanged.connect(lambda *_: _sync_scrollbar_range())

    # ---------- wheel event: vertical wheel controls horizontal scrolling ----------
    # ---------- wheel event: mouse wheel switches the selected top tab ----------
    class TopTabWheelFilter(QtCore.QObject):
        def __init__(self, gui_window, scroll_area, target_scrollbar):
            super().__init__(gui_window)
            self._gui = gui_window
            self._scroll = scroll_area
            self._bar = target_scrollbar

        def _current_index(self):
            buttons = getattr(self._gui, "_top_tab_buttons", []) or []
            for i, btn in enumerate(buttons):
                try:
                    if btn.isChecked():
                        return i
                except Exception:
                    pass
            return 0

        def _activate_index(self, index):
            buttons = getattr(self._gui, "_top_tab_buttons", []) or []
            if not buttons:
                return

            index = max(0, min(int(index), len(buttons) - 1))
            btn = buttons[index]

            # Trigger the original button logic:
            # this checks the button and calls the hidden original show_xxx button.
            try:
                btn.click()
            except Exception:
                return

            # Make the selected tab visible inside the horizontal scroll area.
            try:
                x = btn.x()
                w = btn.width()
                view_w = self._scroll.viewport().width()
                left = self._bar.value()
                right = left + view_w

                if x < left:
                    self._bar.setValue(max(self._bar.minimum(), x - 8))
                elif x + w > right:
                    self._bar.setValue(min(self._bar.maximum(), x + w - view_w + 8))
            except Exception:
                pass

        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Wheel:
                try:
                    delta = event.angleDelta().y()
                    if delta == 0:
                        delta = event.angleDelta().x()

                    if delta == 0:
                        return False

                    cur = self._current_index()

                    # Wheel down -> next tab; wheel up -> previous tab.
                    if delta < 0:
                        self._activate_index(cur + 1)
                    else:
                        self._activate_index(cur - 1)

                    event.accept()
                    return True
                except Exception:
                    pass

            return super().eventFilter(obj, event)

    self._top_tab_wheel_filter = TopTabWheelFilter(self, scroll, inner_bar)

    try:
        scroll.viewport().installEventFilter(self._top_tab_wheel_filter)
        scroll.installEventFilter(self._top_tab_wheel_filter)
        tab_host.installEventFilter(self._top_tab_wheel_filter)
        top_scrollbar.installEventFilter(self._top_tab_wheel_filter)

        for btn in self._top_tab_buttons:
            btn.installEventFilter(self._top_tab_wheel_filter)
    except Exception:
        pass

    try:
        bar.setMinimumHeight(72)
        bar.setMaximumHeight(88)
        bar.setStyleSheet("border:0px; background: transparent;")
    except Exception:
        pass

    # Make sure range is correct after Qt finishes layout.
    QtCore.QTimer.singleShot(0, _sync_scrollbar_range)
    QtCore.QTimer.singleShot(100, _sync_scrollbar_range)

    try:
        if self._top_tab_buttons:
            self._top_tab_buttons[0].setChecked(True)
        if btn_map[0][1] is not None:
            btn_map[0][1].click()
    except Exception:
        pass

refactor_ui_layout = _refactor_ui_layout
