# -*- coding: utf-8 -*-
"""Application-wide Qt font, color, control, and scrollbar theme."""

from PyQt5 import QtGui, QtWidgets


def apply_flat_theme(app: QtWidgets.QApplication) -> None:
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    font = QtGui.QFont("Arial")
    font.setPointSize(11)
    app.setFont(font)

    qss = """
    QWidget {
        color: #1D1D1F;
        font-family: Arial;
        font-size: 11pt;
    }

    /* Make combo boxes show a clear down arrow */
    QComboBox {
        padding-right: 28px;              /* 给箭头留位置 */
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border: 0px;
        border-top-right-radius: 10px;
        border-bottom-right-radius: 10px;
    }

    QComboBox::down-arrow {
        width: 0px;
        height: 0px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 2px solid #6E6E73;    /* 灰色倒三角 */
        margin-right: 10px;
    }

    QGroupBox {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 12px;
        margin-top: 16px;
        padding: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: #1D1D1F;
        font-weight: 600;
    }

    QPushButton {
        background: #FFFFFF;
        border: 1px solid #D2D2D7;
        border-radius: 10px;
        padding: 6px 12px;
        min-height: 28px;
    }
    QPushButton:hover { background: #F2F2F7; border-color: #C7C7CC; }
    QPushButton:pressed { background: #EAEAEE; border-color: #BDBDC2; }
    QPushButton:checked { background: #E9E9EE; border-color: #BDBDC2; }

    QLineEdit, QComboBox {
        background: #FFFFFF;
        border: 1px solid #D2D2D7;
        border-radius: 10px;
        padding: 6px 10px;
        min-height: 28px;
        selection-background-color: #0A84FF;
    }
    QLineEdit:focus, QComboBox:focus { border: 1px solid #0A84FF; }

    QComboBox::drop-down { border: 0px; width: 24px; }
    QComboBox QAbstractItemView {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 10px;
        selection-background-color: #0A84FF;
        selection-color: white;
        padding: 4px;
    }

    /* QTabWidget (top navigation) */
    QTabWidget::pane { border: 0px; background: transparent; }
    QTabBar::tab {
        background: #FFFFFF;
        border: 0px solid #D2D2D7;
        padding: 6px 14px;
        margin-right: 8px;
        border-radius: 10px;
        min-height: 28px;
    }
    QTabBar::tab:selected { background: #E9E9EE; border-color: #BDBDC2; }
    QTabBar::tab:hover { background: #F2F2F7; }

    QScrollArea { border: 0px; background: transparent; }

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 6px 4px 6px 4px;
}
QScrollBar::handle:vertical {
    background: rgba(60, 60, 67, 0.35);
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: rgba(60, 60, 67, 0.50); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 4px 6px 4px 6px;
}
QScrollBar::handle:horizontal {
    background: rgba(60, 60, 67, 0.35);
    border-radius: 5px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: rgba(60, 60, 67, 0.50); }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

/* ComboBox arrow + popup */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 26px;
    border: 0px;
}
QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 10px;
    selection-background-color: #0A84FF;
    selection-color: white;
    padding: 4px;
    outline: 0;
}


/* ComboBox popup visibility fix (Qt sometimes uses QTreeView/QTableView under the hood) */
QAbstractItemView {
    background: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid #E5E5EA;
    border-radius: 10px;
    outline: 0;
}
QAbstractItemView::item {
    color: #1D1D1F;
    padding: 6px 10px;
    min-height: 28px;
}
QAbstractItemView::item:selected {
    background: #0A84FF;
    color: #FFFFFF;
    border-radius: 8px;
}
QTreeView, QTableView, QListView {
    background: #FFFFFF;
    color: #1D1D1F;
}

/* Make QComboBox popup items clearly visible */
QListView {
    background: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid #E5E5EA;
    border-radius: 10px;
    outline: 0;
}
QListView::item {
    color: #1D1D1F;
    padding: 6px 10px;
    min-height: 28px;
}
QListView::item:selected {
    background: #0A84FF;
    color: #FFFFFF;
}

/* Kill unexpected frame borders/lines around the top bar (DO NOT style all QFrame, it breaks combo popups) */
QFrame#show_layout {
    border: 0px;
    background: transparent;
}

/* ComboBox popup - force readable items (some Qt styles use QFrame/QAbstractItemView internally) */
QComboBox QAbstractItemView, QComboBox QListView, QListView, QTreeView, QTableView {
    background: #FFFFFF;
    color: #1D1D1F;
}
QComboBox QAbstractItemView::item, QListView::item, QTreeView::item, QTableView::item {
    color: #1D1D1F;
}
QComboBox QAbstractItemView::item:selected, QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {
    background: #0A84FF;
    color: #FFFFFF;
}
"""
    app.setStyleSheet(qss)
