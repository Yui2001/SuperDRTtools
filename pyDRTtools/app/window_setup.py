# -*- coding: utf-8 -*-
"""Main-window state initialization and Qt signal wiring."""

from PyQt5 import QtCore, QtWidgets

from ..infrastructure.parallel_fitting import FitProcessManager
from ..infrastructure.peak_fitting import PeakProcessManager
from ..ui import layout
from ..ui.widgets import ExternalFileDropFilter, FileListDelegate


def initialize_window_state(self):
    """Create the UI and initialize non-computational window state."""

    # Initialize the generated widget tree before creating state that reads it.
    self.ui = layout.Ui_MainWindow()
    self.ui.setupUi(self)

    # No spectrum is active until a file or project is loaded.
    self.data = None

    # multi-file stores (batch import + file list)
    self.data_store = {}  # working objects keyed by full path
    self.data_store_raw = {}  # raw objects keyed by full path
    self.current_file_key = None
    self.current_plot_option = 'EIS_data'
    self._eis_selected_raw_indices = []

    # project state (no UI/layout changes)
    self.current_project_path = None
    self._project_dirty = False
    self._project_loading = False
    self._project_io_busy = False

    # fit status tracking
    self.file_meta = {}  # keyed by file path: {fitted, signature, display_name}
    self._fit_all_total = 0

    # selected run mode (simple/bayesian/BHT)
    self.selected_run_mode = 'simple'

    # DRT comparison settings
    self.drt_comp_settings = {
        'ymin': None,
        'ymax': None,
        'start_color': '#08306B',
        'end_color': '#DEEBF7',
    }
    # DRT map (2D heatmap) settings
    self.drt_map_settings = {
        'vmin': None,  # color scale min (auto if None)
        'vmax': None,  # color scale max (auto if None)
        'major_levels': 10,  # number of major levels (Origin-style)
        'minor_levels': 10,  # number of minor levels between majors (controls smoothness)
        'fill_mode': 'contour',  # 'contour' (Fill to Contour Lines) or 'grid' (Fill to Grid Lines)
    }
    self._drt_map_hover_last = None
    self._suppress_file_select = False

    # DRT comparison selection highlight
    self._drt_comp_lines = []  # list of matplotlib Line2D
    self._drt_comp_selected = None  # selected Line2D or None
    self._drt_comp_hover_last = None  # (key, idx) for hover tooltip

    # Peak comparison display settings and hover state.
    self.peak_comp_settings = {
        'ymin': None,
        'ymax': None,
    }
    self._peak_comp_canvas = None
    self._peak_comp_hover_last = None

    # Kramers-Kronig (lin-KK) settings cache
    self.kk_settings = {
        'c': 0.85,
        'max_m': 50,
        'fit_type': 'complex',
    }

    # cache default styles for run-select buttons (so we can restore when not selected)
    self._run_btn_default_style = {
        'simple': self.ui.simple_run_button.styleSheet(),
        'bayesian': self.ui.bayesian_button.styleSheet(),
        'BHT': self.ui.HT_button.styleSheet(),
    }


def connect_window_signals(self):
    """Connect UI events and initialize GUI-side worker coordination."""
    # linking buttons with various functions
    # import button
    self.ui.import_button.clicked.connect(self.import_files)
    self.ui.induct_choice.currentIndexChanged.connect(self.inductance_callback)  # activated when item change

    # files list interactions (if present in layout)
    if hasattr(self.ui, "files_list"):
        self.ui.files_list.currentRowChanged.connect(self._on_file_selected)
        self.ui.files_list.model().rowsMoved.connect(self._on_files_reordered)
        self.ui.files_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.files_list.customContextMenuRequested.connect(self._show_files_context_menu)
        self.ui.files_list.setItemDelegate(FileListDelegate(self.ui.files_list))

        # enable drag & drop import onto Files list, keep internal move reorder
        try:
            self.ui.files_list.setAcceptDrops(True)
            self.ui.files_list.viewport().setAcceptDrops(True)
            self.ui.files_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        except Exception:
            pass
        try:
            self._files_drop_filter = ExternalFileDropFilter(self)
            self.ui.files_list.installEventFilter(self._files_drop_filter)
            self.ui.files_list.viewport().installEventFilter(self._files_drop_filter)
        except Exception:
            pass

    # also accept external file drops onto the main window
    try:
        self.setAcceptDrops(True)
    except Exception:
        pass

    # show buttons
    self.ui.show_EIS.clicked.connect(lambda: self.plotting_callback('EIS_data'))
    if hasattr(self.ui, 'show_KK_res'):
        self.ui.show_KK_res.clicked.connect(lambda: self.plotting_callback('KK_residual'))
    self.ui.show_mag.clicked.connect(lambda: self.plotting_callback('Magnitude'))
    self.ui.show_phase.clicked.connect(lambda: self.plotting_callback('Phase'))
    self.ui.show_re.clicked.connect(lambda: self.plotting_callback('Re_data'))
    self.ui.show_im.clicked.connect(lambda: self.plotting_callback('Im_data'))
    self.ui.show_re_res.setText('DRT Residual')
    self.ui.show_re_res.clicked.connect(lambda: self.plotting_callback('DRT_residual'))
    try:
        self.ui.show_im_res.hide()
    except Exception:
        pass
    self.ui.show_DRT.clicked.connect(lambda: self.plotting_callback('DRT_data'))
    self.ui.show_score.clicked.connect(lambda: self.plotting_callback('Score'))
    if hasattr(self.ui, 'show_DRT_comp'):
        self.ui.show_DRT_comp.clicked.connect(lambda: self.plotting_callback('DRT_comparison'))
    if hasattr(self.ui, 'show_DRT_map'):
        self.ui.show_DRT_map.clicked.connect(lambda: self.plotting_callback('DRT_map'))

    # run mode select buttons
    self.ui.simple_run_button.clicked.connect(lambda: self._select_run_mode('simple'))
    self.ui.bayesian_button.clicked.connect(lambda: self._select_run_mode('bayesian'))
    self.ui.HT_button.clicked.connect(lambda: self._select_run_mode('BHT'))

    # fit buttons (added below the run mode selection)
    if hasattr(self.ui, 'fit_one_button'):
        self.ui.fit_one_button.clicked.connect(self.fit_selected_callback)
    if hasattr(self.ui, 'fit_all_button'):
        self.ui.fit_all_button.clicked.connect(self.fit_all_callback)
    if hasattr(self.ui, 'run_kkr_button'):
        self.ui.run_kkr_button.clicked.connect(self.kk_run_callback)

    # Process-based Fit All is implemented in fit_parallel.py.
    self._fit_process_manager = FitProcessManager()
    self._fit_worker_count = 0
    self._fit_initial_worker_count = 0
    self._fit_process_note = ''
    self._fit_poll_timer = QtCore.QTimer(self)
    self._fit_poll_timer.setInterval(100)
    self._fit_poll_timer.timeout.connect(self._poll_fit_processes)

    self._fit_all_active = False
    self._fit_all_pending = 0
    self._fit_all_done = 0
    self._fit_all_skipped = 0
    self._fit_all_errors = 0
    self._fit_all_restore = {}

    # Peak deconvolution uses the same non-blocking, adaptive process model as
    # Fit All, but its worker implementation remains isolated from the GUI.
    self._peak_process_manager = PeakProcessManager()
    self._peak_worker_count = 0
    self._peak_process_note = ''
    self._peak_poll_timer = QtCore.QTimer(self)
    self._peak_poll_timer.setInterval(100)
    self._peak_poll_timer.timeout.connect(self._poll_peak_processes)
    self._peak_active = False
    self._peak_total = 0
    self._peak_pending = 0
    self._peak_done = 0
    self._peak_errors = 0
    self._peak_skipped = 0
    self._updating_peak_table = False

    self.ui.peak_decon_button.clicked.connect(self.peak_selected_callback)
    if hasattr(self.ui, 'peak_all_button'):
        self.ui.peak_all_button.clicked.connect(self.peak_all_callback)
    if hasattr(self.ui, 'peak_results_table'):
        self.ui.peak_results_table.itemChanged.connect(self._on_peak_result_item_changed)
    if hasattr(self.ui, 'show_peak_comp'):
        self.ui.show_peak_comp.clicked.connect(lambda: self.plotting_callback('Peak_comparison'))

    # default: simple selected
    self._select_run_mode('simple')

    # export result buttons
    self.ui.export_DRT_button.clicked.connect(self.export_DRT)
    self.ui.export_EIS_button.clicked.connect(self.export_EIS)
    self.ui.export_peak_button.clicked.connect(self.export_peak_results)
    self.ui.export_fig_button.clicked.connect(self.export_fig)

    # Existing project controls: connect logic only; do not change button layout/style.
    if hasattr(self.ui, 'open_project_button'):
        self.ui.open_project_button.clicked.connect(self.open_project_callback)
    if hasattr(self.ui, 'save_project_button'):
        self.ui.save_project_button.clicked.connect(self.save_project_callback)
        self.ui.save_project_button.setToolTip(
            'Right-click to choose Save Project As...'
        )
        # Right-click on the existing Save button provides Save As without changing the button.
        self.ui.save_project_button.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.save_project_button.customContextMenuRequested.connect(
            self._show_save_project_context_menu
        )
    if hasattr(self.ui, 'save_project_as_button'):
        self.ui.save_project_as_button.clicked.connect(self.save_project_as_callback)
    if hasattr(self.ui, 'new_window_button'):
        self.ui.new_window_button.clicked.connect(self.open_new_window_callback)
    self._project_buttons_connected = True

    self._install_project_dirty_tracking()
    self._refresh_peak_results_panel()
    self._update_peak_comparison_availability()
