#!/usr/bin/env python
"""GridFlow GUI - single-task script
A colourful, user-friendly front-end that wraps the GridFlow CLI commands.
Copyright © 2025 Bhuwan Shah - released under the AGPL-v3.
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from threading import Event
from functools import partial
from os.path import expanduser
from typing import Optional, Dict, Callable, Any

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QPushButton, QComboBox, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QAction, QTextEdit, QSizePolicy, QMenu,
    QScrollArea, QGraphicsOpacityEffect, QSplitter, QCompleter
)
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QSettings, QTimer, QObject, QPropertyAnimation, QEasingCurve, QRect

from gridflow.thread_stopper import ThreadManager
from gridflow.logging_utils import setup_logging
from gridflow.commands import (
    download_command, download_cmip5_command, download_prism_command,
    crop_command, clip_command, catalog_command
)

# ----------------------------------------------------------------------
#  FIRST-RUN FLAG - saved in ~/.gridflow_gui.ini
# ----------------------------------------------------------------------
_SETTINGS_ORG = "GridFlow"
_SETTINGS_APP = "GUI"

def is_first_run() -> bool:
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    first = s.value("first_run", True, type=bool)
    if first:
        s.setValue("first_run", False)
    return first

# ----------------------------------------------------------------------
#   PRESET DICT (workers, log-verbosity)
# ----------------------------------------------------------------------
PRESETS = {
    "Beginner":      (4,  "minimal"),
    "Intermediate":  (8,  "normal"),
    "Advanced":      (25, "debug"),
    "Custom":        (None, None)        # unlocked values
}

def mk_label(text: str, indent: int = 0, required: bool = False) -> QLabel:
    """Return a label that is left-aligned and shares a common width, with optional indentation and required marker."""
    lbl = QLabel(text)
    lbl.setMinimumWidth(LABEL_COL + indent)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    style = f"padding-left: {indent}px;" if indent > 0 else ""
    lbl.setStyleSheet(style)
    return lbl

# --------------------------------------------------------------------------
#  Modern theming helper – full replacement
# --------------------------------------------------------------------------
def apply_theme(
    app: QApplication,
    name: str = "default",
    base_pt: int = 14,
    small_pt: int = 12,
    menubar_h_px: int = 30,
) -> None:
    """
    Apply / re-apply a named colour theme and global font sizes.

    Parameters
    ----------
    app            : QApplication
    name           : str   ( default | cosmic | sand | ocean )
    base_pt        : int   point size for all widgets except menus
    small_pt       : int   point size for menus / status text
    menubar_h_px   : int   minimum menubar height
    """
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch() if screen else 96.0  # Fallback DPI
    scale_factor = dpi / 96.0  # Standard DPI is 96
    base_pt = int(base_pt * scale_factor)
    small_pt = int(small_pt * scale_factor)
    menubar_h_px = int(menubar_h_px * scale_factor)

    # Set global font
    app.setFont(QFont("Inter", base_pt))

    # Color palette per theme
    name = name.lower()

    if name == "cosmic":                     # Cosmic Night
        window_bg, base_bg = "#1a1b26", "#24283b"
        border, accent     = "#414868", "#7aa2f7"
        button_bg          = "#7aa2f7"
        button_hover       = "#8eafff"
        button_down        = "#628bf5"
        text               = "#c0caf5"
        card_bg            = "#2a2e48"
        halo_bg            = "rgba(255,255,255,0.08)"
        progress_bg        = "#3a3f5c"  # Lighter than #666
        progress_text      = "#ffffff"  # White text
        log_border         = "#414868"
        log_bg             = "rgba(65,72,104,0.3)"

    elif name == "sand":                     # Ashy Sands
        window_bg, base_bg = "#c9c2a6", "#e8e5d7"
        border, accent     = "#9e9982", "#a8a288"
        button_bg          = "#969173"
        button_hover       = "#a8a288"
        button_down        = "#857f66"
        text               = "#3c3a32"
        card_bg            = "#e8e5d7"
        halo_bg            = "rgba(168,162,136,0.17)"
        progress_bg        = "#d5d2c1"  # Lighter than #666
        progress_text      = "#ffffff"  # White text
        log_border         = "#9e9982"
        log_bg             = "rgba(158,153,130,0.2)"

    elif name == "ocean":                    # Ocean Breeze
        window_bg, base_bg = "#a3dffa", "#e6f7ff"
        border, accent     = "#4a90e2", "#0077b6"
        button_bg          = "#006494"
        button_hover       = "#0a77b6"
        button_down        = "#00517a"
        text               = "#003087"
        card_bg            = "#e6f7ff"
        halo_bg            = "rgba(0,119,182,0.15)"
        progress_bg        = "#c7e9ff"  # Lighter than #666
        progress_text      = "#ffffff"  # White text
        log_border         = "#4a90e2"
        log_bg             = "rgba(74,144,226,0.15)"

    else:                                    # Default light
        window_bg, base_bg = "#f4f7fc", "#ffffff"
        border, accent     = "#d1d9e6", "#4d90fe"
        button_bg          = "#4d90fe"
        button_hover       = "#6da8ff"
        button_down        = "#3579e6"
        text               = "#1a1a1a"
        card_bg            = "#ffffff"
        halo_bg            = "rgba(77,144,254,0.12)"
        progress_bg        = "#d8dee9"  # Slightly darker than #e8ecef for contrast
        progress_text      = "#1a1a1a"  # Dark text for visibility
        log_border         = "#d1d9e6"
        log_bg             = "rgba(209,217,230,0.15)"

    # Qt palette
    pal = QPalette()
    pal.setColor(QPalette.Window,            QColor(window_bg))
    pal.setColor(QPalette.WindowText,        QColor(text))
    pal.setColor(QPalette.Base,              QColor(base_bg))
    pal.setColor(QPalette.AlternateBase,     QColor(window_bg))
    pal.setColor(QPalette.Text,              QColor(text))
    pal.setColor(QPalette.Button,            QColor(button_bg))
    pal.setColor(QPalette.ButtonText,        QColor("#ffffff"))
    pal.setColor(QPalette.Highlight,         QColor(accent))
    pal.setColor(QPalette.HighlightedText,   QColor("#ffffff"))
    app.setPalette(pal)

    # Style-sheet
    app.setStyleSheet(f"""
    /* Menubar */
    QMenuBar {{
        background:{border};
        color:{text};
        font-size:{small_pt}pt;
        min-height:{menubar_h_px}px;
        padding:6px 8px;
        font-weight:500;
    }}
    QMenuBar::item:selected {{
        background:{button_hover};
        color:#ffffff;
        border-radius:4px;
    }}

    /* Menus */
    QMenu {{
        background:{base_bg};
        color:{text};
        border:1px solid {border};
        font-size:{small_pt}pt;
        padding:4px;
    }}
    QMenu::item:selected {{
        background:{accent};
        color:#ffffff;
        border-radius:4px;
    }}

    /* Text-entry widgets & combos */
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
        font-size:{base_pt}pt;
        background:{base_bg};
        color:{text};
        border:1px solid {border};
        padding:6px;
        border-radius:4px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border:1px solid {accent};
        background:{card_bg};
    }}
    QComboBox QAbstractItemView {{
        font-size:{base_pt}pt;
        background:{base_bg};
        color:{text};
        border:1px solid {border};
        selection-background-color:{accent};
        selection-color:#ffffff;
    }}
    QComboBox QAbstractItemView::item:hover    {{ background:{button_hover}; color:#ffffff; }}
    QComboBox QAbstractItemView::item:selected {{ background:{button_down};  color:#ffffff; }}

    /* Push-buttons */
    QPushButton {{
        font-size:{base_pt}pt;
        background:{button_bg};
        color:#ffffff;
        border:none;
        border-radius:8px;
        padding:8px 16px;
        font-weight:600;
    }}
    QPushButton:hover       {{ background:{button_hover}; }}
    QPushButton:pressed,
    QPushButton:checked     {{ background:{button_down}; }}

    /* Progress-bar */
    QProgressBar {{
        text-align:center;
        font-size:{small_pt}pt;
        font-weight:bold;  /* Bold font for all themes */
        background:{progress_bg};
        border:1px solid {border};
        border-radius:8px;
        color:{progress_text};  /* Theme-specific text color */
    }}
    QProgressBar::chunk {{
        background:qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                   stop:0 {accent}, stop:1 {button_hover});
        border-radius:6px;
        margin:2px;
    }}

    /* Labels */
    QLabel {{ color:{text}; font-size:{base_pt}pt; }}

    /* Card-like sections */
    QWidget#card {{
        background:{card_bg};
        border-radius:12px;
        padding:15px;
        border:1px solid {border};
    }}

    /* Halo behind the logo */
    QWidget#halo {{
        background:{halo_bg};
        border-radius:15px;
    }}

    /* Log section */
    QWidget#logCard {{
        background:{log_bg};
        border:1px solid {log_border};
        border-radius:8px;
        padding:10px;
    }}
    QTextEdit {{
        background:{base_bg};
        color:{text};
        border:none;
        border-radius:4px;
        padding:10px;
    }}

    /* Custom style for large checkboxes */
    QCheckBox#largeCheckbox {{
        font-size:{base_pt + 2}pt;
    }}
    QCheckBox#largeCheckbox::indicator {{
        width:24px;
        height:24px;
    }}
    QCheckBox#largeCheckbox::indicator:unchecked {{
        border:2px solid {border};
        background:{base_bg};
    }}
    QCheckBox#largeCheckbox::indicator:checked {{
        border:2px solid {accent};
        background:{accent};
    }}
    /* Slim, themed vertical scrollbar */
    QScrollBar:vertical {{
        width:12px;
        margin:0;                     /* let handle span entire track */
        background:transparent;
    }}
    QScrollBar::handle:vertical {{
        border-radius:6px;
        background:{accent};
        min-height:24px;              /* ensure it can’t shrink into the arrow areas */
    }}

    /* hide and style the “up”/“down” sub-/add-buttons */
    QScrollBar::sub-line:vertical,
    QScrollBar::add-line:vertical {{
        height:12px;
        background:transparent;
        subcontrol-origin: margin;
    }}
    QScrollBar::sub-line:vertical {{
        subcontrol-position: top;
    }}
    QScrollBar::add-line:vertical {{
        subcontrol-position: bottom;
    }}
    QScrollBar::sub-line:vertical:hover,
    QScrollBar::add-line:vertical:hover {{
        background:{halo_bg};         /* subtle hover feedback */
    }}

    /* no extra coloring on the “page” areas */
    QScrollBar::sub-page:vertical,
    QScrollBar::add-page:vertical {{
        background:transparent;
    }}
    """)

# --------------------------- constants ---------------------------
# LOGO_SIZE = (350, 150)
COPYRIGHT_TEXT = "© 2025 Bhuwan Shah  |  GridFlow"
ABOUT_DIALOG_HTML = (
    "<h2>GridFlow</h2>"
    "<p>Graphical front-end for high-resolution climate data processing.</p>"
    "<p>Copyright © 2025 Bhuwan Shah<br>"
    "Released under the AGPL-v3 licence.</p>"
)
# LABEL_COL = 120

# --------------------------- Qt‑logging bridge ---------------------------
class QtHandler(logging.Handler):
    def __init__(self, log_signal, progress_signal):
        super().__init__()
        self.log_signal = log_signal
        self.progress_signal = progress_signal
        self.progress_regex = re.compile(r"Progress: (\d+)/(\d+) files")
        self.completed_regex = re.compile(r"Completed: (\d+)/(\d+) files")

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)
        m = self.progress_regex.search(msg) or self.completed_regex.search(msg)
        if m:
            current, total = map(int, m.groups())
            self.progress_signal.emit(current, total)

def get_scaled_size(base_width, base_height, app):
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch() if screen else 96.0  # Fallback DPI
    scale_factor = dpi / 96.0
    return int(base_width * scale_factor), int(base_height * scale_factor)

def mk_label(text: str, indent: int = 0, required: bool = False, label_col: int = 120) -> QLabel:
    """Return a label that is left-aligned and shares a common width, with optional indentation and required marker."""
    lbl = QLabel(text)
    lbl.setMinimumWidth(label_col + indent)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    style = f"padding-left: {indent}px;" if indent > 0 else ""
    lbl.setStyleSheet(style)
    return lbl

# --------------------------- main window ---------------------------
class GridFlowGUI(QMainWindow):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    task_completed = pyqtSignal(bool)
    stopping = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.thread_manager = ThreadManager(verbose=True)  # Initialize ThreadManager
        self.task_running = False

        # Initialize LOGO_SIZE and LABEL_COL after QApplication is available
        self.LOGO_SIZE = get_scaled_size(350, 150, QApplication.instance())
        self.LABEL_COL = get_scaled_size(120, 0, QApplication.instance())[0]

        self.mk_label = lambda text, indent=0, required=False: mk_label(
            text, indent, required, self.LABEL_COL
        )

        VOCAB_DIR = Path(__file__).parent / "vocab"

        def _load(name):
            # Handle both development and bundled environments
            if getattr(sys, 'frozen', False):
                # Running as executable (PyInstaller)
                base_path = os.path.join(sys._MEIPASS, 'gui', 'vocab')
            else:
                # Running as script
                base_path = Path(__file__).parent / 'vocab'
            f = Path(base_path) / name
            self.log_signal.emit(f"Attempting to load vocab file: {f}")
            if not f.exists():
                self.log_signal.emit(f"Vocab file not found: {f}")
                return []
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError as e:
                self.log_signal.emit(f"Failed to parse vocab file {f}: {e}")
                return []
        
        # CMIP6 vocab
        self.cmip6_activity_id     = _load("cmip6_activity_id.json")
        self.cmip6_experiment_id   = _load("cmip6_experiment_id.json")
        self.cmip6_variable_id     = _load("cmip6_variable_id.json")
        self.cmip6_table_id        = _load("cmip6_table_id.json")
        self.cmip6_source_id       = _load("cmip6_source_id.json")
        self.cmip6_grid_label      = _load("cmip6_grid_label.json")
        self.cmip6_member_id       = _load("cmip6_member_id.json")
        self.cmip6_variant_label   = _load("cmip6_variant_label.json")
        self.cmip6_institution_id  = _load("cmip6_institution_id.json")
        self.cmip6_source_type     = _load("cmip6_source_type.json")
        self.cmip6_frequency       = _load("cmip6_time_frequency.json")
        self.cmip6_resolution       = _load("cmip6_resolution.json")

        # CMIP5 vocab
        self.cmip5_model           = _load("cmip5_model.json")
        self.cmip5_experiment      = _load("cmip5_experiment.json")
        self.cmip5_variable        = _load("cmip5_variable.json")
        self.cmip5_time_frequency  = _load("cmip5_time_frequency.json")
        self.cmip5_ensemble        = _load("cmip5_ensemble.json")
        self.cmip5_institute       = _load("cmip5_institute.json")

        self.setWindowTitle("GridFlow Data Processor")
        self.init_ui()
        self.init_logging()

    def init_logging(self):
        log_dir = Path(expanduser('~/.gridflow/logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir, "minimal", prefix="gridflow_")
        self.log_signal.connect(self.on_log_message)
        self.progress_signal.connect(self.on_progress_update)
        self.task_completed.connect(self.on_task_completed)
        self.stopping.connect(self.on_stopping)
        self.stopped.connect(self.on_task_stopped)
        self.qt_handler = QtHandler(self.log_signal, self.progress_signal)
        self.qt_handler.setFormatter(logging.Formatter("%(message)s"))
        self.qt_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.qt_handler)

    def init_ui(self):
        """
        Top-to-bottom layout order:
            1. Header card (logo + pickers)
            2. Form card (scrollable)
            3. Workers / verbosity row + buttons
            4. Progress-bar
            5. LOG PANE (in splitter)
            6. Footer

        The header, form, and controls live in a QScrollArea to prevent squashing.
        The form card has its own QScrollArea for independent scrolling of arguments.
        A vertical QSplitter separates the top section from the log pane.
        """
        # Central container
        container = QWidget()
        vmain = QVBoxLayout(container)
        vmain.setSpacing(15)
        self.setCentralWidget(container)

        # Maximize window on startup
        # self.resize(1024, 768)
        # screen = QApplication.primaryScreen().availableGeometry()
        # width = min(screen.width() * 0.8, 1024)  # 80% of screen width, capped at 1024
        # height = min(screen.height() * 0.8, 768)  # 80% of screen height, capped at 768
        # self.resize(width, height)

        screen = QApplication.primaryScreen()
        available_geometry = screen.availableGeometry() if screen else QRect(0, 0, 1024, 768)
        width = min(available_geometry.width() * 0.8, 1024)
        height = min(available_geometry.height() * 0.8, 768)
        self.resize(width, height)

        # Splitter (top scroll | log)
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(10)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgba(0,0,0,0.1), stop:1 rgba(0,0,0,0.2));
                border: 1px solid rgba(0,0,0,0.3);
                border-radius: 4px;
            }
        """)
        vmain.addWidget(splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # TOP PANE
        top_scroll = QScrollArea()
        top_scroll.setWidgetResizable(True)
        top_scroll.setStyleSheet("QScrollArea { border: none; }")
        splitter.addWidget(top_scroll)

        upper = QWidget()
        ulay = QVBoxLayout(upper)
        ulay.setSpacing(20)
        # allocate space: header 2, form 3, controls 1
        ulay.setStretch(0, 2)   # header card
        ulay.setStretch(1, 3)   # form card
        ulay.setStretch(2, 1)   # workers/buttons row
        top_scroll.setWidget(upper)

        # HEADER CARD (logo + pickers)
        header = QWidget(objectName="card")
        h = QVBoxLayout(header)
        h.setContentsMargins(20, 20, 20, 20)
        h.setAlignment(Qt.AlignHCenter)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        # Logo inside "halo"
        self.logo_lbl = QLabel(alignment=Qt.AlignCenter)
        self.logo_lbl.setScaledContents(True)
        self.logo_lbl.setMaximumHeight(150)
        self.logo_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.update_logo_pixmap()

        opacity = QGraphicsOpacityEffect(self.logo_lbl)
        self.logo_lbl.setGraphicsEffect(opacity)
        opacity.setOpacity(0.0)

        self.logo_anim = QPropertyAnimation(opacity, b"opacity")
        self.logo_anim.setDuration(1500)
        self.logo_anim.setStartValue(0.0)
        self.logo_anim.setEndValue(1.0)
        self.logo_anim.setEasingCurve(QEasingCurve.InOutQuad)

        halo = QWidget(objectName="halo")
        halo_l = QVBoxLayout(halo)
        halo_l.setContentsMargins(15, 15, 15, 15)
        halo_l.addWidget(self.logo_lbl, 0, Qt.AlignCenter)
        h.addWidget(halo)

        # Picker form
        header_form = QFormLayout()
        header_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_form.setHorizontalSpacing(20)
        header_form.setVerticalSpacing(10)

        self.skill_combo = QComboBox()
        self.skill_combo.addItems(PRESETS.keys())
        self.skill_combo.currentTextChanged.connect(self.on_skill_change)
        self.skill_combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        header_form.addRow(mk_label("Skill level:"), self.skill_combo)

        ds_wrap = QWidget()
        ds_h = QHBoxLayout(ds_wrap)
        ds_h.setContentsMargins(0, 0, 0, 0)
        ds_h.setSpacing(15)

        self.src_combo = QComboBox()
        self.src_combo.addItems(["CMIP6", "CMIP5", "PRISM"])
        self.src_combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        proc_lbl = mk_label("Process:")
        self.proc_combo = QComboBox()
        self.proc_combo.addItems(["Download", "Spatial Crop", "Spatial Clip", "Catalog Build"])
        self.proc_combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.src_combo.currentTextChanged.connect(
            lambda _: self.update_form(self.proc_combo.currentText()))
        self.proc_combo.currentTextChanged.connect(self.update_form)

        ds_h.addWidget(self.src_combo)
        ds_h.addWidget(proc_lbl)
        ds_h.addWidget(self.proc_combo)
        ds_h.addStretch(1)

        header_form.addRow(mk_label("Data Source:"), ds_wrap)
        h.addLayout(header_form)
        ulay.addWidget(header)

        # FORM CARD (scrollable)
        form_card = QWidget(objectName="card")
        form_l = QVBoxLayout(form_card)
        form_l.setContentsMargins(20, 20, 20, 20)

        # Scrollable form area
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setStyleSheet("QScrollArea { border: none; }")
        form_scroll.setViewportMargins(6, 6, 6, 6)
        form_l.addWidget(form_scroll)

        form_content = QWidget()
        form_content_l = QVBoxLayout(form_content)
        form_content_l.setContentsMargins(0, 0, 0, 0)
        form_scroll.setWidget(form_content)

        self.form_layout = QFormLayout()
        self.form_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form_layout.setHorizontalSpacing(20)
        self.form_layout.setHorizontalSpacing(26)
        self.form_layout.setVerticalSpacing(12)
        self.form_layout.setContentsMargins(20, 20, 20, 20)
        form_content_l.addLayout(self.form_layout)
        self.arg_widgets = {}
        ulay.addWidget(form_card)

        # Workers + verbosity + buttons
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Workers:"))
        self.workers_edit = QLineEdit("4")
        self.workers_edit.setMaximumWidth(60)
        cfg_row.addWidget(self.workers_edit)

        cfg_row.addSpacing(20)
        cfg_row.addWidget(QLabel("Verbosity:"))
        self.verbosity_combo = QComboBox()
        self.verbosity_combo.addItems(["minimal", "normal", "verbose", "debug"])
        cfg_row.addWidget(self.verbosity_combo)
        cfg_row.addStretch(1)
        ulay.addLayout(cfg_row)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.setSpacing(15)
        ulay.addLayout(btn_row)

        self.start_btn.clicked.connect(self.start_task)
        self.stop_btn.clicked.connect(self.stop_task)

        # Progress bar
        self.progress_bar = QProgressBar()
        ulay.addWidget(self.progress_bar)

        # LOG PANE (splitter bottom)
        log_container = QWidget(objectName="logCard")
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(10, 10, 10, 10)

        self.log_text = QTextEdit(readOnly=True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_container)

        splitter.setStretchFactor(0, 3)  # Top grows more
        splitter.setStretchFactor(1, 1)

        # Restore saved splitter sizes, or fall back to first-run/default
        saved = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value("split_sizes")
        if saved:
            splitter.setSizes([int(s) for s in saved])
        else:
            splitter.setSizes([600, 100] if is_first_run() else [500, 100])

        splitter.setCollapsible(0, False)  # Prevent top from collapsing
        splitter.setCollapsible(1, True)   # Allow log to collapse

        # Footer
        footer = QLabel(COPYRIGHT_TEXT, alignment=Qt.AlignCenter)
        footer.setStyleSheet("color: rgba(0,0,0,0.7); font-size: 10pt; margin-top: 4px;")
        vmain.addWidget(footer)

        # Menu-bar, theming, startup
        about_act = QAction("&About GridFlow..", self)
        about_act.triggered.connect(self.show_about)
        self.menuBar().addMenu("&Help").addAction(about_act)

        self.theme_menu = self.menuBar().addMenu("&Theme")
        for t in ("Default", "Cosmic", "Sand", "Ocean"):
            act = QAction(t, self, checkable=True)
            act.triggered.connect(partial(self.set_theme, t.lower()))
            self.theme_menu.addAction(act)
        self.theme_menu.actions()[3].setChecked(True)

        self.current_theme = "ocean"
        self.current_font_base = 14

        view_menu = self.menuBar().addMenu("&View")
        font_menu = QMenu("Font &Size  aA", self)
        view_menu.addMenu(font_menu)
        self.font_size_actions = {}
        for label, size in [("Small", 12), ("Medium", 14), ("Large", 16)]:
            act = QAction(label, self, checkable=True)
            act.triggered.connect(lambda ch=False, s=size: self.set_font_size(s))
            font_menu.addAction(act)
            self.font_size_actions[size] = act
        self.font_size_actions[self.current_font_base].setChecked(True)

        apply_theme(QApplication.instance(),
                    name=self.current_theme,
                    base_pt=self.current_font_base,
                    small_pt=max(self.current_font_base - 2, 8))

        # Final startup actions
        self.update_form(self.proc_combo.currentText())
        QTimer.singleShot(500, self.maybe_show_tutorial)
        self.skill_combo.setCurrentText("Beginner")
        QTimer.singleShot(0, self.logo_anim.start)

    # def update_logo_pixmap(self):
    #     if hasattr(self, '_cached_logo_pixmap'):
    #         self.logo_lbl.setPixmap(self._cached_logo_pixmap)
    #         return

    #     if getattr(sys, 'frozen', False):
    #         logo_file = Path(os.path.join(sys._MEIPASS, 'gridflow_logo.png'))
    #     else:
    #         logo_file = Path('gridflow_logo.png')

    #     self.log_signal.emit(f"Attempting to load logo file: {logo_file}")
    #     if not logo_file.exists():
    #         self.log_signal.emit(f"Logo file not found: {logo_file}")
    #         self.logo_lbl.clear()
    #         return

    #     pm = QPixmap(str(logo_file))
    #     max_w, max_h = LOGO_SIZE
    #     pm = pm.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     self._cached_logo_pixmap = pm
    #     self.logo_lbl.setPixmap(pm)

    def update_logo_pixmap(self):
        if hasattr(self, '_cached_logo_pixmap'):
            self.logo_lbl.setPixmap(self._cached_logo_pixmap)
            return

        if getattr(sys, 'frozen', False):
            logo_file = Path(os.path.join(sys._MEIPASS, 'gridflow_logo.png'))
        else:
            logo_file = Path('gridflow_logo.png')

        self.log_signal.emit(f"Attempting to load logo file: {logo_file}")
        if not logo_file.exists():
            self.log_signal.emit(f"Logo file not found: {logo_file}")
            self.logo_lbl.clear()
            return

        pm = QPixmap(str(logo_file))
        max_w, max_h = self.LOGO_SIZE  # Use instance attribute
        pm = pm.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cached_logo_pixmap = pm
        self.logo_lbl.setPixmap(pm)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, 'logo_lbl') and not hasattr(self, '_cached_logo_pixmap'):
            self.update_logo_pixmap()

    def maybe_show_tutorial(self):
        if not is_first_run():
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Welcome to GridFlow")
        msg.setText("I noticed it's your first time running GridFlow.\n"
                    "Would you like to try a demo configuration?")
        msg.setInformativeText("Choose a process to pre-fill with demo settings:")
        download_btn = msg.addButton("CMIP6 Download", QMessageBox.AcceptRole)
        crop_btn = msg.addButton("Spatial Crop", QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.No)
        msg.setDefaultButton(download_btn)
        msg.exec_()

        self.skill_combo.setCurrentText("Beginner")
        if msg.clickedButton() == download_btn:
            self.src_combo.setCurrentText("CMIP6")
            self.proc_combo.setCurrentText("Download")
            self.update_form("Download")
            self.arg_widgets["project"].setText("CMIP6")
            self.arg_widgets["activity"].setText("ScenarioMIP")
            self.arg_widgets["experiment"].setText("ssp585")
            self.arg_widgets["variable"].setText("tas")
            self.arg_widgets["frequency"].setText("mon")
            self.arg_widgets["model"].setText("CanESM5")
            self.arg_widgets["ensemble"].setText("r1i1p1f1")
            self.arg_widgets["output_dir"].setText("cmip6_data")
            self.arg_widgets["metadata_dir"].setText("metadata")
            self.arg_widgets["save_mode"].setCurrentText("flat")
            self.arg_widgets["retries"].setText("3")
            self.arg_widgets["timeout"].setText("30")
            self.arg_widgets["max_downloads"].setText("10")
            self.arg_widgets["latest"].setChecked(True)
            self.arg_widgets["username"].setText("")
            self.arg_widgets["retry_failed"].setText("")
            QMessageBox.information(
                self, "Demo Configuration",
                "The form is pre-filled for a CMIP6 download (tas, ScenarioMIP, ssp585).\n"
                "Click 'Start' to download up to 10 files, or change the source/process."
            )
        elif msg.clickedButton() == crop_btn:
            self.src_combo.setCurrentText("CMIP6")
            self.proc_combo.setCurrentText("Spatial Crop")
            self.update_form("Spatial Crop")
            self.arg_widgets["input_dir"].setText("cmip6_data")
            self.arg_widgets["output_dir"].setText("cmip6_cropped_data")
            self.arg_widgets["min_lat"].setText("35.0")
            self.arg_widgets["max_lat"].setText("45.0")
            self.arg_widgets["min_lon"].setText("-105.0")
            self.arg_widgets["max_lon"].setText("-95.0")
            self.arg_widgets["buffer_km"].setText("50.0")
            QMessageBox.information(
                self, "Demo Configuration",
                "The form is pre-filled for Spatial Crop with bounds 35N-45N, 95W-105W, 50km buffer.\n"
                "Ensure cmip6_data has NetCDF files (run 'Download' first if needed), then click 'Start'."
            )

    def on_skill_change(self, level: str):
        workers, verb = PRESETS[level]
        self.workers_edit.setReadOnly(level != "Custom")
        if workers is not None:
            self.workers_edit.setText(str(workers))
        if verb is not None:
            self.verbosity_combo.setCurrentText(verb)

    def update_form(self, process: str) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self.arg_widgets.clear()

        src = self.src_combo.currentText()
        if src == "PRISM" and process in ["Spatial Crop", "Spatial Clip", "Catalog Build"]:
            QMessageBox.warning(
                self, "Invalid Selection",
                f"PRISM data is not compatible with {process}. "
                "Please select CMIP5 or CMIP6 for Spatial Crop, Spatial Clip, or Catalog Build, or choose 'Download' for PRISM."
            )
            process = "Download"

        valid_processes = ["Download", "Spatial Crop", "Spatial Clip", "Catalog Build"]
        if src == "PRISM":
            valid_processes = ["Download"]

        self.proc_combo.blockSignals(True)
        self.proc_combo.clear()
        self.proc_combo.addItems(valid_processes)
        self.proc_combo.setCurrentText(process if process in valid_processes else valid_processes[0])
        self.proc_combo.blockSignals(False)

        def add_line(label, default="", tip="", indent=0, vocab=None, required=False):
            w = QLineEdit(default)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            w.setToolTip(tip)
            w.setPlaceholderText(f"Enter {label.lower()}…" if not required else f"Enter {label.lower()} (required)")
            if vocab:
                comp = QCompleter(vocab, w)
                comp.setCaseSensitivity(Qt.CaseInsensitive)
                comp.setFilterMode(Qt.MatchContains)
                comp.setCompletionMode(QCompleter.PopupCompletion)
                comp.setMaxVisibleItems(12)
                w.setCompleter(comp)
                original_focus = w.focusInEvent
                def on_focus(event):
                    w.completer().setCompletionPrefix("")
                    w.completer().complete()
                    original_focus(event)
                w.focusInEvent = on_focus
            self.form_layout.addRow(mk_label(label, indent, required), w)
            self.arg_widgets[label.lower().replace(" ", "_")] = w

        def add_file(label, default="", tip="", dir_=False, indent=0, required: bool = False):
            le = QLineEdit(default)
            le.setToolTip(tip)
            le.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            placeholder = f"Select {label.lower()}"
            if required:
                placeholder += " (required)"
            le.setPlaceholderText(placeholder + "...")

            btn = QPushButton("Browse")
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.clicked.connect(lambda *_: self.browse_file(le, dir_))

            wrap = QWidget()
            hl = QHBoxLayout(wrap)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(8)
            hl.addWidget(le, 1)
            hl.addWidget(btn, 0)

            self.form_layout.addRow(mk_label(label, indent, required), wrap)
            self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

            self.arg_widgets[label.lower().replace(" ", "_")] = le

        def add_chk(label, val=False, tip="", indent=0):
            ck = QCheckBox()
            ck.setChecked(val)
            ck.setToolTip(tip)
            if label in ["Latest", "No Verify SSL"]:
                ck.setObjectName("largeCheckbox")
            self.form_layout.addRow(mk_label(label, indent=indent), ck)
            self.arg_widgets[label.lower().replace(" ", "_")] = ck

        def add_combo(label, opts, default="", tip="", indent=0):
            cb = QComboBox()
            cb.addItems(opts)
            if default:
                cb.setCurrentText(default)
            cb.setToolTip(tip)
            self.form_layout.addRow(mk_label(label, indent=indent), cb)
            self.arg_widgets[label.lower().replace(" ", "_")] = cb

        indent_amount = 0

        if process == "Download":
            if src == "CMIP6":
                proj_le = QLineEdit("CMIP6")
                proj_le.setReadOnly(True)
                proj_le.setEnabled(False)
                self.form_layout.addRow(mk_label("Project:", indent_amount), proj_le)
                self.arg_widgets["project"] = proj_le

                add_line("Activity", "ScenarioMIP", "Activity ID for the CMIP6 dataset.",
                        indent=indent_amount, vocab=self.cmip6_activity_id, required=True)
                add_line("Experiment", "ssp585", "Experiment ID for the dataset.",
                        indent=indent_amount, vocab=self.cmip6_experiment_id, required=False)
                add_line("Variable", "tas", "Variable ID to download.",
                        indent=indent_amount, vocab=self.cmip6_variable_id, required=True)
                add_line("Frequency", "mon", "Time frequency of the data.",
                        indent=indent_amount, vocab=self.cmip6_frequency, required=True)
                add_line("Model", "", "Model/source ID.",
                        indent=indent_amount, vocab=self.cmip6_source_id, required=False)
                add_line("Resolution", "100 km", "Nominal resolution of the data.", 
                         indent=indent_amount, vocab=self.cmip6_resolution, required=True)
                add_line("Ensemble", "r1i1p1f1", "Variant label/ensemble member.",
                        indent=indent_amount, vocab=self.cmip6_variant_label, required=False)
                add_line("Institution", "", "Institution ID.",
                        indent=indent_amount, vocab=self.cmip6_institution_id, required=False)
                add_line("Source Type", "", "Source type of the model.",
                        indent=indent_amount, vocab=self.cmip6_source_type, required=False)
                add_line("Grid Label", "", "Grid label for the dataset.",
                        indent=indent_amount, vocab=self.cmip6_grid_label, required=False)
                add_line("Start Date", "", "Start date for data filtering.", indent=indent_amount, required=False)
                add_line("End Date", "", "End date for data filtering.", indent=indent_amount, required=False)
                add_chk("Latest", True, "Check to download only the latest version.", indent=indent_amount)
                add_line("Extra Params", "", "Additional query parameters as JSON string.", indent=indent_amount, required=False)
                add_file("Output Dir", "cmip6_data", "Directory to save downloaded NetCDF files.", dir_=True, indent=indent_amount, required=True)
                add_file("Metadata Dir", "metadata", "Directory to save metadata JSON files.", dir_=True, indent=indent_amount, required=True)
                add_combo("Save Mode", ["flat", "structured"], "flat", "File organization: flat or structured.", indent=indent_amount)
                add_line("Max Downloads", "10", "Maximum number of files to download.", indent=indent_amount, required=False)
                self.arg_widgets["max_downloads"].setText("10")

                adv_chk = QCheckBox("Show advanced options")
                adv_chk.setObjectName("largeCheckbox")
                adv_chk.setToolTip("Toggle retries, SSL and other advanced flags")
                self.form_layout.addRow(mk_label("", indent_amount), adv_chk)

                adv_widget = QWidget()
                adv_layout = QFormLayout(adv_widget)
                adv_layout.setContentsMargins(0, 0, 0, 0)

                retries_le = QLineEdit("3")
                retries_le.setToolTip("Number of download retry attempts.")
                adv_layout.addRow(mk_label("Retries", indent_amount), retries_le)

                timeout_le = QLineEdit("30")
                timeout_le.setToolTip("HTTP request timeout in seconds.")
                adv_layout.addRow(mk_label("Timeout", indent_amount), timeout_le)

                no_ssl_ck = QCheckBox()
                no_ssl_ck.setObjectName("largeCheckbox")
                no_ssl_ck.setChecked(False)
                no_ssl_ck.setToolTip("Disable SSL verification for downloads.")
                adv_layout.addRow(mk_label("No Verify SSL", indent_amount), no_ssl_ck)

                user_le = QLineEdit("")
                adv_layout.addRow(mk_label("Username", indent_amount), user_le)

                pass_le = QLineEdit("")
                adv_layout.addRow(mk_label("Password", indent_amount), pass_le)

                retry_file = QLineEdit("")
                browse_retry = QPushButton("Browse")
                browse_retry.clicked.connect(lambda: self.browse_file(retry_file, False))
                retry_wrap = QWidget()
                retry_lay = QHBoxLayout(retry_wrap)
                retry_lay.setContentsMargins(0, 0, 0, 0)
                retry_lay.setSpacing(8)
                retry_lay.addWidget(retry_file, 1)
                retry_lay.addWidget(browse_retry, 0)
                adv_layout.addRow(mk_label("Retry Failed", indent_amount), retry_wrap)

                adv_widget.setVisible(False)
                adv_chk.toggled.connect(adv_widget.setVisible)
                self.form_layout.addRow(adv_widget)

                self.arg_widgets["retries"] = retries_le
                self.arg_widgets["timeout"] = timeout_le
                self.arg_widgets["no_verify_ssl"] = no_ssl_ck
                self.arg_widgets["username"] = user_le
                self.arg_widgets["password"] = pass_le
                self.arg_widgets["retry_failed"] = retry_file

            elif src == "CMIP5":
                proj_le = QLineEdit("CMIP5")
                proj_le.setReadOnly(True)
                proj_le.setEnabled(False)
                self.form_layout.addRow(mk_label("Project:", indent_amount), proj_le)
                self.arg_widgets["project"] = proj_le

                add_line("Model", "CanESM2", "Model name.",
                        indent=indent_amount, vocab=self.cmip5_model, required=True)
                add_line("Experiment", "historical", "Experiment ID.",
                        indent=indent_amount, vocab=self.cmip5_experiment, required=False)
                add_line("Variable", "tas", "Variable name.",
                        indent=indent_amount, vocab=self.cmip5_variable, required=True)
                add_line("Time Frequency", "mon", "Time frequency.",
                        indent=indent_amount, vocab=self.cmip5_time_frequency, required=True)
                add_line("Ensemble", "r1i1p1", "Ensemble member.",
                        indent=indent_amount, vocab=self.cmip5_ensemble, required=False)
                add_line("Institute", "", "Institute name.",
                        indent=indent_amount, vocab=self.cmip5_institute, required=False)
                add_line("Start Date", "", "Start date for data filtering.", indent=indent_amount, required=False)
                add_line("End Date", "", "End date for data filtering.", indent=indent_amount, required=False)
                add_chk("Latest", True, "Check to download only the latest version.", indent=indent_amount)
                add_line("Extra Params", "", "Additional query parameters as JSON string.", indent=indent_amount, required=False)
                add_file("Output Dir", "cmip5_data", "Directory to save downloaded NetCDF files.", dir_=True, indent=indent_amount, required=True)
                add_file("Metadata Dir", "metadata", "Directory to save metadata JSON files.", dir_=True, indent=indent_amount, required=True)
                add_combo("Save Mode", ["flat", "structured"], "flat", "File organization: flat or structured.", indent=indent_amount)
                add_line("Max Downloads", "", "Maximum number of files to download.", indent=indent_amount, required=False)

                adv_chk = QCheckBox("Show advanced options")
                adv_chk.setObjectName("largeCheckbox")
                adv_chk.setToolTip("Toggle retries, SSL and other advanced flags")
                self.form_layout.addRow(mk_label("", indent_amount), adv_chk)

                adv_widget = QWidget()
                adv_layout = QFormLayout(adv_widget)
                adv_layout.setContentsMargins(0, 0, 0, 0)

                retries_le = QLineEdit("3")
                retries_le.setToolTip("Number of download retry attempts.")
                adv_layout.addRow(mk_label("Retries", indent_amount), retries_le)

                timeout_le = QLineEdit("30")
                timeout_le.setToolTip("HTTP request timeout in seconds.")
                adv_layout.addRow(mk_label("Timeout", indent_amount), timeout_le)

                user_le = QLineEdit("")
                adv_layout.addRow(mk_label("Username", indent_amount), user_le)

                pass_le = QLineEdit("")
                adv_layout.addRow(mk_label("Password", indent_amount), pass_le)

                openid_le = QLineEdit("")
                adv_layout.addRow(mk_label("OpenID", indent_amount), openid_le)

                no_ssl_ck = QCheckBox()
                no_ssl_ck.setObjectName("largeCheckbox")
                no_ssl_ck.setChecked(False)
                no_ssl_ck.setToolTip("Disable SSL verification for downloads.")
                adv_layout.addRow(mk_label("No Verify SSL", indent_amount), no_ssl_ck)

                retry_file = QLineEdit("")
                browse_retry = QPushButton("Browse")
                browse_retry.clicked.connect(lambda: self.browse_file(retry_file, False))
                retry_wrap = QWidget()
                retry_lay = QHBoxLayout(retry_wrap)
                retry_lay.setContentsMargins(0, 0, 0, 0)
                retry_lay.setSpacing(8)
                retry_lay.addWidget(retry_file, 1)
                retry_lay.addWidget(browse_retry, 0)
                adv_layout.addRow(mk_label("Retry Failed", indent_amount), retry_wrap)

                adv_widget.setVisible(False)
                adv_chk.toggled.connect(adv_widget.setVisible)
                self.form_layout.addRow(adv_widget)

                self.arg_widgets["retries"] = retries_le
                self.arg_widgets["timeout"] = timeout_le
                self.arg_widgets["username"] = user_le
                self.arg_widgets["password"] = pass_le
                self.arg_widgets["openid"] = openid_le
                self.arg_widgets["no_verify_ssl"] = no_ssl_ck
                self.arg_widgets["retry_failed"] = retry_file

            else:  # PRISM
                add_combo("Variable", ["ppt", "tmax", "tmin", "tmean", "tdmean", "vpdmin", "vpdmax"], "tmean", "Climate variable to download.", indent=indent_amount)
                add_combo("Resolution", ["4km", "800m"], "4km", "Spatial resolution.", indent=indent_amount)
                add_combo("Time Step", ["daily", "monthly"], "daily", "Temporal resolution.", indent=indent_amount)
                add_line("Start Date", "2020-01-01", "Start date for data.", indent=indent_amount, required=True)
                add_line("End Date", "2020-01-04", "End date for data.", indent=indent_amount, required=True)
                add_file("Output Dir", "prism_data", "Directory to save downloaded files.", dir_=True, indent=indent_amount, required=True)
                add_file("Metadata Dir", "metadata", "Directory to save metadata files.", dir_=True, indent=indent_amount, required=True)
                add_line("Retries", "3", "Number of download retry attempts.", indent=indent_amount, required=False)
                add_line("Timeout", "30", "HTTP request timeout in seconds.", indent=indent_amount, required=False)

        elif process == "Spatial Crop":
            add_file("Input Dir", "cmip6_data", "Directory containing input NetCDF files.", dir_=True, required=True)
            add_file("Output Dir", "cmip6_cropped_data", "Directory to save cropped NetCDF files.", dir_=True, required=True)
            add_line("Min Lat", "35.0", "Minimum latitude bound.", required=True)
            add_line("Max Lat", "45.0", "Maximum latitude bound.", required=True)
            add_line("Min Lon", "-105.0", "Minimum longitude bound.", required=True)
            add_line("Max Lon", "-95.0", "Maximum longitude bound.", required=True)
            add_line("Buffer KM", "50.0", "Buffer distance in kilometers.", required=False)

        elif process == "Spatial Clip":
            add_file("Input Dir", "cmip6_data", "Directory containing input NetCDF files.", dir_=True, required=True)
            add_file("Shapefile Path", "", "Path to shapefile for clipping.", dir_=False, required=True)
            add_line("Buffer KM", "20.0", "Buffer distance in kilometers.", required=False)
            add_file("Output Dir", "cmip6_clipped_data", "Directory to save clipped NetCDF files.", dir_=True, required=True)

        elif process == "Catalog Build":
            add_file("Input Dir", "cmip6_data", "Directory containing NetCDF files to catalog.", dir_=True, required=True)
            add_file("Output Dir", "catalog", "Directory to save catalog JSON file.", dir_=True, required=True)

        if is_first_run() and process == self.proc_combo.currentText():
            if process == "Download":
                if src == "CMIP6":
                    self.arg_widgets["max_downloads"].setText("10")
                elif src == "CMIP5":
                    self.arg_widgets["max_downloads"].setText("10")
                elif src == "PRISM":
                    self.arg_widgets["variable"].setCurrentText("tmean")
                    self.arg_widgets["resolution"].setCurrentText("4km")
                    self.arg_widgets["time_step"].setCurrentText("monthly")
                    self.arg_widgets["start_date"].setText("2020-01")
                    self.arg_widgets["end_date"].setText("2020-03")
            elif process == "Spatial Crop":
                self.arg_widgets["min_lat"].setText("35.0")
                self.arg_widgets["max_lat"].setText("45.0")
                self.arg_widgets["min_lon"].setText("-105.0")
                self.arg_widgets["max_lon"].setText("-95.0")
                self.arg_widgets["buffer_km"].setText("50.0")
            elif process == "Spatial Clip":
                self.arg_widgets["shapefile_path"].setText("shapefiles/iowa_border/iowa_border.shp")
                self.arg_widgets["buffer_km"].setText("20.0")
            elif process == "Catalog Build":
                self.arg_widgets["input_dir"].setText("cmip6_data")
                self.arg_widgets["output_dir"].setText("metadata")

    def set_theme(self, name: str, checked: bool = True):
        self.current_theme = name.lower()
        apply_theme(QApplication.instance(),
                    name=self.current_theme,
                    base_pt=self.current_font_base,
                    small_pt=max(self.current_font_base - 2, 8))
        for act in self.theme_menu.actions():
            act.setChecked(act.text().lower() == self.current_theme)

    def set_font_size(self, size_pt: int):
        if size_pt == self.current_font_base:
            return
        self.current_font_base = size_pt
        for sz, act in self.font_size_actions.items():
            act.setChecked(sz == size_pt)
        apply_theme(QApplication.instance(),
                    name=self.current_theme,
                    base_pt=size_pt,
                    small_pt=max(size_pt - 2, 8))

    def browse_file(self, line_edit: QLineEdit, is_dir=False):
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "Choose directory")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file")
        if p:
            line_edit.setText(p)

    def start_task(self) -> None:
        if self.thread_manager.is_running():
            self.log_text.append("❗ A task is already running")
            QMessageBox.warning(self, "Task Running", "A task is already running. Please stop it before starting a new one.")
            return

        args_dict: Dict[str, object] = {}
        int_fields = ["timeout", "retries", "max_downloads", "workers"]
        float_fields = ["min_lat", "max_lat", "min_lon", "max_lon", "buffer_km"]
        for key, w in self.arg_widgets.items():
            if isinstance(w, QLineEdit):
                val = w.text().strip()
                if key in int_fields and val:
                    try:
                        args_dict[key] = int(val)
                        w.setStyleSheet("")
                    except ValueError:
                        w.setStyleSheet("border:2px solid red;")
                        self.log_text.append(f"❗ Invalid {key} value: {val}. Must be an integer.")
                        QMessageBox.critical(self, "Invalid Input", f"Invalid {key} value: {val}. Must be an integer.")
                        return
                elif key in float_fields and val:
                    try:
                        args_dict[key] = float(val)
                        w.setStyleSheet("")
                    except ValueError:
                        w.setStyleSheet("border:2px solid red;")
                        self.log_text.append(f"❗ Invalid {key} value: {val}. Must be a number.")
                        QMessageBox.critical(self, "Invalid Input", f"Invalid {key} value: {val}. Must be a number.")
                        return
                else:
                    args_dict[key] = val or None
            elif isinstance(w, QCheckBox):
                args_dict[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                args_dict[key] = w.currentText()

        output_dir = args_dict.get("output_dir")
        if output_dir and os.path.exists(output_dir) and os.listdir(output_dir):
            resp = QMessageBox.warning(
                self, "Output Directory Not Empty",
                f"The output directory '{output_dir}' contains files. Existing files may be overwritten.\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if resp == QMessageBox.No:
                return

        if self.src_combo.currentText() in ["CMIP6", "CMIP5"] and self.proc_combo.currentText() == "Download":
            args_dict["id"] = args_dict.pop("username", None)
            args_dict["password"] = args_dict.pop("password", None)
            if self.src_combo.currentText() == "CMIP5":
                args_dict["openid"] = args_dict.pop("open_id", None)
                args_dict["time_frequency"] = args_dict.pop("time_frequency", None)
            if not any([args_dict["id"], args_dict["password"], args_dict.get("openid", None)]):
                project = self.src_combo.currentText()
                resp = QMessageBox.warning(
                    self, "Missing Authentication",
                    f"No authentication credentials provided. Downloads may fail for restricted {project} data.\n"
                    "Continue without credentials?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if resp == QMessageBox.No:
                    return

        if self.src_combo.currentText() == "CMIP6" and self.proc_combo.currentText() == "Download":
            required_fields = ["project", "activity", "variable", "frequency", "resolution"]
            for field in required_fields:
                if not args_dict.get(field):
                    self.log_text.append(f"❗ {field.title()} is required for CMIP6 Download")
                    QMessageBox.critical(self, "Missing Input", f"{field.title()} is required for CMIP6 Download")
                    return

        if self.src_combo.currentText() == "CMIP5" and self.proc_combo.currentText() == "Download":
            required_fields = ["project", "model", "variable", "time_frequency"]
            for field in required_fields:
                if not args_dict.get(field):
                    self.log_text.append(f"❗ {field.title()} is required for CMIP5 Download")
                    QMessageBox.critical(self, "Missing Input", f"{field.title()} is required for CMIP5 Download")
                    return

        if self.src_combo.currentText() == "PRISM" and self.proc_combo.currentText() == "Download":
            if not args_dict.get("start_date") or not args_dict.get("end_date"):
                self.log_text.append("❗ Start Date and End Date are required for PRISM Download")
                QMessageBox.critical(self, "Missing Input", "Start Date and End Date are required for PRISM Download")
                return
            if not args_dict.get("variable") or not args_dict.get("resolution") or not args_dict.get("time_step"):
                self.log_text.append("❗ Variable, Resolution, and Time Step are required for PRISM Download")
                QMessageBox.critical(self, "Missing Input", "Variable, Resolution, and Time Step are required for PRISM Download")
                return

        if self.proc_combo.currentText() == "Spatial Crop":
            for field in ["input_dir", "output_dir", "min_lat", "max_lat", "min_lon", "max_lon"]:
                if not args_dict.get(field):
                    self.log_text.append(f"❗ {field.replace('_', ' ').title()} is required for Spatial Crop")
                    QMessageBox.critical(self, "Missing Input", f"{field.replace('_', ' ').title()} is required for Spatial Crop")
                    return

        if self.proc_combo.currentText() == "Spatial Clip":
            if not args_dict.get("shapefile_path"):
                self.log_text.append("❗ Shapefile Path is required for Spatial Clip")
                QMessageBox.critical(self, "Missing Input", "Shapefile Path is required for Spatial Clip")
                return

        if self.proc_combo.currentText() == "Catalog Build":
            for field in ["input_dir", "output_dir"]:
                if not args_dict.get(field):
                    self.log_text.append(f"❗ {field.replace('_', ' ').title()} is required for Catalog Build")
                    QMessageBox.critical(self, "Missing Input", f"{field.replace('_', ' ').title()} is required for Catalog Build")
                    return

        args_dict["workers"] = args_dict.get("workers", self.workers_edit.text().strip())
        if args_dict["workers"]:
            try:
                args_dict["workers"] = int(args_dict["workers"])
            except ValueError:
                self.log_text.append(f"❗ Invalid workers value: {args_dict['workers']}. Must be a number.")
                QMessageBox.critical(self, "Invalid Input", f"Invalid workers value: {args_dict['workers']}. Must be a number.")
                return
        args_dict["log_level"] = self.verbosity_combo.currentText()
        args_dict["demo"] = False
        args_dict["config"] = None
        args_dict["test"] = False
        args_dict["dry_run"] = False
        args_dict["id"] = args_dict.get("id", None)
        args_dict["password"] = args_dict.get("password", None)
        args_dict["openid"] = args_dict.get("openid", None)
        args_dict["no_verify_ssl"] = args_dict.get("no_verify_ssl", False)
        args_dict["retry_failed"] = args_dict.get("retry_failed", None)
        args_dict["output_dir"] = args_dict.get("output_dir", None)
        args_dict["metadata_dir"] = args_dict.get("metadata_dir", None)
        args_dict["save_mode"] = args_dict.get("save_mode", "flat")
        args_dict["latest"] = args_dict.get("latest", True)
        args_dict["extra_params"] = args_dict.get("extra_params", None)

        class _Args:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        cli_args = _Args(**args_dict)

        src, proc = self.src_combo.currentText(), self.proc_combo.currentText()
        dispatch = {
            ("CMIP6", "Download"): download_command,
            ("CMIP5", "Download"): download_cmip5_command,
            ("PRISM", "Download"): download_prism_command,
            ("*", "Spatial Crop"): crop_command,
            ("*", "Spatial Clip"): clip_command,
            ("*", "Catalog Build"): catalog_command,
        }
        func = dispatch.get((src, proc), dispatch.get(("*", proc)))

        def worker_wrapper(shutdown_event: Event):
            try:
                cli_args.stop_event = shutdown_event
                cli_args.stop_flag = shutdown_event.is_set
                # Periodically check shutdown_event
                while not shutdown_event.is_set():
                    func(cli_args)
                    # Break the loop if the task is complete or stopped
                    if not cli_args.stop_event.is_set():
                        self.task_completed.emit(True)
                        break
                    else:
                        self.log_signal.emit("Worker stopped due to shutdown signal")
                        break
                if shutdown_event.is_set():
                    self.log_signal.emit("Worker aborted due to shutdown signal")
                    self.task_completed.emit(False)
            except Exception as exc:
                self.log_signal.emit(f"{type(exc).__name__}: {exc}")
                self.task_completed.emit(False)

        self.thread_manager.add_worker(worker_wrapper, worker_name=f"{src}_{proc}_Worker")
        self.task_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.append(f"▶ Starting {src} — {proc} …")

    def stop_task(self) -> None:
        if not self.thread_manager.is_running():
            self.log_text.append("❗ No task is running")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return

        resp = QMessageBox.question(
            self, "Confirm Stop",
            "Are you sure you want to stop the current task?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if resp == QMessageBox.Yes:
            self.log_text.append("⏹ Sending stop signal to download tasks...")
            self.stopping.emit()
            self.thread_manager.stop()
            # Wait briefly to ensure threads have stopped
            import time
            timeout = 5  # seconds
            start_time = time.time()
            while self.thread_manager.is_running() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            if self.thread_manager.is_running():
                self.log_text.append("⚠ Some tasks could not be stopped within timeout")
            self.task_running = False
            self.stopped.emit()

    def on_stopping(self):
        self.log_text.append("📡 Sending stop signal to task... Please wait...")
        self.stop_btn.setEnabled(True)

    def on_task_stopped(self):
        self.log_text.append("✅ Task stopped successfully")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        QMessageBox.information(self, "Task Stopped", "The task has been stopped successfully.")

    def on_log_message(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        if "ERROR" in msg or "CRITICAL" in msg:
            if "Permission denied" in msg:
                QMessageBox.critical(self, "Permission Error", f"Permission denied: {msg}\nRun the GUI as Administrator or check directory permissions.")
            elif "No NetCDF files found" in msg:
                QMessageBox.critical(self, "No Files Found", f"{msg}\nRun a Download task to generate files in the input directory.")
            else:
                QMessageBox.critical(self, "Error", msg)

    def on_progress_update(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_task_completed(self, ok: bool):
        self.task_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.append("✅ Task completed" if ok else "❌ Task failed")

    def show_about(self):
        QMessageBox.about(self, "About GridFlow", ABOUT_DIALOG_HTML)

    def closeEvent(self, event):
        if self.thread_manager.is_running():
            self.thread_manager.stop()
        splitter = self.centralWidget().findChild(QSplitter)
        if splitter:
            sizes = splitter.sizes()
            QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue("split_sizes", sizes)
        super().closeEvent(event)

def main():
    import ctypes
    if sys.platform == "win32":
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # Per-Monitor DPI Aware
    os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    wnd = GridFlowGUI()
    wnd.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()