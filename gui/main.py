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

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QPushButton, QComboBox, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QAction, QTextEdit, QSizePolicy, QMenu,
    QScrollArea, QGraphicsOpacityEffect, QSplitter, QCompleter
)
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QFontMetrics
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer, QObject, QPropertyAnimation, QEasingCurve

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
    "Beginner": (4, "minimal"),
    "Advanced": (25, "debug"),
}

def mk_label(text: str, indent: int = 0, required: bool = False) -> QLabel:
    """Return a label that is left-aligned and shares a common width, with optional indentation and required marker."""
    lbl = QLabel(text)
    lbl.setMinimumWidth(LABEL_COL + indent)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    style = f"padding-left: {indent}px;" if indent > 0 else ""
    lbl.setStyleSheet(style)
    return lbl

# ----------------------------------------------------------------------
#  Modern theming helper – full replacement
# ----------------------------------------------------------------------
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
        progress_bg        = "#3a3f5c"
        progress_text      = "#ffffff"
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
        progress_bg        = "#d5d2c1"
        progress_text      = "#ffffff"
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
        progress_bg        = "#c7e9ff"
        progress_text      = "#ffffff"
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
        progress_bg        = "#d8dee9"
        progress_text      = "#1a1a1a"
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
        font-weight:bold;
        background:{progress_bg};
        border:1px solid {border};
        border-radius:8px;
        color:{progress_text};
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
        margin:0;
        background:transparent;
    }}
    QScrollBar::handle:vertical {{
        border-radius:6px;
        background:{accent};
        min-height:24px;
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
        background:{halo_bg};
    }}

    /* no extra coloring on the “page” areas */
    QScrollBar::sub-page:vertical,
    QScrollBar::add-page:vertical {{
        background:transparent;
    }}

    """)

# --------------------------- constants ---------------------------
LOGO_SIZE = (250, 100)
COPYRIGHT_TEXT = "© 2025 Bhuwan Shah  |  GridFlow"
ABOUT_DIALOG_HTML = (
    "<h2>GridFlow</h2>"
    "<p>Graphical front-end for GridFlow Library.</p>"
    "<p>Copyright © 2025 Bhuwan Shah<br>"
    "Released under the AGPL-v3 licence.</p>"
    "<p><b>GitHub:</b> <a href='https://github.com/shahbhuwan'>https://github.com/shahbhuwan/GridFlow</a><br>"
    "<b>Email:</b> <a href='mailto:bshah@iastate.edu'>bshah@iastate.edu</a></p>"
)
LABEL_COL = 120
COMMON_VARIABLES = ["pr", "tas", "tasmax", "tasmin", "hurs", "huss"]

# --------------------------- worker thread ---------------------------
class WorkerThread(QThread):
    log_message     = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    task_completed  = pyqtSignal(bool)
    error_occurred  = pyqtSignal(str)
    stopping        = pyqtSignal()
    stopped         = pyqtSignal()

    def __init__(self, command_func: Callable, args: Any, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.command_func = command_func
        self.args         = args
        self.stop_event   = Event()
        self.is_stopping  = False
        self.force_stop   = False

    def run(self):
        try:
            self.args.stop_event = self.stop_event
            self.args.stop_flag  = self.stop_event.is_set
            self.command_func(self.args)
            self.task_completed.emit(True)
        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
            self.task_completed.emit(False)
        finally:
            if self.is_stopping:
                self.stopped.emit()

    def stop(self, force: bool = False) -> None:
        if not self.isRunning():
            return
        self.is_stopping = True
        self.force_stop  = force
        self.stop_event.set()
        self.stopping.emit()
        grace_ms = 2000 if force else 10000
        if self.wait(grace_ms):
            return
        if force:
            logging.warning("Forcing thread termination")
            self.terminate()
            self.wait()
        else:
            logging.warning("Task still running; call stop(force=True) to kill.")

# --------------------------- Qt‑logging bridge ---------------------------
class QtHandler(logging.Handler):
    def __init__(self, log_signal, progress_signal):
        super().__init__()
        self.log_signal = log_signal
        self.progress_signal = progress_signal
        self.success_regex = re.compile(r"Successfully downloaded: (.+)")
        self.progress_regex = re.compile(r"Progress: (\d+)/(\d+) files")
        self.completed_regex = re.compile(r"Completed: (\d+)/(\d+) files")

    def emit(self, record):
        msg = self.format(record)
        # Apply clean log filter only in minimal mode (logging.INFO)
        if self.level == logging.INFO:
            success_match = self.success_regex.search(msg)
            if success_match:
                filename = success_match.group(1)
                self.log_signal.emit(f"Downloaded {filename}")
        else:
            # In normal, verbose, debug modes, emit all messages
            self.log_signal.emit(msg)
        # Always process progress updates
        progress_match = self.progress_regex.search(msg) or self.completed_regex.search(msg)
        if progress_match:
            current, total = map(int, progress_match.groups())
            self.progress_signal.emit(current, total)

# --------------------------- main window ---------------------------
class GridFlowGUI(QMainWindow):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, base_pt: int = 14):
        super().__init__()

        VOCAB_DIR = Path(__file__).parent / "vocab"

        def _load(name):
            if getattr(sys, 'frozen', False):
                base_path = os.path.join(sys._MEIPASS, 'gui', 'vocab')
            else:
                base_path = Path(__file__).parent / 'vocab'
            f = Path(base_path) / name
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
        self.cmip6_resolution      = _load("cmip6_resolution.json")

        # CMIP5 vocab
        self.cmip5_model           = _load("cmip5_model.json")
        self.cmip5_experiment      = _load("cmip5_experiment.json")
        self.cmip5_variable        = _load("cmip5_variable.json")
        self.cmip5_time_frequency  = _load("cmip5_time_frequency.json")
        self.cmip5_ensemble        = _load("cmip5_ensemble.json")
        self.cmip5_institute       = _load("cmip5_institute.json")

        self.setWindowTitle("GridFlow Data Processor")
        self.worker_thread: Optional[WorkerThread] = None
        self.current_font_base = base_pt

        self.init_ui()
        self.init_logging()

        # Set default window size to half screen width and 80% height
        screen = QApplication.primaryScreen().size()
        width = screen.width() // 2
        height = int(screen.height() * 0.8)
        self.resize(width, height)

    def init_logging(self):
        log_dir = Path("C:/GridFlow/Logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir, "minimal", prefix="gridflow_")
        self.log_signal.connect(self.on_log_message)
        self.progress_signal.connect(self.on_progress_update)
        self.qt_handler = QtHandler(self.log_signal, self.progress_signal)
        self.qt_handler.setFormatter(logging.Formatter("%(message)s"))
        self.qt_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.qt_handler)

    def init_ui(self):
        scr = QApplication.primaryScreen().size()
        small_screen = scr.width() < 1280
        margin = 8 if small_screen else 12
        spacing = 5 if small_screen else 8
        base_logo_height = 80 if small_screen else 120
        logo_height = int(base_logo_height * (scr.height() / 1080))

        container = QWidget()
        vmain = QVBoxLayout(container)
        vmain.setSpacing(spacing)
        vmain.setContentsMargins(margin, margin, margin, margin)
        self.setCentralWidget(container)

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
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        top_scroll = QScrollArea()
        top_scroll.setWidgetResizable(True)
        top_scroll.setStyleSheet("QScrollArea { border: none; }")
        splitter.addWidget(top_scroll)

        upper = QWidget()
        ulay = QVBoxLayout(upper)
        ulay.setSpacing(spacing)
        ulay.setContentsMargins(0, 0, 0, 0)
        ulay.setStretch(0, 1)
        ulay.setStretch(1, 8)
        ulay.setStretch(2, 1)
        top_scroll.setWidget(upper)

        header = QWidget(objectName="card")
        h = QVBoxLayout(header)
        h.setContentsMargins(margin, margin, margin, margin)
        h.setAlignment(Qt.AlignHCenter)

        self.logo_lbl = QLabel(alignment=Qt.AlignCenter)
        self.logo_lbl.setScaledContents(True)
        self.logo_lbl.setMaximumHeight(logo_height)
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
        halo_l.setContentsMargins(10, 10, 10, 10)
        halo_l.addWidget(self.logo_lbl, 0, Qt.AlignCenter)
        h.addWidget(halo)

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

        form_card = QWidget(objectName="card")
        form_l = QVBoxLayout(form_card)
        form_l.setContentsMargins(margin, margin, margin, margin)

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
        self.form_layout.setVerticalSpacing(12)
        self.form_layout.setContentsMargins(margin, margin, margin, margin)
        form_content_l.addLayout(self.form_layout)
        self.arg_widgets = {}
        ulay.addWidget(form_card)

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
        self.stop_btn.setToolTip("Not yet Implemented")
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.setSpacing(15)
        ulay.addLayout(btn_row)

        self.start_btn.clicked.connect(self.start_task)
        self.stop_btn.clicked.connect(self.stop_task)

        self.progress_bar = QProgressBar()
        ulay.addWidget(self.progress_bar)

        log_container = QWidget(objectName="logCard")
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(10, 10, 10, 10)

        self.log_text = QTextEdit(readOnly=True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_container)

        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)

        total_height = 768
        saved = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value("split_sizes")
        if saved:
            splitter.setSizes([int(s) for s in saved])
        else:
            splitter.setSizes([int(total_height * 0.7), int(total_height * 0.3)])

        command_wrapper = QWidget()
        command_layout = QHBoxLayout(command_wrapper)
        command_layout.setContentsMargins(0, 0, 0, 0)
        command_layout.setSpacing(6)

        fm = QFontMetrics(self.font())  
        min_height = fm.height() * 2.2

        self.command_display = QTextEdit()
        self.command_display.setReadOnly(True)
        self.command_display.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.command_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.command_display.setMinimumHeight(min_height)
        self.command_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.command_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.command_display.setVisible(False)
        self.command_display.setStyleSheet("""
            QTextEdit {
                font-size: 11pt;
                color: #fff;
                background: #4a90e2;
                padding: 6px;
                border-radius: 4px;
                border: none;
            }
        """)
        command_layout.addWidget(self.command_display, 1)

        self.copy_button = QPushButton("Copy Command")
        self.copy_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.copy_button.setMinimumHeight(min_height)
        self.copy_button.setStyleSheet("""
            QPushButton {
                font-size: 11pt;
                background: #006494;
                color: #fff;
                padding: 8px 12px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover   { background: #0a77b6; }
            QPushButton:pressed { background: #00517a; }
        """)
        self.copy_button.setVisible(False)
        self.copy_button.clicked.connect(self.copy_command)
        command_layout.addWidget(self.copy_button, 0)

        splitter.addWidget(command_wrapper)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(2, False)

        footer = QLabel(COPYRIGHT_TEXT, alignment=Qt.AlignCenter)
        footer.setStyleSheet("color: rgba(0,0,0,0.7); font-size: 10pt; margin-top: 4px;")
        vmain.addWidget(footer)

        total_h = QApplication.primaryScreen().size().height()
        splitter.setSizes([int(total_h*0.70), int(total_h*0.25), command_wrapper.sizeHint().height()])

        about_act = QAction("&About GridFlow..", self)
        about_act.triggered.connect(self.show_about)
        self.menuBar().addMenu("&About").addAction(about_act)

        self.theme_menu = self.menuBar().addMenu("&Theme")
        for t in ("Default", "Cosmic", "Sand", "Ocean"):
            act = QAction(t, self, checkable=True)
            act.triggered.connect(partial(self.set_theme, t.lower()))
            self.theme_menu.addAction(act)
        self.theme_menu.actions()[3].setChecked(True)

        self.current_theme = "ocean"

        view_menu = self.menuBar().addMenu("&View")
        font_menu = QMenu("Font &Size  aA", self)
        view_menu.addMenu(font_menu)
        self.font_size_actions = {}
        for label, size in [("Small", 12), ("Medium", 14), ("Large", 16)]:
            act = QAction(label, self, checkable=True)
            act.triggered.connect(lambda ch=False, s=size: self.set_font_size(s))
            font_menu.addAction(act)
            self.font_size_actions[size] = act

        if small_screen:
            self.set_font_size(12)
        else:
            self.set_font_size(self.current_font_base)
        self.font_size_actions[self.current_font_base].setChecked(True)

        apply_theme(QApplication.instance(),
                    name=self.current_theme,
                    base_pt=self.current_font_base,
                    small_pt=max(self.current_font_base - 2, 8))

        self.update_form(self.proc_combo.currentText())
        QTimer.singleShot(500, self.maybe_show_tutorial)
        self.skill_combo.setCurrentText("Beginner")
        QTimer.singleShot(0, self.logo_anim.start)

    def on_skill_change(self, level: str):
        workers, verb = PRESETS[level]
        self.workers_edit.setReadOnly(level != "Advanced")
        self.workers_edit.setText(str(workers))
        self.verbosity_combo.setCurrentText(verb)
        self.update_form(self.proc_combo.currentText())

    def update_form(self, process: str) -> None:
            while self.form_layout.rowCount():
                self.form_layout.removeRow(0)
            self.arg_widgets.clear()

            src = self.src_combo.currentText()
            skill_level = self.skill_combo.currentText()

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
                    # Required Fields
                    proj_le = QLineEdit("CMIP6")
                    proj_le.setReadOnly(True)
                    proj_le.setEnabled(False)
                    self.form_layout.addRow(mk_label("Project", indent_amount, required=True), proj_le)
                    self.arg_widgets["project"] = proj_le

                    add_line("Activity", "ScenarioMIP", "The CMIP6 activity or program (e.g., ScenarioMIP, CMIP).",
                            indent=indent_amount, vocab=self.cmip6_activity_id, required=True)
                    
                    if skill_level == "Beginner":
                        add_combo("Variable", COMMON_VARIABLES, "tas", "Select a common climate variable.",
                                indent=indent_amount)
                    else:
                        add_line("Variable", "tas", "The climate variable to download (e.g., tas for temperature).",
                                indent=indent_amount, vocab=self.cmip6_variable_id, required=True)

                    add_line("Frequency", "mon", "The temporal frequency of the data (e.g., mon, day).",
                            indent=indent_amount, vocab=self.cmip6_frequency, required=True)
                    add_line("Resolution", "100 km", "The nominal resolution of the data (e.g., 100 km).",
                            indent=indent_amount, vocab=self.cmip6_resolution, required=True)
                    add_file("Output", "./Output", "The directory where downloaded NetCDF files will be saved.",
                            dir_=True, indent=indent_amount, required=True)
                    add_file("Metadata", "./Metadata", "The directory where metadata JSON files will be saved.",
                            dir_=True, indent=indent_amount, required=True)
                    add_file("Logs", "./Logs", "The directory where log files will be saved.",
                            dir_=True, indent=indent_amount, required=True)

                    # Optional Fields (Advanced mode only)
                    if skill_level == "Advanced":
                        add_line("Experiment", "ssp585", "The experiment identifier (e.g., ssp585, historical).",
                                indent=indent_amount, vocab=self.cmip6_experiment_id)
                        add_line("Model", "", "The climate model/source ID (e.g., CanESM5).",
                                indent=indent_amount, vocab=self.cmip6_source_id)
                        add_line("Ensemble", "r1i1p1f1", "The ensemble member or variant label (e.g., r1i1p1f1).",
                                indent=indent_amount, vocab=self.cmip6_variant_label)
                        add_line("Institution", "", "The institution responsible for the model (e.g., CCCma).",
                                indent=indent_amount, vocab=self.cmip6_institution_id)
                        add_line("Source Type", "", "The type of model (e.g., AOGCM).",
                                indent=indent_amount, vocab=self.cmip6_source_type)
                        add_line("Grid Label", "", "The grid label for the dataset (e.g., gn).",
                                indent=indent_amount, vocab=self.cmip6_grid_label)
                        add_line("Start Date", "", "Filter data starting from this date (format: YYYY-MM-DD).",
                                indent=indent_amount)
                        add_line("End Date", "", "Filter data up to this date (format: YYYY-MM-DD).",
                                indent=indent_amount)
                        add_chk("Latest", True, "If checked, downloads only the latest version of each file.",
                                indent=indent_amount)
                        add_line("Extra Params", "", "Additional query parameters in JSON format (e.g., {\"key\": \"value\"}).",
                                indent=indent_amount)
                        add_combo("Save Mode", ["Flat", "Structured"], "Flat", "Choose 'Flat' for a single directory or 'Structured' for a hierarchical organization.",
                                indent=indent_amount)
                        add_line("Max Downloads", "", "The maximum number of files to download (leave blank for no limit).",
                                indent=indent_amount)

                        adv_chk = QCheckBox("Show advanced options")
                        adv_chk.setObjectName("largeCheckbox")
                        adv_chk.setToolTip("Show additional options for retries, timeouts, and authentication.")
                        self.form_layout.addRow(mk_label("", indent_amount), adv_chk)

                        adv_widget = QWidget()
                        adv_layout = QFormLayout(adv_widget)
                        adv_layout.setContentsMargins(0, 0, 0, 0)

                        retries_le = QLineEdit("3")
                        retries_le.setToolTip("The number of times to retry a failed download (default: 3).")
                        adv_layout.addRow(mk_label("Retries", indent_amount), retries_le)

                        timeout_le = QLineEdit("30")
                        timeout_le.setToolTip("The HTTP request timeout in seconds (default: 30).")
                        adv_layout.addRow(mk_label("Timeout", indent_amount), timeout_le)

                        no_ssl_ck = QCheckBox()
                        no_ssl_ck.setObjectName("largeCheckbox")
                        no_ssl_ck.setChecked(False)
                        no_ssl_ck.setToolTip("Disable SSL verification for downloads (not recommended).")
                        adv_layout.addRow(mk_label("No Verify SSL", indent_amount), no_ssl_ck)

                        user_le = QLineEdit("")
                        user_le.setToolTip("Username for authenticated downloads (if required).")
                        adv_layout.addRow(mk_label("Username", indent_amount), user_le)

                        pass_le = QLineEdit("")
                        pass_le.setToolTip("Password for authenticated downloads (if required).")
                        adv_layout.addRow(mk_label("Password", indent_amount), pass_le)

                        retry_file = QLineEdit("")
                        retry_file.setToolTip("Path to a '.json' file listing failed downloads to retry.")
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
                    # Required Fields
                    proj_le = QLineEdit("CMIP5")
                    proj_le.setReadOnly(True)
                    proj_le.setEnabled(False)
                    self.form_layout.addRow(mk_label("Project", indent_amount, required=True), proj_le)
                    self.arg_widgets["project"] = proj_le

                    if skill_level == "Beginner":
                        add_combo("Variable", COMMON_VARIABLES, "tas", "Select a common climate variable.",
                                indent=indent_amount)
                    else:
                        add_line("Variable", "tas", "The climate variable to download (e.g., tas for temperature).",
                                indent=indent_amount, vocab=self.cmip5_variable, required=True)

                    add_line("Frequency", "mon", "The temporal frequency of the data (e.g., mon, day).",
                            indent=indent_amount, vocab=self.cmip5_time_frequency, required=True)
                    add_line("Model", "CanESM2", "The CMIP5 model name (e.g., CanESM2).",
                            indent=indent_amount, vocab=self.cmip5_model, required=True)
                    add_file("Output", "./Output", "The directory where downloaded NetCDF files will be saved.",
                            dir_=True, indent=indent_amount, required=True)
                    add_file("Metadata", "./Metadata", "The directory where metadata JSON files will be saved.",
                            dir_=True, indent=indent_amount, required=True)
                    add_file("Logs", "./Logs", "The directory where log files will be saved.",
                            dir_=True, indent=indent_amount, required=True)

                    # Optional Fields (Advanced mode only)
                    if skill_level == "Advanced":
                        add_line("Experiment", "historical", "The experiment identifier (e.g., historical, rcp45).",
                                indent=indent_amount, vocab=self.cmip5_experiment)
                        add_line("Ensemble", "r1i1p1", "The ensemble member (e.g., r1i1p1).",
                                indent=indent_amount, vocab=self.cmip5_ensemble)
                        add_line("Institute", "", "The institute responsible for the model (e.g., CCCma).",
                                indent=indent_amount, vocab=self.cmip5_institute)
                        add_line("Start Date", "", "Filter data starting from this date (format: YYYY-MM-DD).",
                                indent=indent_amount)
                        add_line("End Date", "", "Filter data up to this date (format: YYYY-MM-DD).",
                                indent=indent_amount)
                        add_chk("Latest", True, "If checked, downloads only the latest version of each file.",
                                indent=indent_amount)
                        add_line("Extra Params", "", "Additional query parameters in JSON format (e.g., {\"key\": \"value\"}).",
                                indent=indent_amount)
                        add_combo("Save Mode", ["Flat", "Structured"], "Flat", "Choose 'Flat' for a single directory or 'Structured' for a hierarchical organization.",
                                indent=indent_amount)
                        add_line("Max Downloads", "", "The maximum number of files to download (leave blank for no limit).",
                                indent=indent_amount)

                        adv_chk = QCheckBox("Show advanced options")
                        adv_chk.setObjectName("largeCheckbox")
                        adv_chk.setToolTip("Show additional options for retries, timeouts, and authentication.")
                        self.form_layout.addRow(mk_label("", indent_amount), adv_chk)

                        adv_widget = QWidget()
                        adv_layout = QFormLayout(adv_widget)
                        adv_layout.setContentsMargins(0, 0, 0, 0)

                        retries_le = QLineEdit("3")
                        retries_le.setToolTip("The number of times to retry a failed download (default: 3).")
                        adv_layout.addRow(mk_label("Retries", indent_amount), retries_le)

                        timeout_le = QLineEdit("30")
                        timeout_le.setToolTip("The HTTP request timeout in seconds (default: 30).")
                        adv_layout.addRow(mk_label("Timeout", indent_amount), timeout_le)

                        user_le = QLineEdit("")
                        user_le.setToolTip("Username for authenticated downloads (if required).")
                        adv_layout.addRow(mk_label("Username", indent_amount), user_le)

                        pass_le = QLineEdit("")
                        pass_le.setToolTip("Password for authenticated downloads (if required).")
                        adv_layout.addRow(mk_label("Password", indent_amount), pass_le)

                        openid_le = QLineEdit("")
                        openid_le.setToolTip("OpenID URL for CMIP5 authentication (if required).")
                        adv_layout.addRow(mk_label("OpenID", indent_amount), openid_le)

                        no_ssl_ck = QCheckBox()
                        no_ssl_ck.setObjectName("largeCheckbox")
                        no_ssl_ck.setChecked(False)
                        no_ssl_ck.setToolTip("Disable SSL verification for downloads (not recommended).")
                        adv_layout.addRow(mk_label("No Verify SSL", indent_amount), no_ssl_ck)

                        retry_file = QLineEdit("")
                        retry_file.setToolTip("Path to a '.json' file listing failed downloads to retry.")
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
                    # Required Fields
                    add_combo("Variable", ["ppt", "tmax", "tmin", "tmean", "tdmean", "vpdmin", "vpdmax"], "tmean",
                            "The PRISM climate variable (e.g., tmean for mean temperature).", indent=indent_amount)
                    add_combo("Resolution", ["4km", "800m"], "4km", "The spatial resolution of PRISM data (4km or 800m).",
                            indent=indent_amount)
                    add_combo("Time Step", ["daily", "monthly"], "daily", "The temporal resolution (daily or monthly).",
                            indent=indent_amount)
                    add_line("Start Date", "2020-01-01", "The start date for PRISM data (format: YYYY-MM-DD or YYYY-MM).",
                            indent=indent_amount, required=True)
                    add_line("End Date", "2020-01-04", "The end date for PRISM data (format: YYYY-MM-DD or YYYY-MM).",
                            indent=indent_amount, required=True)
                    add_file("Output", "./Output", "The directory where downloaded PRISM files will be saved.",
                            dir_=True, indent=indent_amount, required=True)
                    add_file("Metadata", "./Metadata", "The directory where PRISM metadata files will be saved.",
                            dir_=True, indent=indent_amount, required=True)
                    add_file("Logs", "./Logs", "The directory where log files will be saved.",
                            dir_=True, indent=indent_amount, required=True)

                    # Optional Fields
                    add_line("Retries", "3", "The number of times to retry a failed download (default: 3).",
                            indent=indent_amount)
                    add_line("Timeout", "30", "The HTTP request timeout in seconds (default: 30).",
                            indent=indent_amount)

            elif process == "Spatial Crop":
                # Required Fields
                add_file("Input", "./Output", "The directory containing NetCDF files to crop.",
                        dir_=True, indent=indent_amount, required=True)
                add_file("Output", "./Output/Cropped", "The directory to save cropped NetCDF files.",
                        dir_=True, indent=indent_amount, required=True)
                add_file("Logs", "./Logs", "The directory where log files will be saved.",
                        dir_=True, indent=indent_amount, required=True)
                add_line("Min Lat", "35.0", "The minimum latitude for cropping (degrees).",
                        indent=indent_amount, required=True)
                add_line("Max Lat", "45.0", "The maximum latitude for cropping (degrees).",
                        indent=indent_amount, required=True)
                add_line("Min Lon", "-105.0", "The minimum longitude for cropping (degrees).",
                        indent=indent_amount, required=True)
                add_line("Max Lon", "-95.0", "The maximum longitude for cropping (degrees).",
                        indent=indent_amount, required=True)

                # Optional Fields
                add_line("Buffer KM", "50.0", "The buffer distance in kilometers to add around the bounding box.",
                        indent=indent_amount)

            elif process == "Spatial Clip":
                # Required Fields
                add_file("Input", "./Output", "The directory containing NetCDF files to clip.",
                        dir_=True, indent=indent_amount, required=True)
                add_file("Shapefile", "", "The path to a shapefile (.shp) defining the clipping boundary.",
                        dir_=False, indent=indent_amount, required=True)
                add_file("Output", "./Output/Clipped", "The directory to save clipped NetCDF files.",
                        dir_=True, indent=indent_amount, required=True)
                add_file("Logs", "./Logs", "The directory where log files will be saved.",
                        dir_=True, indent=indent_amount, required=True)

                # Optional Fields
                add_line("Buffer KM", "20.0", "The buffer distance in kilometers to add around the shapefile.",
                        indent=indent_amount)

            elif process == "Catalog Build":
                # Required Fields
                add_file("Input", "./Output", "The directory containing NetCDF files to include in the catalog.",
                        dir_=True, indent=indent_amount, required=True)
                add_file("Output", "./Metadata", "The directory to save the generated catalog JSON file.",
                        dir_=True, indent=indent_amount, required=True)
                add_file("Logs", "./Logs", "The directory where log files will be saved.",
                        dir_=True, indent=indent_amount, required=True)

            if is_first_run() and process == self.proc_combo.currentText():
                if process == "Download":
                    if src == "CMIP6":
                        pass
                    elif src == "CMIP5":
                        pass
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
                    self.arg_widgets["input_dir"].setText("./Output")
                    self.arg_widgets["output_dir"].setText("./Metadata")

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

# Replace the entire start_task method (lines ~980-1095) with:
    def start_task(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_text.append("❗ A task is already running")
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
                        return
                elif key in float_fields and val:
                    try:
                        args_dict[key] = float(val)
                        w.setStyleSheet("")
                    except ValueError:
                        w.setStyleSheet("border:2px solid red;")
                        self.log_text.append(f"❗ Invalid {key} value: {val}. Must be a number.")
                        return
                else:
                    args_dict[key] = val or None
            elif isinstance(w, QCheckBox):
                args_dict[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                args_dict[key] = w.currentText()

        if self.src_combo.currentText() in ["CMIP6", "CMIP5"] and self.proc_combo.currentText() == "Download":
            args_dict["id"] = args_dict.pop("username", None)
            args_dict["password"] = args_dict.pop("password", None)
            if self.src_combo.currentText() == "CMIP5":
                args_dict["openid"] = args_dict.pop("open_id", None)
                args_dict["time_frequency"] = args_dict.pop("frequency", None)
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
                    return
            # Ensure optional fields are set to None if not provided
            optional_fields = [
                "experiment", "model", "ensemble", "institution", "source_type", "grid_label",
                "start_date", "end_date", "extra_params", "max_downloads", "retries", "timeout",
                "username", "password", "retry_failed", "save_mode", "latest", "no_verify_ssl"
            ]
            for field in optional_fields:
                args_dict.setdefault(field, None)

        if self.src_combo.currentText() == "CMIP5" and self.proc_combo.currentText() == "Download":
            required_fields = ["project", "model", "variable", "time_frequency"]
            for field in required_fields:
                if not args_dict.get(field):
                    self.log_text.append(f"❗ {field.title()} is required for CMIP5 Download")
                    return
            # Ensure optional fields are set to None if not provided
            optional_fields = [
                "experiment", "ensemble", "institute", "start_date", "end_date", "extra_params",
                "max_downloads", "retries", "timeout", "username", "password", "openid",
                "retry_failed", "save_mode", "latest", "no_verify_ssl"
            ]
            for field in optional_fields:
                args_dict.setdefault(field, None)

        if self.src_combo.currentText() == "PRISM" and self.proc_combo.currentText() == "Download":
            if not args_dict.get("start_date") or not args_dict.get("end_date"):
                self.log_text.append("❗ Start Date and End Date are required for PRISM Download")
                return
            if not args_dict.get("variable") or not args_dict.get("resolution") or not args_dict.get("time_step"):
                self.log_text.append("❗ Variable, Resolution, and Time Step are required for PRISM Download")
                return

        if self.proc_combo.currentText() == "Spatial Crop":
            for field in ["min_lat", "max_lat", "min_lon", "max_lon"]:
                if not args_dict.get(field):
                    self.log_text.append(f"❗ {field.replace('_', ' ').title()} is required for Spatial Crop")
                    return

        if self.proc_combo.currentText() == "Spatial Clip" and not args_dict.get("demo"):
            if not args_dict.get("shapefile"):
                self.log_text.append("❗ Shapefile Path is required for Spatial Clip unless in demo mode")
                return

        if self.proc_combo.currentText() == "Catalog Build":
            for field in ["input", "output"]:
                if not args_dict.get(field):
                    self.log_text.append(f"❗ {field.replace('_', ' ').title()} Dir is required for Catalog Build")
                    return

        args_dict["workers"] = args_dict.get("workers", self.workers_edit.text().strip())
        if args_dict["workers"]:
            try:
                args_dict["workers"] = int(args_dict["workers"])
            except ValueError:
                self.log_text.append(f"❗ Invalid workers value: {args_dict['workers']}. Must be a number.")
                return
        args_dict["log_level"] = self.verbosity_combo.currentText()
        args_dict["demo"] = is_first_run()
        args_dict["config"] = None
        args_dict["test"] = False
        args_dict["dry_run"] = False
        args_dict["id"] = args_dict.get("id", None)
        args_dict["password"] = args_dict.get("password", None)
        args_dict["openid"] = args_dict.get("openid", None)
        args_dict["no_verify_ssl"] = args_dict.get("no_verify_ssl", False)
        args_dict["retry_failed"] = args_dict.get("retry_failed", None)
        args_dict["output_dir"] = args_dict.get("output", "./Output")  # Map 'output' to 'output_dir'
        args_dict["metadata_dir"] = args_dict.get("metadata", "./Metadata")  # Map 'metadata' to 'metadata_dir'
        args_dict["save_mode"] = args_dict.get("save_mode", "flat")
        args_dict["latest"] = args_dict.get("latest", True)
        args_dict["extra_params"] = args_dict.get("extra_params", None)

        src, proc = self.src_combo.currentText(), self.proc_combo.currentText()
        command_parts = ["gridflow"]

        process_map = {
            "Download": "download",
            "Spatial Crop": "crop",
            "Spatial Clip": "clip",
            "Catalog Build": "catalog"
        }
        subcommand = process_map.get(proc, proc.lower())
        command_parts.append(subcommand)

        if proc == "Download":
            if src in ["CMIP6", "CMIP5"]:
                for key in ["project", "activity", "experiment", "variable", "frequency", "model", "resolution",
                            "ensemble", "institution", "source_type", "grid_label", "start_date", "end_date",
                            "extra_params", "output", "metadata", "save_mode", "max_downloads", "retries",
                            "timeout", "id", "password", "openid", "retry_failed"]:
                    if key in args_dict and args_dict[key]:
                        command_parts.append(f"--{key.replace('_', '-')} \"{args_dict[key]}\"")
                if args_dict.get("latest"):
                    command_parts.append("--latest")
                if args_dict.get("no_verify_ssl"):
                    command_parts.append("--no-verify-ssl")
            else:  # PRISM
                for key in ["variable", "resolution", "time_step", "start_date", "end_date", "output",
                            "metadata", "retries", "timeout"]:
                    if key in args_dict and args_dict[key]:
                        command_parts.append(f"--{key.replace('_', '-')} \"{args_dict[key]}\"")
        elif proc == "Spatial Crop":
            for key in ["input", "output", "min_lat", "max_lat", "min_lon", "max_lon", "buffer_km"]:
                if key in args_dict and args_dict[key]:
                    command_parts.append(f"--{key.replace('_', '-')} \"{args_dict[key]}\"")
        elif proc == "Spatial Clip":
            for key in ["input", "shapefile", "buffer_km", "output"]:
                if key in args_dict and args_dict[key]:
                    command_parts.append(f"--{key.replace('_', '-')} \"{args_dict[key]}\"")
        elif proc == "Catalog Build":
            for key in ["input", "output"]:
                if key in args_dict and args_dict[key]:
                    command_parts.append(f"--{key.replace('_', '-')} \"{args_dict[key]}\"")

        if args_dict["workers"]:
            command_parts.append(f"--workers {args_dict['workers']}")
        if args_dict["log_level"]:
            command_parts.append(f"--log-level {args_dict['log_level']}")

        command_str = " ".join(command_parts)

        self.command_display.setText(f"Running: {command_str}")
        self.command_display.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffffff; background: #4a90e2; padding: 8px; border-radius: 4px; border: none;")
        self.command_display.setVisible(True)
        self.copy_button.setVisible(True)

        class _Args:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        cli_args = _Args(**args_dict)

        dispatch = {
            ("CMIP6", "Download"): download_command,
            ("CMIP5", "Download"): download_cmip5_command,
            ("PRISM", "Download"): download_prism_command,
            ("*", "Spatial Crop"): crop_command,
            ("*", "Spatial Clip"): clip_command,
            ("*", "Catalog Build"): catalog_command,
        }
        func = dispatch.get((src, proc), dispatch.get(("*", proc)))

        self.worker_thread = WorkerThread(func, cli_args)
        self.worker_thread.log_message.connect(self.on_log_message)
        self.worker_thread.progress_update.connect(self.on_progress_update)
        self.worker_thread.task_completed.connect(self.on_task_completed)
        self.worker_thread.stopped.connect(self.on_task_stopped)
        self.worker_thread.stopping.connect(self.on_stopping)
        self.worker_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.append(f"▶ Starting {src} — {proc} …")

    def stop_task(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            resp = QMessageBox.question(
                self, "Confirm Stop",
                "Are you sure you want to stop the current task?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                self.worker_thread.stop(force=False)
                self.log_text.append("⏹ Stopping task …")
                self.stop_btn.setText("Force Stop")
        else:
            self.log_text.append("❗ No task is running")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stop")

    def on_task_stopped(self):
        self.log_text.append("✅ Task stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.progress_bar.setValue(0)
        if self.skill_combo.currentText() == "Advanced":
            current_text = self.command_display.text()
            self.command_display.setText(current_text.replace("Running:", "Stopped:"))
            self.command_display.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffffff; background: #fd7e14; padding: 8px; border-radius: 4px; border: none;")
            self.command_display.setVisible(True)
            self.copy_button.setVisible(True)
        else:
            self.command_display.setVisible(False)
            self.copy_button.setVisible(False)

    def copy_command(self):
        command_text = self.command_display.toPlainText()
        command = command_text.split(":", 1)[1].strip() if ":" in command_text else command_text
        QApplication.clipboard().setText(command)
        self.log_text.append("📋 Command copied to clipboard")

    def on_stopping(self):
        if self.worker_thread and self.worker_thread.force_stop:
            self.stop_btn.setEnabled(False)
        else:
            self.stop_btn.setEnabled(True)

    def on_log_message(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_progress_update(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_task_completed(self, ok: bool):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.append("✅ Task completed" if ok else "❌ Task failed")
        if self.skill_combo.currentText() == "Advanced":
            current_text = self.command_display.toPlainText()
            if ok:
                self.command_display.setText(current_text.replace("Running:", "Completed:"))
                self.command_display.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffffff; background: #28a745; padding: 8px; border-radius: 4px; border: none;")
            else:
                self.command_display.setText(current_text.replace("Running:", "Failed:"))
                self.command_display.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffffff; background: #dc3545; padding: 8px; border-radius: 4px; border: none;")
            self.command_display.setVisible(True)
            self.copy_button.setVisible(True)
        else:
            self.command_display.setVisible(False)
            self.copy_button.setVisible(False)

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

        scr = QApplication.primaryScreen().size()
        base_width, base_height = 250, 100
        scale_factor = min(scr.width() / 1920, scr.height() / 1080)
        max_w = int(base_width * scale_factor)
        max_h = int(base_height * scale_factor)

        pm = QPixmap(str(logo_file))
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
        resp = QMessageBox.question(
            self, "Welcome to GridFlow",
            "I noticed it's your first time running GridFlow.\n"
            "Would you like to run a demo configuration?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if resp == QMessageBox.Yes:
            self.skill_combo.setCurrentText("Beginner")
            hdr = QLabel("CMIP6 core query", objectName="sectionHeader")
            hdr.setStyleSheet("font-weight:600; margin-top:12px; margin-bottom:4px;")
            self.form_layout.addRow(hdr)
            self.src_combo.setCurrentText("CMIP6")
            self.proc_combo.setCurrentText("Download")
            self.update_form("Download")
            self.arg_widgets["project"].setText("CMIP6")
            self.arg_widgets["activity"].setText("ScenarioMIP")
            self.arg_widgets["variable"].setCurrentText("tas")
            self.arg_widgets["frequency"].setText("mon")
            self.arg_widgets["resolution"].setText("100 km")
            self.arg_widgets["output_dir"].setText(str(Path(sys.executable).parent / "CMIP6" / "Download"))
            self.arg_widgets["metadata_dir"].setText(str(Path(sys.executable).parent / "Temp"))
            QMessageBox.information(
                self, "Demo Configuration",
                "The form has been pre-filled with a demo configuration for CMIP6 data (tas, ScenarioMIP).\n"
                "Click 'Start' to download files, or change the source/process to try another demo."
            )

    def show_about(self):
        QMessageBox.about(self, "About GridFlow", ABOUT_DIALOG_HTML)

    def closeEvent(self, event):
        splitter = self.centralWidget().findChild(QSplitter)
        if splitter:
            sizes = splitter.sizes()
            QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue("split_sizes", sizes)
        super().closeEvent(event)

def pick_base_font():
    scr = QApplication.primaryScreen().size()
    if scr.width() < 1280:
        return 11
    return 14

def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    base_pt = pick_base_font()
    wnd = GridFlowGUI(base_pt=base_pt)
    wnd.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()