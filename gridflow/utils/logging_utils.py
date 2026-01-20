# Copyright (c) 2025 Bhuwan Shah
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import sys
from pathlib import Path
from datetime import datetime


class MinimalFilter(logging.Filter):
    """
    Show only selected high-level INFO lines on the console when level == 'minimal'.
    Non-INFO levels are handled by a separate filter in setup_logging.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return False  # INFO-only filter; non-INFO handled elsewhere
        msg = record.getMessage()
        return (
            # Success/Progress
            msg.startswith('Progress:') or
            msg.startswith('Node') or       
            msg.startswith('Parallel') or  
            msg.startswith('Completed:') or
            msg.startswith('Downloaded') or
            msg.startswith('Found ') or
            msg.startswith('Querying ') or
            msg.startswith('Process finished') or
            
            # Start/Stop
            msg.startswith('Running in demo') or
            msg.startswith('Example Command') or
            msg.startswith('Stop signal received!') or
            msg.startswith('Execution was interrupted') or
            
            # Failures (simple)
            msg.startswith('Failed:') or
            msg.startswith('All nodes failed')
        )


class SuppressWarnErrFilter(logging.Filter):
    """Hide WARNING/ERROR/CRITICAL on console for 'minimal' mode (still logged to file)."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def setup_logging(log_dir: str, level: str, prefix: str = "downloader") -> None:
    try:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir_path / f"{prefix}_{timestamp}.log"

        level_map = {'minimal': logging.INFO, 'verbose': logging.INFO, 'debug': logging.DEBUG}
        numeric_level = level_map.get(level.lower(), logging.INFO)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # FIX: Global logger must allow DEBUG for the file to see it

        # Clear existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        # --- HANDLER 1: FILE (Detailed) ---
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_fmt = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_fmt)
        
        # FIX: Force File Handler to ALWAYS be DEBUG.
        # It will now record success messages (DEBUG) and failures (WARNING) regardless of CLI mode.
        file_handler.setLevel(logging.DEBUG)

        # --- HANDLER 2: CONSOLE (Clean) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_fmt = logging.Formatter('%(message)s')  # Two spaces for indentation
        console_handler.setFormatter(console_fmt)

        if level.lower() == 'minimal':
            # Console is strict: INFO only, no Warnings (handled by file), specific whitelist
            console_handler.setLevel(logging.INFO)
            console_handler.addFilter(SuppressWarnErrFilter())
            console_handler.addFilter(MinimalFilter())
            
            # REMOVED: The line that downgraded file_handler to INFO was here.
            
        else:
            # In verbose/debug mode, console shows everything
            console_handler.setLevel(numeric_level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Silence noisy libraries
        if level.lower() == 'minimal':
            logging.getLogger("requests").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)

        # Initial log entry
        file_handler.handle(
            logging.LogRecord(name="root", level=logging.INFO, pathname=__file__, lineno=0,
                              msg=f"Logging initialized. Level: {level.upper()}", args=(), exc_info=None)
        )

    except Exception as e:
        print(f"FATAL: Failed to initialize logging: {e}", file=sys.stderr)


