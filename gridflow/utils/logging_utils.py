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
    A custom log filter that only allows specific INFO messages to pass,
    creating a less cluttered "minimal" output. All other levels (DEBUG,
    WARNING, ERROR, CRITICAL) are unaffected.
    """
    def filter(self, record):
        # Allow any record that is not at the INFO level to pass
        if record.levelno != logging.INFO:
            return True
        
        # For INFO level, only allow messages that match specific patterns
        msg = record.getMessage()
        return (
            msg.startswith('Progress:') or
            msg.startswith('Completed:') or
            msg.startswith('Downloaded') or
            msg.startswith('Found ') or
            msg.startswith('Retrying ') or
            msg.startswith('Querying node:') or
            msg.startswith('All nodes failed') or
            msg.startswith('Process finished') or
            msg.startswith('Execution was interrupted')
        )

def setup_logging(log_dir: str, level: str, prefix: str = "downloader") -> None:
    """
    Configures logging with three distinct levels: minimal, verbose, and debug.

    - minimal: Shows only key progress, completion, and error messages.
    - verbose: Shows all INFO level messages and above.
    - debug: Shows all messages, including detailed DEBUG statements.

    Args:
        log_dir: The directory where the log file will be saved.
        level: The desired logging level ('minimal', 'verbose', 'debug').
        prefix: A prefix for the log file name.
    """
    try:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir_path / f"{prefix}_{timestamp}.log"

        log_levels = {
            'minimal': logging.INFO,
            'verbose': logging.INFO,
            'debug': logging.DEBUG
        }
        numeric_level = log_levels.get(level.lower(), logging.INFO)

        # Get the root logger
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        # Clear existing handlers to prevent duplicate logs
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create handlers
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler(sys.stdout)

        # Set formatters
        # For 'debug', we use a more detailed format
        if level.lower() == 'debug':
            formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add filters if necessary
        if level.lower() == 'minimal':
            console_handler.addFilter(MinimalFilter())
            # File log for minimal should still be verbose
            file_handler.setLevel(logging.INFO) 
        else:
            # For verbose and debug, both handlers use the same level
            console_handler.setLevel(numeric_level)
            file_handler.setLevel(numeric_level)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.info(f"Logging initialized. Level: {level.upper()}. Log file: {log_file}")

    except Exception as e:
        # Fallback to basic logging if setup fails
        print(f"FATAL: Failed to initialize logging: {e}. Using basic console logging.", file=sys.stderr)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

