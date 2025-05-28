# thread_stopper.py
import threading
import signal
import time
import os
import sys
from typing import Callable, List, Dict, Optional

class ThreadManager:
    def __init__(self, verbose: bool = True, shutdown_event: Optional[threading.Event] = None):
        self.threads: List[threading.Thread] = []
        self.thread_info: Dict[int, str] = {}
        self.shutdown_event = shutdown_event or threading.Event()
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()
        self.verbose = verbose
        self._setup_signal_handler()

    def add_worker(self, worker_function: Callable[[threading.Event], None], worker_name: str = "Worker") -> None:
        def wrapped_worker(shutdown_event: threading.Event):
            thread_id = threading.current_thread().ident
            with self.lock:
                self.thread_info[thread_id] = worker_name
            if self.verbose:
                print(f"[{sys.platform}] {worker_name} (Thread ID: {thread_id}) started")
            try:
                worker_function(shutdown_event)
            except Exception as e:
                if self.verbose:
                    print(f"[{sys.platform}] {worker_name} (Thread ID: {thread_id}) error: {e}")
            finally:
                with self.lock:
                    if threading.current_thread() in self.threads:
                        self.threads.remove(threading.current_thread())
                    self.thread_info.pop(thread_id, None)

        thread = threading.Thread(target=wrapped_worker, args=(self.shutdown_event,))
        thread.daemon = True
        with self.lock:
            self.threads.append(thread)
        thread.start()

    def stop(self) -> None:
        if self.verbose:
            print(f"[{sys.platform}] Stopping all threads...")
        self.shutdown_event.set()
        with self.lock:
            threads = self.threads.copy()
        for thread in threads:
            thread.join()
        with self.lock:
            self.threads.clear()
            self.thread_info.clear()
        if self.verbose:
            print(f"[{sys.platform}] All threads stopped.")

    def is_running(self) -> bool:
        with self.lock:
            return len(self.threads) > 0

    def get_thread_info(self) -> Dict[int, str]:
        with self.lock:
            return self.thread_info.copy()

    def is_shutdown(self) -> bool:
        """Check if shutdown_event is set."""
        return self.shutdown_event.is_set()

    def _setup_signal_handler(self) -> None:
        def signal_handler(_sig, _frame):
            if self.verbose:
                print(f"[{sys.platform}] Termination signal received. Stopping all threads...")
            self.stop()
            if self.verbose:
                print(f"[{sys.platform}] All threads stopped. Exiting.")
            sys.exit(0)

        # Only set signal handlers if not already set by GUI
        if not signal.getsignal(signal.SIGINT):
            signal.signal(signal.SIGINT, signal_handler)
        if sys.platform != "win32" and not signal.getsignal(signal.SIGTERM):
            signal.signal(signal.SIGTERM, signal_handler)