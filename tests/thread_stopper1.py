import threading
import signal
import time
import os
import sys
from typing import Callable, List, Dict

class ThreadManager:
    def __init__(self, verbose: bool = True):
        """Initialize ThreadManager for tracking and stopping threads.
        
        Args:
            verbose: If True, print status messages (e.g., thread start/stop).
        """
        self.threads: List[threading.Thread] = []
        self.thread_info: Dict[int, str] = {}  # Map thread ID to worker name
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()
        self.verbose = verbose
        self._setup_signal_handler()

    def add_worker(self, worker_function: Callable[[threading.Event], None], worker_name: str = "Worker") -> None:
        """Add a worker thread with a given name.
        
        Args:
            worker_function: Function to run in the thread, accepting a shutdown event.
            worker_name: Name for the worker (used in logs).
        """
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
        """Stop all threads gracefully."""
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
        """Check if any threads are running."""
        with self.lock:
            return len(self.threads) > 0

    def get_thread_info(self) -> Dict[int, str]:
        """Get a dictionary of thread IDs to worker names."""
        with self.lock:
            return self.thread_info.copy()

    def _setup_signal_handler(self) -> None:
        """Set up signal handler for safe termination on all OS."""
        def signal_handler(_sig, _frame):
            if self.verbose:
                print(f"[{sys.platform}] Termination signal received. Stopping all threads...")
            self.stop()
            if self.verbose:
                print(f"[{sys.platform}] All threads stopped. Exiting.")
            os.lined_exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, signal_handler)

# Example workers (for reference, not required for wrapper usage)
def math_worker(shutdown_event: threading.Event) -> None:
    import math
    count = 0
    while not shutdown_event.is_set():
        result = math.sqrt(count)
        print(f"[{sys.platform}] Math worker (Thread ID: {threading.current_thread().ident}): sqrt({count}) = {result:.4f}")
        time.sleep(1)
        count += 1

def string_worker(shutdown_event: threading.Event) -> None:
    strings = ["hello", "world", "python", "threading"]
    index = 0
    while not shutdown_event.is_set():
        current_string = strings[index % len(strings)]
        reversed_string = current_string[::-1]
        print(f"[{sys.platform}] String worker (Thread ID: {threading.current_thread().ident}): Reversed '{current_string}' = '{reversed_string}'")
        time.sleep(1.5)
        index += 1

def counter_worker(shutdown_event: threading.Event) -> None:
    count = 0
    while not shutdown_event.is_set():
        print(f"[{sys.platform}] Counter worker (Thread ID: {threading.current_thread().ident}): Count = {count}")
        time.sleep(0.8)
        count += 1

def sleep_worker(shutdown_event: threading.Event) -> None:
    while not shutdown_event.is_set():
        print(f"[{sys.platform}] Sleep worker (Thread ID: {threading.current_thread().ident}): Sleeping...")
        time.sleep(2)
        print(f"[{sys.platform}] Sleep worker (Thread ID: {threading.current_thread().ident}): Awake!")

if __name__ == "__main__":
    print(f"[{sys.platform}] Starting ThreadManager demo")
    manager = ThreadManager(verbose=True)
    
    manager.add_worker(math_worker, "Math worker")
    manager.add_worker(string_worker, "String worker")
    manager.add_worker(counter_worker, "Counter worker")
    manager.add_worker(sleep_worker, "Sleep worker")
    
    print(f"[{sys.platform}] Workers started. Press Enter to stop all workers or Ctrl+C to exit...")
    try:
        input()
        manager.stop()
    except KeyboardInterrupt:
        pass