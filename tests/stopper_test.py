import sys
import time
import math
import threading
from gridflow.thread_stopper import ThreadManager

# Worker functions
def math_worker(shutdown_event):
    """Worker that computes square roots."""
    count = 0
    while not shutdown_event.is_set():
        result = math.sqrt(count)
        print(f"[{sys.platform}] Math worker (Thread ID: {threading.current_thread().ident}): sqrt({count}) = {result:.4f}")
        time.sleep(1)
        count += 1

def string_worker(shutdown_event):
    """Worker that reverses strings."""
    strings = ["hello", "world", "python", "threading"]
    index = 0
    while not shutdown_event.is_set():
        current_string = strings[index % len(strings)]
        reversed_string = current_string[::-1]
        print(f"[{sys.platform}] String worker (Thread ID: {threading.current_thread().ident}): Reversed '{current_string}' = '{reversed_string}'")
        time.sleep(1.5)
        index += 1

def counter_worker(shutdown_event):
    """Worker that counts up."""
    count = 0
    while not shutdown_event.is_set():
        print(f"[{sys.platform}] Counter worker (Thread ID: {threading.current_thread().ident}): Count = {count}")
        time.sleep(0.8)
        count += 1

def sleep_worker(shutdown_event):
    """Worker that simulates long tasks with sleep."""
    while not shutdown_event.is_set():
        print(f"[{sys.platform}] Sleep worker (Thread ID: {threading.current_thread().ident}): Sleeping...")
        time.sleep(2)
        print(f"[{sys.platform}] Sleep worker (Thread ID: {threading.current_thread().ident}): Awake!")

# Main execution
if __name__ == "__main__":
    print(f"[{sys.platform}] Starting thread stopper demo at {time.strftime('%I:%M %p %Z, %B %d, %Y')}")
    manager = ThreadManager(verbose=True)
    
    # Add 4 workers
    print(f"[{sys.platform}] Adding 4 workers...")
    manager.add_worker(math_worker, "Math worker")
    manager.add_worker(string_worker, "String worker")
    manager.add_worker(counter_worker, "Counter worker")
    manager.add_worker(sleep_worker, "Sleep worker")
    
    # Print initial thread info
    print(f"[{sys.platform}] Active threads: {manager.get_thread_info()}")
    
    print(f"[{sys.platform}] Workers started. Press Enter to stop all workers or Ctrl+C to exit...")
    try:
        input()
        print(f"[{sys.platform}] Stopping all workers...")
        manager.stop()
        print(f"[{sys.platform}] Demo completed. Active threads: {manager.get_thread_info()}")
    except KeyboardInterrupt:
        pass  # Signal handler in ThreadManager handles Ctrl+C