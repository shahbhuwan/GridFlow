import ctypes
# Set DPI awareness to Per-Monitor DPI Aware (Windows 8.1+)
ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2