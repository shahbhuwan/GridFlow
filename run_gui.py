from PyQt5.QtWidgets import QApplication
import sys
import ctypes

if __name__ == "__main__":
    if sys.platform == "win32":
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # Per-Monitor DPI Aware
    app = QApplication(sys.argv)
    from gui import GridFlowGUI
    window = GridFlowGUI()
    window.show()
    sys.exit(app.exec_())