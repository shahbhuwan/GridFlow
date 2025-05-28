import sys
from gridflow import __main__

def main():
    # If no arguments or only the script name, launch GUI
    if len(sys.argv) <= 1:
        from gui.main import main as gui_main
        gui_main()
    else:
        # Run CLI from __main__.py
        __main__.main()

if __name__ == '__main__':
    main()
