# build_executable.py
import PyInstaller.__main__
from pathlib import Path
import os

def get_data_files(data_dir):
    """
    Recursively finds all files in a directory and formats them for
    PyInstaller's --add-data flag.
    """
    data_files = []
    base_dir = Path(data_dir)
    for file_path in base_dir.rglob('*'):
        if file_path.is_file():
            # The destination path is relative to the app's root
            destination = file_path.relative_to(base_dir.parent)
            # PyInstaller uses ';' on Windows and ':' on other OSes
            data_files.append(f'{str(file_path)}{os.pathsep}{str(destination.parent)}')
    return data_files

if __name__ == '__main__':
    # --- Configuration ---
    gui_script = 'gridflow/gui.py'
    app_name = 'GridFlow'
    icon_path = 'gridflow_logo.ico'
    
    # Directories containing data files to be bundled
    data_directories_to_bundle = [
        'gridflow/vocab',
        'iowa_border',
        'gridflow_logo.svg' # Also bundle the svg logo
    ]

    # --- Build Command Assembly ---
    command = [
        gui_script,
        '--name', app_name,
        '--onefile',      # Create a single executable file
        '--windowed',     # No console window in the background
        '--noconfirm',    # Overwrite previous builds without asking
    ]

    if Path(icon_path).exists():
        command.extend(['--icon', icon_path])

    # Add all data files from the specified directories
    for data_dir in data_directories_to_bundle:
        if Path(data_dir).is_dir():
            for data_arg in get_data_files(data_dir):
                command.extend(['--add-data', data_arg])
        elif Path(data_dir).is_file(): # Handle single files
             command.extend(['--add-data', f'{data_dir}{os.pathsep}.'])


    print("Running PyInstaller with command:")
    print(f"pyinstaller {' '.join(command)}")

    # --- Run PyInstaller ---
    PyInstaller.__main__.run(command)

    print("\nBuild complete!")
    print(f"Find your application in the 'dist/{app_name}' folder.")

