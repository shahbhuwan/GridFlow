@echo off
setlocal

:: Change the current directory to the directory where this script is located.
:: the script is "Run as administrator".
cd /d "%~dp0"

echo ======================================================
echo  GridFlow CLI Environment Setup
echo ======================================================
echo.

REM --- Check if Python is installed and meets version requirements (3.10+) ---
python -c "import sys; exit(1) if not ((3,10) <= sys.version_info < (3,13)) else exit(0)" >nul 2>&1

if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL ERROR] Python is missing or too old.
    echo GridFlow requires Python 3.10 or newer.
    echo.
    echo Opening Python download page...
    start https://www.python.org/downloads/
    echo.
    echo Please install Python, checking the box "Add Python to PATH" during installation.
    echo Then run this script again.
    pause
    exit /b 1
)

echo Enter the full path where you want the environment installed.
echo Example: C:\Environments\GridFlow_Env
set /p VENV_NAME="Install Path (Leave empty for default '.\gridflow_env'): "
if "%VENV_NAME%"=="" set VENV_NAME=gridflow_env

echo Checking for existing virtual environment at '%VENV_NAME%'...

REM --- Create the virtual environment if it doesn't exist ---
if not exist "%VENV_NAME%\" (
    echo Creating virtual environment...
    python -m venv "%VENV_NAME%"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo --- Activating the virtual environment ---
call "%VENV_NAME%\Scripts\activate.bat"

echo.
echo --- Installing Dependencies ---
REM Using --no-cache-dir to prevent Permission Denied errors on cached wheels
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r requirements_dev.txt

echo.
echo --- Installing GridFlow ---
pip install --no-cache-dir .
if %errorlevel% neq 0 (
    echo ERROR: Failed to install GridFlow. Please check for errors above.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo  Setup Complete!
echo ======================================================
echo.
echo To use the GridFlow CLI, you must first activate this environment.
echo From this directory, run the following command:
echo.
echo   call %VENV_NAME%\Scripts\activate.bat
echo.
echo Once activated, you can use the 'gridflow' command.
echo.
pause