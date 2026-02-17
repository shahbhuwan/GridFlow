@echo off
setlocal

:: Change the current directory to the directory where this script is located.
:: the script is "Run as administrator".
cd /d "%~dp0"

echo ======================================================
echo  GridFlow CLI Environment Setup
echo ======================================================
echo.

REM --- Detect a compatible Python interpreter ---
REM Tier 1: Try "python" on PATH and check that it is 3.10 <= version < 3.13
set "PYTHON_CMD="

python -c "import sys; exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto :python_found
)

REM Tier 2: Try the Windows Python Launcher (py) with 3.11
py -3.11 -c "print('ok')" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3.11"
    goto :python_found
)

REM Tier 3: Neither worked — tell the user what to do
echo.
echo [CRITICAL ERROR] Compatible Python not found.
echo GridFlow CLI requires Python 3.10, 3.11, or 3.12.
echo.
echo Recommended: Install Python 3.11 from https://www.python.org/downloads/
echo              Make sure to check "Add Python to PATH" during installation.
echo.
echo If you already have Python 3.11 installed but it is not on your PATH,
echo install the Python Launcher for Windows ("py") so this script can find it:
echo   https://docs.python.org/3/using/windows.html#launcher
echo.
echo Then run this script again.
pause
exit /b 1

:python_found
echo Detected Python: %PYTHON_CMD%
for /f "delims=" %%v in ('%PYTHON_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set PY_VER=%%v
echo Python version:  %PY_VER%
echo.

echo Enter the full path where you want the environment installed.
echo Example: C:\Environments\GridFlow_Env
set /p VENV_NAME="Install Path (Leave empty for default '.\gridflow_env'): "
if "%VENV_NAME%"=="" set VENV_NAME=gridflow_env

echo Checking for existing virtual environment at '%VENV_NAME%'...

REM --- Create the virtual environment if it doesn't exist ---
if not exist "%VENV_NAME%\" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv "%VENV_NAME%"
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
pip install --no-cache-dir -r requirements_cli.txt
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