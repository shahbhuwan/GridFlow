import sys
import os
import subprocess
import shutil
from pathlib import Path

# Define versions to look for (standard Windows launcher names or executables on PATH)
TARGET_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]

def find_python_executable(version):
    """Try to find a python executable for the given version."""
    # Method 1: py launcher
    try:
        # Check if py launcher knows about this version
        result = subprocess.run(["py", f"-{version}", "-c", "import sys; print(sys.executable)"], 
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 2: Check PATH for pythonX.Y
    exe_name = f"python{version}"
    path = shutil.which(exe_name)
    if path:
        return path
        
    return None

def test_version(version):
    print(f"\n{'='*50}")
    print(f"Testing Python {version}...")
    
    exe = find_python_executable(version)
    if not exe:
        print(f"❌ Python {version} not found. Skipping.")
        return False
        
    print(f"Found executable: {exe}")
    
    try:
        # Create venv
        venv_dir = Path(f"test_env_{version.replace('.', '')}")
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        
        print(f"Creating venv at {venv_dir}...")
        subprocess.run([exe, "-m", "venv", str(venv_dir)], check=True)
        
        # Install
        pip_exe = venv_dir / "Scripts" / "pip.exe"
        print("Installing GridFlow...")
        subprocess.run([str(pip_exe), "install", "."], check=True)
        
        # Verify CLI
        gridflow_exe = venv_dir / "Scripts" / "gridflow.exe"
        print("Verifying CLI...")
        result = subprocess.run([str(gridflow_exe), "--help"], capture_output=True, text=True, check=True)
        
        if "GridFlow" in result.stdout or "usage:" in result.stdout:
            print(f"✅ Python {version} PASSED")
            return True
        else:
            print(f"❌ Python {version} FAILED (Command ran but output unexpected)")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Python {version} FAILED (Command failed)")
        if e.stdout: print(e.stdout)
        if e.stderr: print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Python {version} FAILED (Unexpected error: {e})")
        return False
    finally:
        # Cleanup
        if venv_dir.exists():
            shutil.rmtree(venv_dir)

def main():
    results = {}
    for v in TARGET_VERSIONS:
        results[v] = test_version(v)
        
    print(f"\n{'='*50}")
    print("SUMMARY")
    for v, status in results.items():
        status_icon = "✅ PASSED" if status else "❌ FAILED/SKIPPED"
        print(f"Python {v}: {status_icon}")

if __name__ == "__main__":
    main()
