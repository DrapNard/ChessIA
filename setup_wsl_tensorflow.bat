@echo off
echo Setting up TensorFlow environment in WSL...

REM Check if WSL is installed
wsl --status > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WSL is not installed. Installing WSL...
    powershell -Command "Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux"
    echo Please restart your computer after WSL installation and run this script again.
    pause
    exit /b
)

REM Check if Ubuntu is installed in WSL
wsl -d Ubuntu -- echo "Ubuntu exists" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ubuntu not found in WSL. Please install Ubuntu from Microsoft Store.
    echo After installation, run this script again.
    start ms-windows-store://pdp/?productid=9PDXGNCFSCZV
    pause
    exit /b
)

echo Copying WSL setup script...
copy /Y wsl_setup.sh %TEMP%\wsl_setup.sh

REM Run the setup script in WSL
echo Running setup in WSL...
wsl bash -c "cp %TEMP%\wsl_setup.sh ~/wsl_setup.sh && chmod +x ~/wsl_setup.sh && ~/wsl_setup.sh"

echo.
echo Setup complete! You can now run your Chess AI in WSL with TensorFlow.
echo To start the Chess AI, open WSL and run: ./run_chess_ai.sh
pause