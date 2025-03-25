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

echo Creating WSL setup script...

REM Create a temporary script to run inside WSL
echo #!/bin/bash > wsl_setup.sh
echo echo "Setting up Python environment for TensorFlow in WSL..." >> wsl_setup.sh
echo cd /mnt/c/Users/DrapNard/ChessIA >> wsl_setup.sh
echo sudo apt-get update -y >> wsl_setup.sh
echo sudo apt-get install -y python3-pip python3-venv python3-dev >> wsl_setup.sh
echo python3 -m venv venv_wsl >> wsl_setup.sh
echo source venv_wsl/bin/activate >> wsl_setup.sh
echo pip install --upgrade pip >> wsl_setup.sh
echo pip install tensorflow numpy matplotlib pandas scikit-learn >> wsl_setup.sh

REM Check for NVIDIA GPU and install CUDA if available
echo if [ -x "$(command -v nvidia-smi)" ]; then >> wsl_setup.sh
echo     echo "NVIDIA GPU detected, installing CUDA support..." >> wsl_setup.sh
echo     sudo apt-get install -y nvidia-cuda-toolkit >> wsl_setup.sh
echo     pip install tensorflow-gpu >> wsl_setup.sh
echo     echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc >> wsl_setup.sh
echo     echo "CUDA installation complete!" >> wsl_setup.sh
echo else >> wsl_setup.sh
echo     echo "No NVIDIA GPU detected, using CPU version of TensorFlow" >> wsl_setup.sh
echo fi >> wsl_setup.sh

REM Create a test script to verify TensorFlow installation
echo echo "Creating TensorFlow test script..." >> wsl_setup.sh
echo echo "import tensorflow as tf" > tf_test.py >> wsl_setup.sh
echo echo "print('TensorFlow version:', tf.__version__)" >> tf_test.py >> wsl_setup.sh
echo echo "print('GPU Available:', tf.config.list_physical_devices('GPU'))" >> tf_test.py >> wsl_setup.sh
echo echo "print('TensorFlow installation successful!')" >> tf_test.py >> wsl_setup.sh
echo python tf_test.py >> wsl_setup.sh

REM Create a launcher script for the Chess AI
echo echo "Creating Chess AI launcher script..." >> wsl_setup.sh
echo echo "#!/bin/bash" > run_chess_ai.sh >> wsl_setup.sh
echo echo "cd /mnt/c/Users/DrapNard/ChessIA" >> run_chess_ai.sh >> wsl_setup.sh
echo echo "source venv_wsl/bin/activate" >> run_chess_ai.sh >> wsl_setup.sh
echo echo "python main.py" >> run_chess_ai.sh >> wsl_setup.sh
echo chmod +x run_chess_ai.sh >> wsl_setup.sh

echo echo "Setup complete! You can now run your Chess AI in WSL with TensorFlow." >> wsl_setup.sh
echo echo "To start the Chess AI, run: ./run_chess_ai.sh" >> wsl_setup.sh

REM Make the script executable and run it in WSL
wsl chmod +x wsl_setup.sh
echo Running setup in WSL...
wsl ./wsl_setup.sh

REM Clean up

echo.
echo Setup complete! You can now run your Chess AI in WSL with TensorFlow.
echo To start the Chess AI, open WSL and run: ./run_chess_ai.sh
pause