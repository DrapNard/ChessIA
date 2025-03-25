#!/bin/bash 
echo "Setting up Python environment for TensorFlow in WSL..." 
cd /mnt/c/Users/DrapNard/ChessIA 
sudo apt-get update -y 
sudo apt-get install -y python3-pip python3-venv python3-dev 
python3 -m venv venv_wsl 
source venv_wsl/bin/activate 
pip install --upgrade pip 
pip install tensorflow numpy matplotlib pandas scikit-learn 
if [ -x "$(command -v nvidia-smi)" ]; then 
    echo "NVIDIA GPU detected, installing CUDA support..." 
    sudo apt-get install -y nvidia-cuda-toolkit 
    pip install tensorflow-gpu 
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" 
    echo "CUDA installation complete!" 
else 
    echo "No NVIDIA GPU detected, using CPU version of TensorFlow" 
fi 
echo "Creating TensorFlow test script..." 
echo "import tensorflow as tf" 
echo "print('TensorFlow version:', tf.__version__)" 
echo "print('GPU Available:', tf.config.list_physical_devices('GPU'))" 
echo "print('TensorFlow installation successful!')" 
python tf_test.py 
echo "Creating Chess AI launcher script..." 
echo "#!/bin/bash" 
echo "cd /mnt/c/Users/DrapNard/ChessIA" 
echo "source venv_wsl/bin/activate" 
echo "python main.py" 
chmod +x run_chess_ai.sh 
echo "Setup complete! You can now run your Chess AI in WSL with TensorFlow." 
echo "To start the Chess AI, run: ./run_chess_ai.sh" 
