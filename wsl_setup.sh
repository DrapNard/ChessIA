#!/bin/bash
echo "Setting up Python venv environment for TensorFlow in WSL..."
cd /mnt/c/Users/DrapNard/ChessIA

# Update and install required packages
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv python3-dev python3-tk \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev

# Clean up any previous installation
if [ -d "venv_wsl" ]; then
    echo "Removing previous venv installation..."
    rm -rf venv_wsl
fi

# Create a new Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv_wsl

# Activate the virtual environment
echo "Activating virtual environment..."
source venv_wsl/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install numpy matplotlib pandas scikit-learn
pip install "tensorflow[and-cuda]"
pip install flask flask-socketio

# Check for NVIDIA GPU and install CUDA if available
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "NVIDIA GPU detected, installing CUDA support..."
    pip install nvidia-cudnn-cu11
    
    echo "CUDA installation complete!"
else
    echo "No NVIDIA GPU detected, using CPU version of TensorFlow"
fi

# Create a test script to verify TensorFlow and NumPy installation
echo "Creating TensorFlow test script..."
cat > tf_test.py << 'EOL'
import numpy as np
print("NumPy version:", np.__version__)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Test TensorFlow operations
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Test matrix multiplication result:")
    print(c)
    
    print("TensorFlow installation successful!")
except Exception as e:
    print("Error testing TensorFlow:", e)
EOL

# Run the test script
echo "Running TensorFlow test..."
python tf_test.py

# Create a launcher script for the Chess AI
echo "Creating Chess AI launcher script..."
cat > run_chess_ai.sh << EOL
#!/bin/bash
cd /mnt/c/Users/DrapNard/ChessIA
source venv_wsl/bin/activate
export PYTHONPATH=/mnt/c/Users/DrapNard/ChessIA
python main.py
EOL

# Create a launcher script for the web server
echo "Creating Web Server launcher script..."
cat > wsl_launch.sh << EOL
#!/bin/bash
cd /mnt/c/Users/DrapNard/ChessIA
source venv_wsl/bin/activate
export PYTHONPATH=/mnt/c/Users/DrapNard/ChessIA
python web_server.py
EOL

chmod +x run_chess_ai.sh
chmod +x wsl_launch.sh

echo "Setup complete! You can now run your Chess AI in WSL with TensorFlow using venv."
echo "To start the Chess AI, run: ./run_chess_ai.sh"
echo "To start the Web Server, run: ./wsl_launch.sh"
