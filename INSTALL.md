## Installation

## 1.) Install CUDA 10.2 (if not yet installed) 
Tested on: Ubuntu 18.04, CUDA: 10.2 (10.2.89)<br/> 

**1.1) Install NVIDIA drivers (using the terminal):** 
- sudo add-apt-repository ppa:graphics-drivers/ppa
- sudo apt update
- sudo apt install nvidia-driver-440
- nvidia-smi
- sudo reboot <br/> <br/>

**1.2) Download and install CUDA 10.2:** 
- wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
- sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
- wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
- sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
- sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
- sudo apt-get update
- sudo apt-get -y install cuda <br/> <br/>

**1.3) Set the environmental path:**
- sudo gedit ~/.bashrc
- add the following 2 lines at the end of the bashrc file:
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
- save the bashrc file and close it
- source ~/.bashrc <br/> <br/>

**1.4) Check if CUDA has been installed properly:**
- nvcc --version *(should the CUDA details)*<br/> <br/>


## 2.) Install maskAL in a virtual environment (using Anaconda)
Tested with: Pytorch 1.8.0 and torchvision 0.9.0<br/>

**2.1) Download and install Anaconda:**
- download anaconda: https://www.anaconda.com/distribution/#download-section (python 3.x version)
- install anaconda (using the terminal, cd to the directory where the file has been downloaded): bash Anaconda3-2019.10-Linux-x86_64.sh <br/> <br/>

**2.2) Make a virtual environment (called maskAL) using the terminal:**
- conda create --name maskAL python=3.8 pip
- conda activate maskAL <br/> <br/>

**2.3) Download the code repository:**
- git clone https://git.wur.nl/blok012/maskAL.git
- cd maskAL <br/> <br/>

**2.4) Install the required software libraries (in the maskAL virtual environment, using the terminal):**
- pip install -U torch==1.8.0 torchvision==0.9.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
- pip install cython pyyaml==5.1
- pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
- pip install jupyter
- pip install opencv-python
- pip install -U fvcore
- pip install nbformat==4.4
- pip install scikit-image matplotlib imageio
- pip install black isort flake8 flake8-bugbear flake8-comprehensions
- pip install -e .
- pip install scikit-learn==0.22.2
- pip install pandas
- pip install h5py
- pip install structlog
- pip install pytorch-lightning==0.8.5
- pip install transformers
- pip install datasets
- pip install onnx
- pip install baal <br/> <br/>

**2.5) Reboot/restart the computer (sudo reboot)** <br/> <br/>

**2.6) Check if Pytorch links with CUDA (in the maskAL virtual environment, using the terminal):**
- python
- import torch
- torch.version.cuda *(should print 10.2)*
- torch.cuda.is_available() *(should True)*
- torch.cuda.get_device_name(0) *(should print the name of the first GPU)*
- quit() <br/> <br/>

**2.7) Check if detectron2 is found in python (in the maskAL virtual environment, using the terminal):**
- python
- import detectron2 *(should not print an error)*
- from detectron2 import model_zoo *(should not print an error)*
- quit() <br/>
