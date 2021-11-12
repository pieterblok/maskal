## Installation (choose to either install CUDA 10.1 (step 1a) or 11.1 (step 1b))

## 1a.) Install CUDA 10.1 (if not yet installed) 
Tested on: Ubuntu 16.04 and 18.04, CUDA: 10.1 (10.1.105)<br/> 

**1a.1) Install NVIDIA drivers (using the terminal):** 
- sudo add-apt-repository ppa:graphics-drivers/ppa
- sudo apt update
- sudo apt install nvidia-driver-440
- nvidia-smi
- sudo reboot <br/> <br/>

**1a.2) Download CUDA 10.1:** 
- download cuda 10.1: https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
- download the deb file, for example: cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb <br/> <br/>

**1a.3) Install CUDA 10.1 (using the terminal, cd to the directory where the file has been downloaded):** 
- sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
- sudo apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
- sudo apt-get update
- sudo apt-get install cuda-toolkit-10-1 <br/> <br/>

**1a.4) Set the environmental path:**
- sudo gedit ~/.bashrc
- add the following 2 lines at the end of the bashrc file:
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
- save the bashrc file and close it
- source ~/.bashrc <br/> <br/>

**1a.5) Check if CUDA has been installed properly:**
- nvcc --version *(should the CUDA details)*<br/> <br/>


## 1b.) Install CUDA 11.1.1 (if not yet installed) 
Tested on: Ubuntu 20.04, CUDA: 11.1.1 (11.1.105-1)<br/> 

**1b.1) Install NVIDIA drivers (using the terminal):** 
- sudo add-apt-repository ppa:graphics-drivers/ppa
- sudo apt update
- sudo apt install nvidia-driver-460
- nvidia-smi
- sudo reboot <br/> <br/>

**1b.2) Install CUDA 11.1.1:** 
- wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
- sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
- wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
- sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
- sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
- sudo apt-get update
- sudo apt-get install cuda-toolkit-11-1 <br/> <br/>

**1b.3) Set the environmental path:**
- sudo gedit ~/.bashrc
- add the following 2 lines at the end of the bashrc file:
export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
- save the bashrc file and close it
- source ~/.bashrc <br/> <br/>

**1b.4) Check if CUDA has been installed properly:**
- nvcc --version *(should the CUDA details)*<br/> <br/>


## 2.) Install maskAL in a virtual environment (using Anaconda)
Tested with: Pytorch 1.8.0 & torchvision 0.9.0 (Ubuntu 16.04/18.04 and CUDA 10.1) and Pytorch 1.7.1 & torchvision 0.8.2 (Ubuntu 20.04 and CUDA 11.1)<br/>

**2.1) Download and install Anaconda:**
- download anaconda: https://www.anaconda.com/distribution/#download-section (python 3.x version)
- install anaconda (using the terminal, cd to the directory where the file has been downloaded): bash Anaconda3-2019.10-Linux-x86_64.sh <br/> <br/>

**2.2) Make a virtual environment (called maskAL) using the terminal:**
- conda create --name maskAL python=3.8 pip
- conda activate maskAL <br/> <br/>

**2.3) Download the code repository:**
- git clone https://github.com/pieterblok/maskAL.git
- cd maskAL <br/> <br/>

**2.4) Install the required software libraries (in the maskAL virtual environment, using the terminal):**
- *for cuda 10.1:* pip install -U torch==1.8.0 torchvision==0.9.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
- *for cuda 11.1:* pip install -U torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/cu111/torch_stable.html
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
- pip install baal 
- pip install xmltodict 
- pip install seaborn 
- pip install statsmodels 
- pip install cerberus 
- pip install darwin-py <br/> <br/>

**2.5) Reboot/restart the computer (sudo reboot)** <br/> <br/>

**2.6) Check if Pytorch links with CUDA (in the maskAL virtual environment, using the terminal):**
- python
- import torch
- torch.version.cuda *(should print 10.1 or 11.1)*
- torch.cuda.is_available() *(should True)*
- torch.cuda.get_device_name(0) *(should print the name of the first GPU)*
- quit() <br/> <br/>

**2.7) Check if detectron2 is found in python (in the maskAL virtual environment, using the terminal):**
- python
- import detectron2 *(should not print an error)*
- from detectron2 import model_zoo *(should not print an error)*
- quit() <br/>
