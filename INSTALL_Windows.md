## Installation, tested on Windows 11

**1) First install CUDA and NVIDIA drivers:**
- [Youtube link](https://www.youtube.com/watch?v=r7Am-ZGMef8) <br/> <br/>

**2) Install C++ build tools:**
- download vs_buildtools version between 2017 and 2019: [Google Drive download link](https://drive.google.com/file/d/1ZM6SXUuLSgii66B1k7gVBxM5USnn-ycg/view?usp=sharing) 
- install c++ build tools: make sure you check the option "C++ build tools" in the popup window (Workloads). Refer to this [Youtube link](https://www.youtube.com/watch?v=_keTL9ymGjw) <br/> <br/>

**3) Make a virtual environment (called maskal) using the terminal:**
- conda create --name maskal python=3.9 pip
- conda activate maskal <br/> <br/>

**4) Downgrade setuptools, to prevent this [error](https://github.com/facebookresearch/detectron2/issues/3811):**
- pip uninstall setuptools
- pip install setuptools==59.5.0 <br/> <br/>

**5) Download the code repository including the submodules:**
- git clone https://github.com/pieterblok/maskal.git 
- cd maskal <br/> <br/>

**6) Install the required software libraries (in the maskal virtual environment, using the terminal):**
- pip install -U torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
- pip install pillow==9.0.1
- pip install cython
- pip install jupyter
- pip install opencv-python
- pip install -U fvcore
- pip install scikit-image matplotlib imageio
- pip install black isort flake8 flake8-bugbear flake8-comprehensions
- pip install -e . 
- pip install baal 
- pip install xmltodict 
- pip install seaborn 
- pip install statsmodels 
- pip install cerberus
- pip install darwin-py <br/> <br/>

**7) Check if Pytorch links with CUDA (in the maskal virtual environment, using the terminal):**
- python
- import torch
- torch.version.cuda *(should print 11.1)*
- torch.cuda.is_available() *(should True)*
- torch.cuda.get_device_name(0) *(should print the name of the first GPU)*
- quit() <br/> <br/>

**8) Check if detectron2 is found in python (in the maskal virtual environment, using the terminal):**
- python
- import detectron2 *(should not print an error)*
- from detectron2 import model_zoo *(should not print an error)*
- quit() <br/><br/>
