# Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking. The instrustions have been tested on an Ubuntu 18.04 system. We recommend using the [install script](install.sh) if you have not already tried that.  

### Requirements  
* Conda installation with Python 3.8. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name sclt python=3.8
source activate sclt
conda activate sclt
```

#### Install PyTorch  
Install PyTorch with cuda11.2.  
```bash

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**Note:**  
- It is possible to use any PyTorch supported version of CUDA (not necessarily v10).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install matplotlib, pandas, tqdm, opencv, scikit-image, visdom, tikzplotlib, gdown, and tensorboad  
```bash
conda install matplotlib pandas tqdm
pip install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
```


#### Install the coco and lvis toolkits  
```bash
conda install cython
pip install pycocotools
pip install lvis
```


#### Install the colorama
```bash
pip install colorama
```


#### Install the mmcv
```bash
pip install mmcv-full==1.3.3
```

#### Install the pillow
```bash
pip install pillow==8.2.0
```

#### Install the protobuf
```bash
pip install protobuf==3.17.0
```


#### Setup the environment
Create the default environment setting files.
```bash
# Environment settings for pytracking. Save at pytracking/evaluation.local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Save at ltr/admin.local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```
You can modify these files to set the paths to datasets, results paths etc.  


#### Install Precise ROI pooling  
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
python -c "from ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import _import_prroi_pooling; _import_prroi_pooling()"
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  
