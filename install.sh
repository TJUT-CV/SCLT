#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.8 ******************"
conda create -y --name $conda_env_name python=3.8

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
source activate $conda_env_name
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda11.2 ******************"
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing scikit-image ******************"
pip install scikit-image

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib


echo ""
echo ""
echo "****************** Installing shapely ******************"
pip install shapely

echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing LVIS toolkit ******************"
pip install lvis



echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama


echo ""
echo ""
echo "****************** Installing mmcv ******************"
pip install mmcv-full==1.3.3

echo ""
echo ""
echo "****************** Installing pillow ******************"
pip install pillow==8.2.0


echo ""
echo ""
echo "****************** Installing protobuf ******************"
pip install protobuf==3.17.0


echo ""
echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"
echo "************************* Need sudo privilege ******************"
sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import _import_prroi_pooling; _import_prroi_pooling()"

echo "Download the default network for stre_model and super_dimp etc."
gdown https://drive.google.com/file/d/1nF6heqbW4NEujCFNhDaaKGZpu8Uhywz2/view?usp=sharing -O pytracking/networks/qg_rcnn_r50_fpn_coco_got10k_lasot.pth
gdown https://drive.google.com/file/d/1s5Rk31RTSIzjs1mjkE-ps8Mr4eP0wEYi/view?usp=sharing -O pytracking/networks/metric_model.pth
gdown https://drive.google.com/file/d/1-_bGTbjMnsOak4dKqAFVgThuU56cesgg/view?usp=sharing -O pytracking/networks/super_dimp.pth.tar



echo "Building mmdetection"
cd pytracking/global_tracking/_submodules/mmdetection
if [ -d "build" ]; then
    rm -r build
fi
find . -name "*.so"  | xargs rm -f
python setup.py build develop
cd ../../../../


echo ""
echo ""
echo "****************** Installing jpeg4py ******************"
while true; do
    read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
    case $install_flag in
        [Yy]* ) sudo apt-get install libturbojpeg; break;;
        [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
        * ) echo "Please answer y or n  ";;
    esac
done



