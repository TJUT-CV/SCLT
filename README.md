# SCLT
- Learning Self-Corrective Network via Adaptive Self-Labeling and Dynamic NMS for High-Performance Long-Term Tracking
- Authors: Zhibin Zhang, Wanli Xue, Kaihua Zhang, Bo Liu, Chengwei Zhang, Jingen Liu, Zhiyong Feng, and Shengyong Chen



### Demo
Demo video: [YouTube](https://youtu.be/b_4Yi5r3kGM), [Bilibili](https://b23.tv/dN1CJF5).


### Hardware
- CPU: Intel Core i9-10900k CPU @ 3.70GHz
- GPU: NVIDIA GeForce RTX 3090 (24G)
- Mem: 32G

### Versions
- Ubuntu 18.04.5 LTS
- CUDA 11.3
- Python 3.8
- Pytorch 1.8.1+cu111
- Torchvision 0.9.1+cu111

## Run Tracker
If you couldn't run our tracker successfully, please first check the `Note` in the last section of this guidance. 

### Step1. Set up the environment
In the code (SCLT),
```
bash install.sh ~/anaconda3 sclt
```
The first parameter `~/anaconda3` indicates the path of anaconda and the second `sclt` indicates the virtual environment used for this project.

### Step2.Run tracker on dataset
```
conda activate sclt
cd <SCLT-path>/pytracking
python run_tracker.py tracker_name parameter_name --dataset_name dataset_name --sequence sequence --debug debug --threads threads
e.g.
python run_tracker.py sclt sclt --dataset_name otb 
```
Here, the dataset_name is the name of the dataset used for evaluation, e.g. ```otb```. See [evaluation.datasets.py](evaluation/datasets.py) for the list of datasets which are supported. The sequence can either be an integer denoting the index of the sequence in the dataset, or the name of the sequence, e.g. ```'Soccer'```.
The ```debug``` parameter can be used to control the level of debug visualizations. ```threads``` parameter can be used to run on multiple threads.

### Note
- If you met the error `ImportError: libcudart.so.11.0: cannot open shared object file: No such file or directory`, please run `sudo cp /usr/local/cuda/lib64/libcudart.so.11.0 /usr/local/lib/libcudart.so.11.0 && sudo ldconfig`.
- If you met the error `RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`, please run `sudo rm ~/.nv`.
- If your tracker stopped with no clear information in the log, and the log stops at `Using /tmp/torch_extensions as PyTorch extensions root...`, please `python -c "from ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import _import_prroi_pooling; _import_prroi_pooling()"` in you local path of `<tracker-path>` in `sclt` virtual environment.


## Results
### LTMU
- [LaSOT](https://drive.google.com/file/d/1j4WuhAbWp7JK9kHeuVbKBE_O3lywuOCR/view?usp=sharing)
- [TLP](https://drive.google.com/file/d/1MffQG5n8mBj-6_-nLIhd6nTCHoJ6bYLE/view?usp=sharing)
- [VOTLT(2021)](https://drive.google.com/file/d/1wfMeYNpwRKy_4DDANRREpFZrAyJxLHri/view?usp=sharing)


