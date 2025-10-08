# Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation [ICML 2025] 

This repository is the official implementaiton of our paper __Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation__

## Setup Environment
For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

In that environment, the requirements can be installed with:
### pytorch:1.7.1-cuda11.0-cudnn8-devel
```shell
docker run --shm-size=8g --gpus all -e NVIDIA_VISIBLE_DEVICES=$GPU -it pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
```

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
# or
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

### pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
```shell
docker run --shm-size=8g --gpus all -e NVIDIA_VISIBLE_DEVICES=$GPU -v /NAS:/NAS -v /SSD:/SSD -it pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
```

```shell
pip install cityscapesscripts==2.2.0 cycler==0.10.0 gdown==4.2.0 humanfriendly==9.2 kiwisolver==1.2.0 kornia==0.5.8 matplotlib==3.4.2 numpy opencv-python pandas Pillow prettytable==2.1.0 pyparsing easydict pytz PyYAML scipy seaborn timm tqdm typing-extensions wcwidth yapf linklink
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```


Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.
Further, download the checkpoint of [HRDA on GTA→Cityscapes](https://drive.google.com/file/d/1O6n1HearrXHZTHxNRWp8HCMyqbulKcSW/view?usp=sharing) and extract it to the folder
 `work_dirs/`.


## Trouble Shooting in Setup Environment
### ImportError: libGL.so.1 is occured
```shell
apt-get install -y libgl1-mesa-glx
```

### ImportError: libgthread-2.0.so.0 is occured
```shell
apt-get install -y libglib2.0-0
```


## Setting
### Dataset - GTA
- https://download.visinf.tu-darmstadt.de/data/from_games/
- Download zip files (01_images.zip ~ 10_images.zip, 01_labels.zip ~ 10_labels.zip)
- Unzip each zip files

### Dataset - CityScapes (CS)
- https://www.cityscapes-dataset.com/downloads/
  - Need to sign up 
- Download "gtFine_trainvaltest.zip" and "leftImg8bit_trainvaltest.zip"

### Dataset - IDD
- https://idd.insaan.iiit.ac.in/dataset/download/
  - Need to sign up 
- Download segmentation file
- Unzip files
- Extract
    - git clone https://github.com/mseg-dataset/mseg-api.git
    - pip install -e [repo root path]
    - python ./mseg/label_preparation/dump_idd_semantic_labels.py --num-workers 8 --datadir [idd data path]

### Data Processing
- GTA
  - python tools/convert_datasets/gta.py data/gta --nproc 8
- CS
  - python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
- IDD
  - python tools/convert_datasets/idd.py data/idd --nproc 8

### For Quantization
Create log, work_dirs, ckpt direcotries
- log: log text
- work_dirs: fp models
- ckpt: save models dir
    - ckpt/CS: for CS dataset
    - ckpt/IDD: for IDD dataset

```shell
mkdir log
mkdir work_dirs
mkdir ckpt
cd ckpt; mkdir CS
cd ckpt; mkidr IDD
```
### For Evaluation
Create result_txts
- result_txts: save result texts of evalution

```shell
mkdir result_txts
```

### Checkpoints
FP and quantized checkpoints are [here](https://drive.google.com/drive/folders/18aI7csI3AukLt9NGEow9zMhjrs8c-3Y6?usp=sharing)
## Quantization
### Example
- BRECQ
```shell
# Example commands (GTA -> CS, BRECQ, W8A8, seed: 2025)
./test_brecq.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 8 8 cs 2025 fixed

# Example commands (GTA -> IDD, BRECQ, W8A8, seed: 2025)
./test_brecq.sh work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a 8 8 idd 2025 fixed
```

- QDROP
```shell
# Example commands (GTA -> CS, QDrop, W8A8, seed: 2025)
./test_qdrop.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 8 8 cs 2025 fixed

# Example commands (GTA -> IDD, QDrop, W8A8, seed: 2025)
./test_qdrop.sh work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a 8 8 idd 2025 fixed
```
- HDRQ

```shell
# Example commands (GTA -> CS, HDRQ, W8A8, seed: 2025)
./test_hdrq.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 8 8 cs 2025 fixed

# Example commands (GTA -> IDD, HDRQ, W8A8, seed: 2025)
./test_hdrq.sh work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a 8 8 idd 2025 fixed
```

## Evaluation (Each Quantization)
```shell
./eval.sh [config path] [model_path] [n_gpus]
```

ex1) HDRQ (IDD)
```shell
# Example commands (HDRQ, W4A4, 4 GPUS, quant_seed: 1005)
./eval.sh work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a ckpt/IDD/HDRQ_PTQTestR101_W4A4_fixed_IDD_seed1005.pt 4
```

ex2) HDRQ (CS)
```shell
# Example commands (HDRQ, W4A4, 4 GPUS, quant_seed: 1005)
./eval.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 ckpt/CS/HDRQ_PTQTestR101_W4A4_fixed_IDD_seed1005.pt 4
```

ex3) QDROP (IDD)
```shell
# Example commands (QDROP, W4A4, 4 GPUS, quant_seed: 1005)
./eval.sh work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a ckpt/IDD/QDROP_PTQTestR101_W4A4_fixed_IDD_seed1005.pt 4
```

ex4) QDROP (CS)
```shell
# Example commands (QDROP, W4A4, 4 GPUS, quant_seed: 1005)
./eval.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 ckpt/CS/QDROP_PTQTestR101_W4A4_fixed_IDD_seed1005.pt 4
```


## Evaluation (Model Merging)

```shell
# Quantization Type: BRECQ, QDROP, HDRQ, FLEX, SMQ

./eval_merge_seed.sh [config path of dataset1] [config path of dataset2] [model_path of dataset1] [model_path of dataset2] [n_gpus] [quantization type] [port_num (for ddp)] [eval_seed]

# using default port num (29703)
./eval_merge_seed.sh [config path of dataset1] [config path of dataset2] [model_path of dataset1] [model_path of dataset2] [n_gpus] [quantization type] [eval_seed]


# eval_merge.sh also working! (without setting seed)
# you can skip a "port_num" option (default port_num is 29703)
./eval_merge.sh [config path of dataset1] [config path of dataset2] [model_path of dataset1] [model_path of dataset2] [n_gpus] [quantization type] [port_num (for ddp)]
```

ex1) QDROP (./eval_merge_seed.sh)
```shell
# Example commands (QDROP, W4A4, 4 GPUS, quant_seed: 200, eval_seed: 5000)
./eval_merge_seed.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a ckpt/CS/QDROP_PTQTestR101_W4A4_fixed_CS_seed200.pt ckpt/IDD/QDROP_PTQTestR101_W4A4_fixed_IDD_seed200.pt 4 QDROP 5000
```

ex2) HDRQ (./eval_merge.sh)
```shell
# Example commands (HDRQ, W4A4, 4 GPUS, quant_seed: 200, eval_seed: 5000)
./eval_merge.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a ckpt/CS/HDRQ_PTQTestR101_W4A4_fixed_CS_seed200.pt ckpt/IDD/HDRQ_PTQTestR101_W4A4_fixed_IDD_seed200.pt 4 HDRQ 29700
```

## Reference
- [HRDA](https://github.com/lhoyer/HRDA)
- [BRECQ](https://github.com/yhhhli/BRECQ)
- [QDrop](https://github.com/wimh966/QDrop)
