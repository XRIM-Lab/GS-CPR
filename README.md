# GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting
**[Changkun Liu](https://lck666666.github.io/),
[Shuai Chen](https://scholar.google.com/citations?user=c0xTh_YAAAAJ&hl=en), 
[Yash Bhalgat](https://scholar.google.com/citations?user=q0VSEHYAAAAJ&hl=en),
[Siyan Hu](https://scholar.google.com/citations?user=S56rLU4AAAAJ&hl=en), 
[Ming Cheng](https://scholar.google.com/citations?user=MPyUxv4AAAAJ&hl=en),
[Zirui Wang](https://scholar.google.com/citations?user=zCBKqa8AAAAJ&hl=en), 
[Victor Prisacariu](https://scholar.google.com/citations?user=GmWA-LoAAAAJ&hl=en) 
and [Tristan BRAUD](https://scholar.google.com/citations?user=ZOZtoQUAAAAJ&hl=en)**

**International Conference on Learning Representations (ICLR) 2025**

**[Project Page](https://xrim-lab.github.io/GS-CPR/) | [Paper](https://openreview.net/forum?id=mP7uV59iJM) | [Video](https://youtu.be/9xhcpLLu7Kg)**

[![GS-CPR](framework_imgs/Method.jpg)](https://openreview.net/forum?id=mP7uV59iJM)
[![GS-CPR_rel](framework_imgs/Method_rel.jpg)](https://openreview.net/forum?id=mP7uV59iJM)

## Installation
### ACT Scaffold-GS environment
We tested our code based on CUDA 12.1, PyTorch 2.5.1, and Python 3.11+.

Clone this repo:

```
git clone https://github.com/XRIM-Lab/GS-CPR.git
cd GS-CPR
```
### Install dependencies for ACT Scaffold-GS rendering

```
cd ACT_Scaffold_GS
conda create -n scaffold_act python=3.11
conda activate scaffold_act
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -r requirements.txt

# install Tiny-cuda-nn
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install depth rendering for 3DGS
git clone git@github.com:leo-frank/diff-gaussian-rasterization-depth.git
cd diff-gaussian-rasterization-depth
python setup.py install
```

### Pretrained 3DGS models, COLMAP files and ACT weights
You can download the pretrained 3DGS models from the provided [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliudg_connect_ust_hk/ElfOnz0vRm9Ot6j47CDzFaoBJrGKoqKGLfb6xYSuMwf7WQ?e=Rrc98i) and unzip them in the folder `GS-CPR/ACT_Scaffold_GS/data/`. You can download pretrained ACT MLP models from the provided [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliudg_connect_ust_hk/ElfOnz0vRm9Ot6j47CDzFaoBJrGKoqKGLfb6xYSuMwf7WQ?e=Rrc98i) and put them in the folder `GS-CPR/ACT_Scaffold_GS/logs/`.
```
ACT_Scaffold_GS
├── data
│   ├── cambridge
│   ├── 7scenes
│   ├── 12scenes
|   ├── Cambridge_semantic
├── logs
|   ├──paper_models
```


And then run the following command to render the synthetic images based on the `coarse_poses`.
```
# generate rendered images based on coarse poses for 7Scenes
bash script_render_pred_7s.sh
# generate rendered images based on coarse poses for 12Scenes
bash script_render_pred_12s.sh
```

For Cambridge Landmarks dataset, we also need calibrated camera intrinsics files for each image before rendering, it will be prepared in [Datasets](#datasets-raw-images--poses--intrinsics) section.
```
# generate rendered images based on coarse poses for Cambridge Landmarks
bash script_render_pred_cam.sh
```

NOTE: For 7scenes COLMAP files, we improve the accuracy of the [sparse point cloud](https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes) courtesy of Torsten Sattler, using rendered dense depth maps in [HLoc](https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes) tool box courtesy of Eric Brachmann for [DSAC*](https://github.com/vislearn/dsacstar). Then, we align all poses in `sparse/0/images.txt` to SfM poses in [ICCV 2021](https://github.com/tsattler/visloc_pseudo_gt_limitations). For 12scenes COLMAP files, we utilize SfM models provided by [ICCV 2021](https://github.com/tsattler/visloc_pseudo_gt_limitations). For Cambridge Landmarks, we use SfM models from [HLoc](https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/Cambridge) toolbox, courtesy of Torsten Sattler. **All these COLMAP files and 3DGS pretrained models have been prepared in the above download link.**

## Train Scaffold-GS models

If you want to train new Scaffold-GS models, you need COLMAP format `sparse/` file (please refer to the examples and data structure in the pretrained models). Additionally, when training new models, remember to comment out the line `if len(cam_infos) <= 1:` in `GS-CPR/ACT_Scaffold_GS/scene/dataset_readers.py` to ensure proper loading of camera info.
```python
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        #if len(cam_infos) <= 1:                #comment when training new 3DGS models
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
            sys.stdout.flush()
```

Run
```
#training Scaffold-gs models with ACT modules
bash train_scaffold_act.sh

#training Scaffold-gs models without ACT modules
bash train_scaffold.sh
```

## Install dependencies for GS-CPR refinement
Create the environment as same as [MASt3R](https://github.com/naver/mast3r#demo)
```
cd GS-CPR
conda activate mast3r
```

## Datasets (raw images + poses + intrinsics)
This paper uses three public datasets:
- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342/)
- [Stanford 12-Scenes](https://graphics.stanford.edu/projects/reloc/)

Following [ACE](https://github.com/nianticlabs/ace), we utilize the same scripts in the `datasets` folder to automatically download and extract the data in a consistent format.

> **Important: make sure you have checked the license terms of each dataset before using it.**

#### {7, 12}-Scenes:

You can use the `datasets/setup_{7,12}scenes.py` scripts to download the data.
As mentioned in our paper, we experimented _Pseudo Ground Truth (PGT)_ camera poses obtained after running SfM on the scenes (see the [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Brachmann_On_the_Limits_of_Pseudo_Ground_Truth_in_Visual_Camera_ICCV_2021_paper.html),
and [associated code](https://github.com/tsattler/visloc_pseudo_gt_limitations/) for details).

To download and prepare the datasets using the PGT poses:

```shell
cd datasets
# Downloads the data to datasets/pgt_7scenes_{chess, fire, ...}
./setup_7scenes.py --poses pgt
# Downloads the data to datasets/pgt_12scenes_{apt1_kitchen, ...}
./setup_12scenes.py --poses pgt
``` 
You can follow [ACE](https://github.com/nianticlabs/ace) to download DSLAM poses and try.

#### Cambridge Landmarks

Simply run:

```shell
cd datasets
# Downloads the data to datasets/Cambridge_{GreatCourt, KingsCollege, ...}
./setup_cambridge.py
```

## GS-CPR Refinement Evaluation
```
#For 7Scenes
python gs_cpr_7s.py --pose_estimator ace --scene chess #for a specific scene
python gs_cpr_7s.py --pose_estimator ace --test_all #for the whole dataset

#For 12Scenes
python gs_cpr_12s.py --pose_estimator ace --scene apt1_kitchen
python gs_cpr_12s.py --pose_estimator ace --test_all #for the whole dataset

#For Cambridge Landmarks
python gs_cpr_cam.py --pose_estimator ace --scene ShopFacade
python gs_cpr_cam.py --pose_estimator ace --test_all #for the whole dataset
```

## GS-CPR_rel Refinement Evaluation
```
#For 7Scenes
python gs_cpr_7s_rel.py --pose_estimator dfnet --scene chess #for a specific scene
python gs_cpr_7s_rel.py --pose_estimator dfnet --test_all #for the whole dataset

#For Cambridge Landmarks
python gs_cpr_cam_rel.py --pose_estimator dfnet --scene ShopFacade
python gs_cpr_cam_rel.py --pose_estimator dfnet --test_all #for the whole dataset
```

You can check the refined poses for each query in `txt` files and the statistic `log` results in `GS-CPR/outputs`.
## Citation
If you find our work helpful, please consider citing:

```bibtex
@inproceedings{
liu2025gscpr,
title={{GS}-{CPR}: Efficient Camera Pose Refinement via 3D Gaussian Splatting},
author={Changkun Liu and Shuai Chen and Yash Sanjay Bhalgat and Siyan HU and Ming Cheng and Zirui Wang and Victor Adrian Prisacariu and Tristan Braud},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=mP7uV59iJM}
}
```
## Star History
If you find our work helpful, please consider star this repo:
[![Star History Chart](https://api.star-history.com/svg?repos=XRIM-Lab/GS-CPR&type=Timeline)](https://www.star-history.com/#XRIM-Lab/GS-CPR&Timeline)

## Acknowledgements
This project is developed based on several fantastic repos: [Scaffold-GS](https://github.com/city-super/Scaffold-GS), [MASt3R](https://github.com/naver/mast3r), [DUSt3R](https://github.com/naver/dust3r), [NeFeS](https://github.com/ActiveVisionLab/NeFeS), [ACE](https://github.com/nianticlabs/ace) and [Depth for 3DGS](https://github.com/leo-frank/diff-gaussian-rasterization-depth). We thank the original authors for their excellent work.
