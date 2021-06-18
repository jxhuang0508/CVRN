# Cross-View Regularization for Domain Adaptive Panoptic Segmentation

## Updates
- *06/2021*: check out our domain generalization paper [FSDR: Frequency Space Domain Randomization for Domain Generalization](https://arxiv.org/abs/2103.02370) (accepted to CVPR 2021). Inspired by the idea of JPEG that converts spatial images into multiple frequency components (FCs), we propose Frequency Space Domain Randomization (FSDR) that randomizes images in frequency space by keeping domain-invariant FCs (DIFs) and randomizing domain-variant FCs (DVFs) only. [Pytorch](https:xx) code and pre-trained models are coming soon.

## Paper
![](./figure_1(2).pdf)

[Cross-View Regularization for Domain Adaptive Panoptic Segmentation](https://arxiv.org/abs/2103.02584)  
[Jiaxing Huang](https://scholar.google.com/citations?user=czirNcwAAAAJ&hl=en&oi=ao),  [Dayan Guan](https://scholar.google.com/citations?user=9jp9QAsAAAAJ&hl=en), [Xiao Aoran](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en)
 
 School of Computer Science Engineering, Nanyang Technological University, Singapore
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021 (**Oral**)

If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2103.02584):

```
@article{huang2021cross,
  title={Cross-View Regularization for Domain Adaptive Panoptic Segmentation},
  author={Huang, Jiaxing and Guan, Dayan and Xiao, Aoran and Lu, Shijian},
  journal={arXiv preprint arXiv:2103.02584},
  year={2021}
}
```

# Install UPSNet
conda env create -f environment.yaml
git clone https://github.com/uber-research/UPSNet.git
cd UPSNet
sh init.sh
cp -r lib/dataset_devkit/panopticapi/panopticapi/ .

# Import Deeplab-v2
git clone https://github.com/yzou2/CRST.git

# Prepare Dataset (Download Cityscapes dataset at UPSNet/data/cityscapes)
cd UPSNet
sh init_cityscapes.sh
cd ..
python cvrn/init_citiscapes_19cls_to_16cls.py

# Prepare CVRN
cp cvrn/models/* UPSNet/upsnet/models
cp cvrn/dataset/* UPSNet/upsnet/dataset
cp cvrn/upsnet/* UPSNet/upsnet

# Evaluation
cd UPSNet
python upsnet/test_cvrn_upsnet.py --cfg ../config/cvrn_upsnet.yaml --weight_path ../pretrained_models/cvrn_upsnet.pth
2021-06-10 14:20:09,688 | base_dataset.py | line 499:           |    PQ     SQ     RQ     N
2021-06-10 14:20:09,688 | base_dataset.py | line 500: --------------------------------------
2021-06-10 14:20:09,688 | base_dataset.py | line 505: All       |  34.0   68.2   43.4    16
2021-06-10 14:20:09,688 | base_dataset.py | line 505: Things    |  27.9   73.6   37.3     6
2021-06-10 14:20:09,688 | base_dataset.py | line 505: Stuff     |  37.7   65.0   47.1    10

python upsnet/test_cvrn_pfpn.py --cfg ../config/cvrn_pfpn.yaml --weight_path ../pretrained_models/cvrn_pfpn.pth
2021-06-10 14:27:36,841 | base_dataset.py | line 361:           |    PQ     SQ     RQ     N
2021-06-10 14:27:36,842 | base_dataset.py | line 362: --------------------------------------
2021-06-10 14:27:36,842 | base_dataset.py | line 367: All       |  31.4   66.4   40.0    16
2021-06-10 14:27:36,842 | base_dataset.py | line 367: Things    |  20.7   68.1   28.2     6
2021-06-10 14:27:36,842 | base_dataset.py | line 367: Stuff     |  37.9   65.4   47.0    10

python upsnet/test_cvrn_psn.py --cfg ../config/cvrn_psn.yaml --weight_path ../pretrained_models/cvrn_psn_maskrcnn_branch.pth
2021-06-10 23:18:22,662 | test_cvrn_psn.py | line 240: combined pano result:
2021-06-10 23:20:32,259 | base_dataset.py | line 361:           |    PQ     SQ     RQ     N
2021-06-10 23:20:32,261 | base_dataset.py | line 362: --------------------------------------
2021-06-10 23:20:32,261 | base_dataset.py | line 367: All       |  32.1   66.6   41.1    16
2021-06-10 23:20:32,261 | base_dataset.py | line 367: Things    |  21.6   68.7   30.2     6
2021-06-10 23:20:32,261 | base_dataset.py | line 367: Stuff     |  38.4   65.3   47.6    10
