# SPA_CVPR2021
#### Xingjia Pan, Yingguo Gao, Zhiwen Lin, Fan Tang, Weiming Dong, Haolei Yuan, Feiyue Huang, Changsheng Xu
The official implementaion of SPA_CVPR2021 paper "[Unveiling the Potential of Structure-preserving for Weakly Supervised Object Localization](https://arxiv.org/abs/2103.04523)"

## Setup
1. Clone this repo:
   ~~~
   SPA_ROOT=/path/to/SPA
   git clone https://github.com/Panxjia/SPA_CVPR2021.git $SPA_ROOT
   cd $SPA_ROOT
   ~~~
2. Create an Anaconda enrironment with python>=3.6 and Pytorch=1.1.0
3.Download the images of ILSVRC2012 dataset and place them at $SPA_ROOT/data/

## Train and test
-Train
    ~~~
    cd scripts
    bash train_spa_ilsvrc.sh
    ~~~
-Test
  
  Download the pretrained model at GoogleDrive(https://drive.google.com/file/d/1Zs0uKmzkPz-zSanqAlxTUTZyENVE-RBl/view?usp=sharing)
    ~~~
    cd scripts
    bash val_spa_ilsvrc.sh
    ~~~



