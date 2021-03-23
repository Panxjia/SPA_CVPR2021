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
3. Prepare the ILSVRC2012 dataset
   1. Download training and validation dataset and put them at $SPA_ROOT/data/ILSVRC
   2. Download [namelist](https://drive.google.com/drive/folders/1MRay8zKYHQmYiPEEdRCaYmGAaSudgSMY?usp=sharing) and put it at $SPA_ROOT/data/ILSVRC  
   The dataset is organised as:
   ~~~
   ILSVRC-
        -- Train
            -- n01440764
            -- ...
        -- val
            -- ILSVRC2012_val_00000001.JPEG
            -- ...
        -- list
   ~~~
4. Download the pretrained models [VGG16](https://drive.google.com/file/d/1OC8apFl2YphcCQ_4TkLNn92NxyNSqWT8/view?usp=sharing) and [Inception V3](https://drive.google.com/file/d/1saaTAMh1O8uE3AL34h1wnH9mR8XFJky0/view?usp=sharing), and place them at $SPA_ROOT/pretrained_models/

## Train and test
- Train
    ~~~
    cd scripts
    bash train_spa_ilsvrc.sh
    ~~~
- Test
  
  Download our models [VGG16](https://drive.google.com/file/d/1Zs0uKmzkPz-zSanqAlxTUTZyENVE-RBl/view?usp=sharing) and place it at $SPA_ROOT/snapshots/vgg_16_baseline_ilsvrc_20e_10_15d
    ~~~
    cd scripts
    bash val_spa_ilsvrc.sh
    ~~~

## Citation

If you find this project useful for your research, please use the following BibTeX entry.
```
@article{pan2021unveiling,
  title={Unveiling the Potential of Structure Preserving for Weakly Supervised Object Localization},
  author={Xingjia Pan and Yingguo Gao and Zhiwen Lin and Fan Tang and Weiming Dong and Haolei Yuan and Feiyue Huang and Changsheng Xu},
  booktitle={CVPR},
  pages={1--8},
  year={2021}
}
```
## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.  
Xingjia Pan: xjia.pan@gmail.com  
Fan Tang: tfan.108@gmail.com

